"""Real-rig glue: camera pose from the marker map, the ball tracker on RealSense frames, and the intercept planner.

The planner is the sim's InterceptPolicy logic (straight-line crossing prediction at the intercept
line, jaw facing the ball, rate limits on the jaw pose, IK in the MuJoCo model) driven by the tracker's
estimate and wall-clock time instead of sim state. Heights are relative to the MEASURED table plane
(config/rig_map.json, in the arm's own kinematic frame).
It only computes targets; whether they are sent to the motors is the caller's business.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import rig_calib
from .ball_intercept_env import BallInterceptEnv
from .ball_tracker import BallTracker, CameraCalib

ROOT = Path(__file__).resolve().parents[1]
CAL_DIR = ROOT / "outputs" / "calibration"
BALL_R = 0.022
DEFAULT_HSV = ((5, 120, 120), (25, 255, 255))
# Tracked region of the real table: x down the table, y across it (m), turned into the arm frame
# with the base's turn on the table (rig_calib.BASE_YAW_DEG).
TABLE_XY = rig_calib.arm_box((0.12, 1.00), (-0.36, 0.38))


def table_z(plane, x, y) -> float:
    a, b, c = plane
    return float(a + b * x + c * y)


def camera_calib(cam, res: rig_calib.Reloc) -> CameraCalib:
    """Calibration of the live camera `cam` (K, dist, size) with the pose from rig_calib.locate()."""
    return CameraCalib(cam.K, cam.dist, res.rvec, res.tvec, cam.width, cam.height)


def locate_camera(cam, rig_map: rig_calib.RigMap, n_frames: int = 30, timeout: float = 6.0) -> rig_calib.Reloc:
    """Camera pose in the arm frame from the table markers, on `n_frames` fresh frames of `cam`.
    Raises rig_calib.CalibrationError with a message for the user when the markers do not allow it."""
    images, fn, t_end = [], None, time.monotonic() + timeout
    while len(images) < n_frames and time.monotonic() < t_end:
        f = cam.read(timeout=2.0, newer_than=fn)
        fn = f.frame_number
        images.append(f.img)
    res = rig_calib.locate(rig_map, cam.K, cam.dist, images)
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    (CAL_DIR / "camera_pose.json").write_text(json.dumps(
        rig_calib.pose_dict(res, rig_map, cam.K, cam.dist, cam.width, cam.height, getattr(cam, "serial", None)), indent=2))
    return res


def on_table(calib: CameraCalib, plane) -> CameraCalib:
    """Same camera, with the world frame laid in the table plane: z' = 0 is the (tilted) table and
    z' = r the plane of the ball centre, on the camera's side. In-plane coordinates equal base-frame
    x, y to better than 0.05 mm for the tilts seen here, so the tracker's output is used as base x, y."""
    o, e1, e2, n = rig_calib.plane_frame(plane)
    if n @ (rig_calib.camera_centre(calib.rvec, calib.tvec) - o) <= BALL_R:
        raise rig_calib.CalibrationError("the camera is not above the table plane")
    R = calib.R @ np.column_stack([e1, e2, n])
    return CameraCalib(calib.K, calib.dist, cv2.Rodrigues(R)[0].reshape(3), calib.R @ o + calib.tvec.reshape(3),
                       calib.width, calib.height)


def load_hsv():
    """Ball colour range (HSV lo, hi) saved by ./rig calibrate (step 5), else a default for an orange ball."""
    p = CAL_DIR / "ball_hsv.json"
    if p.exists():
        d = json.loads(p.read_text())
        return tuple(d["lo"]), tuple(d["hi"])
    return DEFAULT_HSV


def hsv_saved_time() -> str | None:
    """When the ball colour was saved, or None if it never was."""
    p = CAL_DIR / "ball_hsv.json"
    return json.loads(p.read_text()).get("time") if p.exists() else None


def save_hsv(lo, hi):
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    (CAL_DIR / "ball_hsv.json").write_text(json.dumps({"lo": list(map(int, lo)), "hi": list(map(int, hi)),
                                                       "time": time.strftime("%Y-%m-%d %H:%M:%S")}))


def make_tracker(calib: CameraCalib, plane, hsv=None) -> BallTracker:
    """Ball tracker for a camera with a known pose, projecting onto the measured table plane."""
    lo, hi = hsv or load_hsv()
    return BallTracker(on_table(calib, plane), ball_r=BALL_R, hsv_lo=lo, hsv_hi=hi, table_xy=TABLE_XY, roi_half=50, bgr=True)


class HostClock:
    """Maps camera hardware timestamps (ms) to time.monotonic(): offset = lower envelope of
    (arrival - hw), so frame intervals come from the camera clock and transport jitter is removed."""

    def __init__(self):
        self.off = None

    def __call__(self, hw_ms: float, t_host: float) -> float:
        o = t_host - hw_ms / 1000.0
        self.off = o if self.off is None else min(self.off + 1e-5, o)   # slow upward drift allowance
        return hw_ms / 1000.0 + self.off


def auto_hsv(img_bgr, table_mask=None):
    """HSV range from the single orange-ish blob in view (ball resting on the table), or None."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 70, 70), (35, 255, 255))
    if table_mask is not None:
        m &= table_mask
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    cands = [i for i in range(1, n) if st[i, 4] >= 30 and max(st[i, 2], st[i, 3]) <= 2 * min(st[i, 2], st[i, 3])]
    if len(cands) != 1:
        return None
    px = hsv[lab == cands[0]]
    h = np.median(px[:, 0])
    lo = (int(max(0, h - 9)), int(max(60, np.percentile(px[:, 1], 5) - 35)), int(max(60, np.percentile(px[:, 2], 5) - 45)))
    hi = (int(min(179, h + 9)), 255, 255)
    return lo, hi


def hsv_from_selection(img_bgr, x0: int, y0: int, x1: int, y1: int):
    """HSV range of the ball selected with the mouse: the pixels in the ellipse inside the box
    (x0, y0)-(x1, y1), image pixels. Same margins as auto_hsv. None if the box holds no pixel."""
    x0, x1 = sorted((int(x0), int(x1)))
    y0, y1 = sorted((int(y0), int(y1)))
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, img_bgr.shape[1] - 1), min(y1, img_bgr.shape[0] - 1)
    if x1 < x0 or y1 < y0:
        return None
    hsv = cv2.cvtColor(img_bgr[y0:y1 + 1, x0:x1 + 1], cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    inside = ((xx + 0.5 - w / 2) / (w / 2)) ** 2 + ((yy + 0.5 - h / 2) / (h / 2)) ** 2 <= 1.0
    px = hsv[inside] if inside.any() else hsv.reshape(-1, 3)
    px = px[px[:, 1] >= np.percentile(px[:, 1], 25)]          # drop the greyest quarter: background, highlight
    hue = np.median(px[:, 0])
    spread = max(9.0, float(np.percentile(np.abs(px[:, 0].astype(float) - hue), 90)) + 3)
    lo = (int(max(0, hue - spread)), int(max(40, np.percentile(px[:, 1], 5) - 35)), int(max(40, np.percentile(px[:, 2], 5) - 45)))
    hi = (int(min(179, hue + spread)), 255, 255)
    return lo, hi


@dataclass
class Plan:
    q: np.ndarray            # arm joint targets (rad)
    jaw: np.ndarray          # fingertip target (x, y, z) robot frame
    yaw: float
    crossing_y: float | None
    t_cross: float | None    # seconds until the ball reaches the line
    ik_err: float
    phase: str               # "home" | "tracking" | "arrived"


class InterceptPlanner:
    """Sim InterceptPolicy, real-time version. Call `plan(tracker, now)` every control tick."""

    def __init__(self, plane, mode: str = "forward", max_lateral_speed: float = 0.6, max_yaw_rate: float = 1.5,
                 max_y: float = 0.24, max_yaw: float = 0.6, hover: float = 0.015):
        self.env = BallInterceptEnv(mode)
        self.plane = plane
        self.gate_x, self.jaw_height = self.env.gate_x, self.env.jaw_height
        self.v_lat, self.w_yaw = max_lateral_speed, max_yaw_rate
        self.max_y, self.max_yaw, self.hover = max_y, max_yaw, hover
        self.q_home = self.env.q_home.copy()
        self.reset()

    def reset(self):
        self.q_cmd = self.q_home.copy()
        self.y_cmd, self.yaw_cmd, self.t_last = 0.0, 0.0, None
        self.env.data.qpos[self.env.arm_qadr] = self.q_home

    def home_jaw(self) -> np.ndarray:
        return self.jaw_target(0.0, 0.0, far=True)

    def jaw_target(self, y, dist, far=False):
        z_tab = table_z(self.plane, self.gate_x, y)
        hover = self.hover if far else self.hover * float(np.clip((dist - 0.10) / 0.20, 0.0, 1.0))
        return np.array([self.gate_x, y, z_tab + self.jaw_height + hover])

    def plan(self, trk: BallTracker, now: float) -> Plan:
        dt = 0.0 if self.t_last is None else min(now - self.t_last, 0.1)
        self.t_last = now
        st = trk.state_at(now) if trk.ready else None
        cross, t_cross, phase = None, None, "home"
        y_goal, yaw_goal, dist = self.y_cmd, self.yaw_cmd, 1.0
        if st is not None and st.vel[0] < -0.05:
            x, y = st.pos
            vx, vy = st.vel
            dist = x - self.gate_x
            if dist < -0.05:
                phase = "arrived"
            else:
                phase = "tracking"
                t_cross = max(dist / -vx, 0.0)
                cross = float(y + vy * t_cross)
                y_goal = float(np.clip(cross, -self.max_y, self.max_y))
                yaw_goal = float(np.clip(np.arctan2(-vy, -vx), -self.max_yaw, self.max_yaw))
        if phase != "arrived":
            self.y_cmd += float(np.clip(y_goal - self.y_cmd, -self.v_lat * dt, self.v_lat * dt))
            self.yaw_cmd += float(np.clip(yaw_goal - self.yaw_cmd, -self.w_yaw * dt, self.w_yaw * dt))
            jaw = self.jaw_target(self.y_cmd, dist, far=(phase == "home"))
            self.q_cmd = self.env.ik_jaw(*jaw, yaw=self.yaw_cmd, q0=self.q_cmd, iters=40)
        jaw = self.jaw_target(self.y_cmd, dist, far=(phase == "home"))
        err = self.env.jaw_error(self.q_cmd, *jaw)
        return Plan(self.q_cmd.copy(), jaw, self.yaw_cmd, cross, t_cross, err, phase)
