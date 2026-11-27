"""Real-rig glue: camera pose from the marker map, the ball tracker on RealSense frames, and the intercept planner.

The planner is the sim's InterceptPolicy logic (straight-line crossing prediction at the intercept
line, jaw facing the ball, rate limits on the jaw pose, IK in the MuJoCo model) driven by the tracker's
estimate and wall-clock time instead of sim state. Heights are relative to the MEASURED table plane
(config/rig_map.json, in the arm's own kinematic frame). The line, the roll direction and the jaw's
heading are in table axes (rig_calib.TableFrame, from the map: how the arm base is turned on the table); the
tracker's estimate is turned into them, the jaw target back into the arm frame for the IK.
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
TRACKED = ((0.12, 1.00), (-0.36, 0.38))      # tracked region of the real table, table axes: x down it, y across (m)


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


def make_tracker(calib: CameraCalib, rig_map: rig_calib.RigMap, hsv=None) -> BallTracker:
    """Ball tracker for a camera with a known pose, projecting onto the map's table plane and looking
    for the ball on the table as the map places it (TRACKED, in the map's table axes)."""
    lo, hi = hsv or load_hsv()
    return BallTracker(on_table(calib, rig_map.plane), ball_r=BALL_R, hsv_lo=lo, hsv_hi=hi,
                       table_xy=rig_map.frame.arm_box(*TRACKED), roi_half=50, bgr=True)


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
    jaw: np.ndarray          # fingertip target (x, y, z), arm frame
    yaw: float               # jaw heading in table axes (0 = facing up the table, toward the ball)
    crossing_y: float | None # predicted crossing, table axes (m along the intercept line)
    t_cross: float | None    # seconds until the ball reaches the line (None: it would stop before the line)
    ik_err: float
    phase: str               # "home" | "tracking" | "arrived"
    jaw_y: float = 0.0       # the jaw's position along the line, table axes


class InterceptPlanner:
    """Sim InterceptPolicy, real-time version. Call `plan(tracker, now)` every control tick.
    Works in table axes (x down the table, the ball rolls toward -x); the IK gets arm-frame targets.

    The ball: the tracker's estimate, extended along its straight path to the intercept line (the
    crossing point), with the measured slowdown of a rolling ball for the arrival time (crossing()).

    Speed: the jaw's command (position along the line, heading, hover height) advances at the task
    limits, and the step is shortened whenever a joint would have to move faster than
    `max_joint_speed`. Every target is therefore the IK solution of a point on the intercept line at
    the planned height: the arm never cuts through other poses to catch up. The arm's own speed limit
    (Arm.speed / the firmware's) is a backstop set above this one."""

    def __init__(self, plane, frame: rig_calib.TableFrame, mode: str = "forward",
                 max_lateral_speed: float = 0.6, max_yaw_rate: float = 1.5,
                 max_y: float = 0.30, max_yaw: float = 0.6, hover: float = 0.015, jaw_height: float | None = None,
                 max_joint_speed: float = 3.0, max_hover_speed: float = 0.15):
        """jaw_height: fingertips above the table at the catch (m); None = the sim's (20 mm, forward mode).
        max_y: reach along the line (+-0.30 m is reachable at catch height with any jaw heading).
        plane, frame: the table plane and the table axes of the rig map (RigMap.plane, RigMap.frame)."""
        self.env = BallInterceptEnv(mode, jaw_height=jaw_height)
        self.plane, self.frame = plane, frame
        self.gate_x, self.jaw_height = self.env.gate_x, self.env.jaw_height
        self.v_lat, self.w_yaw, self.v_hover, self.w_joint = max_lateral_speed, max_yaw_rate, max_hover_speed, max_joint_speed
        self.max_y, self.max_yaw, self.hover = max_y, max_yaw, hover
        self.turn = frame.turn
        self.q_home = home_pose(self.env, plane, frame, hover)
        # Joint 1's share of the base's turn (its sign as the home pose found it). Subtracting it from q
        # shows the arm in the sim scene with the table in front, as if the base were not turned.
        self.j1_turn = float(np.sign(self.q_home[0] - self.env.q_home[0])) * abs(self.turn)
        self.reset()

    def reset(self):
        """The start state: the jaw at the middle of the line, hovering. Call it when the arm IS there
        (after the move to plan().q); to bring a jaw back from elsewhere use recentre()."""
        self.q_cmd = self.q_home.copy()
        self.y_cmd, self.yaw_cmd, self.hover_cmd, self.t_last = 0.0, 0.0, self.hover, None
        self.recentring = False
        self.env.data.qpos[self.env.arm_qadr] = self.q_home

    def recentre(self):
        """Go back to the middle of the line, at the planner's speeds."""
        self.recentring = True

    def home_jaw(self) -> np.ndarray:
        return self.jaw_target(0.0, self.hover)

    def jaw_target(self, y, hover):
        """Arm-frame fingertip target on the intercept line at table y, `hover` above the catch height."""
        return self.frame.line_point(self.plane, y, self.jaw_height + hover, self.gate_x)

    def line_point(self, y, dz: float = 0.0) -> np.ndarray:
        """Arm-frame point of the intercept line at table y, `dz` above the table (for drawing)."""
        return self.frame.line_point(self.plane, y, dz, self.gate_x)

    def _advance(self, y_goal: float, yaw_goal: float, hover_goal: float, dt: float) -> None:
        """Move the command toward the goals: task-space rate limits, then a shorter step if a joint
        would exceed max_joint_speed. The command and q_cmd always agree."""
        if dt <= 0.0:
            return
        step = np.array([np.clip(y_goal - self.y_cmd, -self.v_lat * dt, self.v_lat * dt),
                         np.clip(yaw_goal - self.yaw_cmd, -self.w_yaw * dt, self.w_yaw * dt),
                         np.clip(hover_goal - self.hover_cmd, -self.v_hover * dt, self.v_hover * dt)])
        if not np.any(step):
            return
        cmd0, limit, frac = np.array([self.y_cmd, self.yaw_cmd, self.hover_cmd]), self.w_joint * dt, 1.0
        for _ in range(4):
            y, yaw, hover = cmd0 + frac * step
            q = self.env.ik_jaw(*self.jaw_target(y, hover), yaw=yaw + self.turn, q0=self.q_cmd, iters=40)
            worst = float(np.abs(q - self.q_cmd).max())
            if worst <= limit * 1.02:
                self.y_cmd, self.yaw_cmd, self.hover_cmd, self.q_cmd = float(y), float(yaw), float(hover), q
                return
            frac *= 0.9 * limit / worst
        # no step along the line is slow enough (the IK changed branch): stay; the command keeps its pose

    def plan(self, trk: BallTracker, now: float) -> Plan:
        dt = 0.0 if self.t_last is None else min(now - self.t_last, 0.1)
        self.t_last = now
        st = trk.state_at(now) if trk.ready else None
        cross, t_cross, phase = None, None, "home"
        y_goal, yaw_goal, hover_goal = self.y_cmd, self.yaw_cmd, self.hover
        vel = self.frame.table_xy(st.vel) if st is not None else None
        if st is not None and vel[0] < -0.05:                 # a tracked ball moving toward the arm
            pos = self.frame.table_xy(st.pos)
            dist = float(pos[0]) - self.gate_x
            if dist < -0.05:
                phase = "arrived"
            else:
                phase, self.recentring = "tracking", False
                cross, t_cross = crossing(pos, vel, self.gate_x)
                y_goal = float(np.clip(cross, -self.max_y, self.max_y))
                yaw_goal = float(np.clip(np.arctan2(-vel[1], -vel[0]), -self.max_yaw, self.max_yaw))
                hover_goal = self.hover * float(np.clip((dist - 0.10) / 0.20, 0.0, 1.0))
        if phase == "home" and self.recentring:
            y_goal, yaw_goal = 0.0, 0.0
            self.recentring = abs(self.y_cmd) > 1e-4 or abs(self.yaw_cmd) > 1e-4
        if phase != "arrived":
            self._advance(y_goal, yaw_goal, hover_goal, dt)
        jaw = self.jaw_target(self.y_cmd, self.hover_cmd)
        err = self.env.jaw_error(self.q_cmd, *jaw)
        return Plan(self.q_cmd.copy(), jaw, self.yaw_cmd, cross, t_cross, err, phase, self.y_cmd)


BALL_DECEL = 0.20            # m/s^2: speed a rolling ball loses on this table (27 Nov 2026: 0.18 .. 0.26 over 31 rolls)


def crossing(pos, vel, x_line: float, decel: float = BALL_DECEL) -> tuple[float, float | None]:
    """A ball at `pos` moving with `vel` toward -x (table axes): (y where its straight path crosses
    x = x_line, seconds until then). The path is the straight line of the tracker's estimate; the
    time counts the ball's slowdown (`decel` m/s^2 along the path; 0 = constant speed). The time is
    None when the ball would come to rest before the line: the y is still where its path crosses."""
    gap = float(pos[0]) - x_line
    y = float(pos[1] + vel[1] * gap / -vel[0])
    if gap <= 0.0:
        return y, 0.0
    speed = float(np.hypot(vel[0], vel[1]))
    d = gap * speed / -float(vel[0])                          # distance to the line along the path
    if decel <= 0.0:
        return y, d / speed
    if speed * speed < 2.0 * decel * d:
        return y, None
    return y, (speed - float(np.sqrt(speed * speed - 2.0 * decel * d))) / decel


def home_pose(env: BallInterceptEnv, plane, frame: rig_calib.TableFrame, dz: float = 0.0) -> np.ndarray:
    """Joint pose with the open jaw at the middle of the intercept line, facing up the table, `dz` above
    the jaw height: the sim's home pose turned at joint 1 by the base's turn, then refined by IK."""
    jaw = frame.line_point(plane, 0.0, env.jaw_height + dz, env.gate_x)
    best, best_err = env.q_home.copy(), np.inf
    for sign in (1.0, -1.0):                     # joint 1's sign in the model: keep the turn that reaches the line
        seed = env.q_home.copy()
        seed[0] = float(np.clip(seed[0] + sign * frame.turn, env.arm_lo[0], env.arm_hi[0]))
        q = env.ik_jaw(*jaw, yaw=frame.turn, q0=seed, iters=400)
        err = env.jaw_error(q, *jaw)
        if err < best_err:
            best, best_err = q, err
    return best
