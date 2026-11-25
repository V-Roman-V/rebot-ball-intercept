"""Conventional-CV ball tracker for a fixed external camera looking at the table.

No learned detector. Per frame:
    HSV colour threshold -> morphological open -> connected components -> pick the blob
    with the right size/roundness nearest the prediction -> centroid -> undistort the point
    -> back-project onto the plane of the ball centre (z = r) -> Kalman filter (x, y, vx, vy).

The same class is meant to run on the real camera: only the calibration (K, dist, rvec,
tvec) and the HSV thresholds change. In the sim, `camera_from_mujoco` gives the exact
calibration of a MuJoCo camera; on the real rig use `calibrate_extrinsics` (solvePnP on
table points with known robot-frame coordinates) after a checkerboard intrinsic calibration.

Coordinates: robot/world frame, z = 0 is the tabletop, the ball centre is at z = r.
Images are RGB (what mujoco.Renderer returns); pass `bgr=True` for OpenCV camera frames.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np


# ----------------------------------------------------------------------------- calibration
@dataclass
class CameraCalib:
    K: np.ndarray           # 3x3 intrinsics
    dist: np.ndarray        # OpenCV distortion coefficients (zeros for an ideal pinhole)
    rvec: np.ndarray        # world -> camera rotation (Rodrigues), OpenCV convention
    tvec: np.ndarray        # world -> camera translation
    width: int
    height: int

    @property
    def R(self) -> np.ndarray:
        return cv2.Rodrigues(self.rvec)[0]

    def project(self, pts_world: np.ndarray) -> np.ndarray:
        """World points (N, 3) -> pixel coordinates (N, 2), with distortion."""
        pts = np.asarray(pts_world, np.float64).reshape(-1, 1, 3)
        uv, _ = cv2.projectPoints(pts, self.rvec, self.tvec, self.K, self.dist)
        return uv.reshape(-1, 2)

    def plane_homography(self, z: float) -> np.ndarray:
        """3x3 H mapping world (x, y, 1) on the plane at height z to undistorted pixels."""
        R, t = self.R, self.tvec.reshape(3)
        return self.K @ np.column_stack([R[:, 0], R[:, 1], R[:, 2] * z + t])


def camera_from_mujoco(model, data, cam_name: str, width: int, height: int) -> CameraCalib:
    """Exact pinhole calibration of a MuJoCo camera (needs mj_forward / mj_kinematics first).
    MuJoCo cameras look along -z with +y up; OpenCV looks along +z with +y down."""
    import mujoco  # noqa: F401  (only needed in the sim)
    cid = model.camera(cam_name).id
    fovy = np.radians(model.cam_fovy[cid])
    f = 0.5 * height / np.tan(0.5 * fovy)
    K = np.array([[f, 0, 0.5 * width], [0, f, 0.5 * height], [0, 0, 1.0]])
    R_wc = data.cam_xmat[cid].reshape(3, 3)            # camera axes in world (columns)
    R_cv = np.diag([1.0, -1.0, -1.0]) @ R_wc.T          # world -> OpenCV camera
    t = -R_cv @ data.cam_xpos[cid]
    return CameraCalib(K, np.zeros(5), cv2.Rodrigues(R_cv)[0].reshape(3), t, width, height)


def calibrate_extrinsics(K: np.ndarray, dist: np.ndarray, img_pts: np.ndarray, world_pts: np.ndarray,
                         width: int, height: int) -> CameraCalib:
    """Real rig: camera pose from >= 4 table points with known robot-frame coordinates (e.g. ArUco
    marker centres measured by touching them with the gripper tip) and their pixel locations."""
    ok, rvec, tvec = cv2.solvePnP(np.asarray(world_pts, np.float64), np.asarray(img_pts, np.float64), K, dist)
    assert ok, "solvePnP failed"
    return CameraCalib(K, np.asarray(dist, float), rvec.reshape(3), tvec.reshape(3), width, height)


# ------------------------------------------------------------------------------- tracker
@dataclass
class Detection:
    uv: np.ndarray          # blob centroid, pixels
    xy: np.ndarray          # ball centre on the z = r plane, world metres
    area: int
    radius_px: float


@dataclass
class TrackState:
    t: float                # timestamp of the last filter update (s)
    pos: np.ndarray         # (x, y) m
    vel: np.ndarray         # (vx, vy) m/s
    n_meas: int             # accepted measurements so far
    lost_for: float         # seconds since the last accepted measurement


class BallTracker:
    def __init__(self, calib: CameraCalib, ball_r: float = 0.022,
                 hsv_lo=(0, 110, 60), hsv_hi=(22, 255, 255),
                 table_xy=((-0.15, 1.05), (-0.35, 0.35)), roi_half: int = 60,
                 accel_noise: float = 2.0, meas_std: float = 0.003, gate_chi2: float = 16.0,
                 min_meas: int = 4, bgr: bool = False):
        self.calib, self.r = calib, ball_r
        self.lo, self.hi = np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8)
        self.bgr = bgr
        self.Hinv = np.linalg.inv(calib.plane_homography(ball_r))   # pixels -> (x, y) on z = r
        self.kernel = np.ones((3, 3), np.uint8)
        self.roi_half = roi_half
        self.q, self.R_meas = accel_noise ** 2, np.eye(2) * meas_std ** 2
        self.gate_chi2, self.min_meas = gate_chi2, min_meas
        self.table_xy = table_xy
        # Pixel mask of the tabletop (only search there) and expected blob area per table point.
        (x0, x1), (y0, y1) = table_xy
        corners = calib.project(np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]]))
        self.table_mask = np.zeros((calib.height, calib.width), np.uint8)
        cv2.fillConvexPoly(self.table_mask, corners.astype(np.int32), 255)
        self.reset()
        self.timing_ms: list[float] = []

    def reset(self):
        self.x = None           # [x, y, vx, vy]
        self.P = None
        self.t = None
        self.n_meas = 0
        self.t_last_meas = None
        self.last_det: Detection | None = None
        self.last_mask: np.ndarray | None = None
        self.last_roi = None
        self.history: list[tuple[float, np.ndarray]] = []   # accepted measurements (t, xy)

    # -- geometry helpers
    def pixel_to_world(self, uv) -> np.ndarray:
        p = cv2.undistortPoints(np.asarray(uv, np.float64).reshape(1, 1, 2), self.calib.K, self.calib.dist,
                                P=self.calib.K).reshape(2)
        w = self.Hinv @ np.array([p[0], p[1], 1.0])
        return w[:2] / w[2]

    def world_to_pixel(self, xy, z: float | None = None) -> np.ndarray:
        xy = np.atleast_2d(xy)
        zz = np.full((len(xy), 1), self.r if z is None else z)
        return self.calib.project(np.hstack([xy, zz]))

    def expected_radius_px(self, xy) -> float:
        """Apparent ball radius (pixels) at table position xy, from the calibration."""
        # f * r / distance, for any camera direction. (An offset of r along world x was used before:
        # right for a camera looking across the table, far too small for one looking along it.)
        p_cam = self.calib.R @ np.array([xy[0], xy[1], self.r]) + self.calib.tvec.reshape(3)
        return float(self.calib.K[0, 0] * self.r / np.linalg.norm(p_cam))

    # -- detection
    def detect(self, img: np.ndarray, roi_center_uv=None) -> Detection | None:
        h, w = img.shape[:2]
        if roi_center_uv is not None:
            u, v = int(roi_center_uv[0]), int(roi_center_uv[1])
            x0, y0 = max(0, u - self.roi_half), max(0, v - self.roi_half)
            x1, y1 = min(w, u + self.roi_half), min(h, v + self.roi_half)
            if x1 - x0 < 8 or y1 - y0 < 8:
                x0, y0, x1, y1 = 0, 0, w, h
        else:
            x0, y0, x1, y1 = 0, 0, w, h
        self.last_roi = (x0, y0, x1, y1)
        crop = img[y0:y1, x0:x1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV if self.bgr else cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.lo, self.hi)
        mask &= self.table_mask[y0:y1, x0:x1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        self.last_mask = (mask, (x0, y0))
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
        best, best_score = None, np.inf
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            uv = cents[i] + (x0, y0)
            xy = self.pixel_to_world(uv)
            r_exp = self.expected_radius_px(xy)
            a_exp = np.pi * r_exp ** 2
            # size: 0.3x .. 2x the expected area (partial occlusion shrinks it); shape: roughly square box
            if not (0.3 * a_exp < area < 2.0 * a_exp) or max(bw, bh) > 2.5 * min(bw, bh):
                continue
            score = abs(np.log(area / a_exp))
            if self.x is not None:   # prefer the blob nearest the prediction
                score += np.linalg.norm(xy - self.x[:2]) / 0.05
            if score < best_score:
                best, best_score = Detection(uv, xy, area, float(np.sqrt(area / np.pi))), score
        return best

    # -- filter
    def _predict(self, t: float):
        dt = t - self.t
        if dt <= 0:
            return
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        G = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * self.q
        self.t = t

    def update(self, z: np.ndarray | None, t: float) -> bool:
        """Fuse one measurement (or None = no detection) taken at time t. Returns True if accepted."""
        if self.x is None:
            if z is None:
                return False
            self.x = np.r_[z, 0.0, 0.0]
            self.P = np.diag([1e-5, 1e-5, 1.0, 1.0])
            self.t, self.t_last_meas, self.n_meas = t, t, 1
            self.history.append((t, z.copy()))
            return True
        self._predict(t)
        if z is None:
            return False
        H = np.eye(2, 4)
        S = H @ self.P @ H.T + self.R_meas
        y = z - H @ self.x
        if self.n_meas >= self.min_meas and y @ np.linalg.solve(S, y) > self.gate_chi2:
            return False   # outlier
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.n_meas += 1
        self.t_last_meas = t
        self.history.append((t, z.copy()))
        return True

    def process(self, img: np.ndarray, t: float) -> Detection | None:
        """Detect in one frame captured at time t and update the filter."""
        t0 = time.perf_counter()
        roi = None
        if self.x is not None and self.t_last_meas is not None and t - self.t_last_meas < 0.1:
            dt = t - self.t
            roi = self.world_to_pixel(self.x[:2] + self.x[2:] * dt)[0]
        det = self.detect(img, roi)
        if det is None and roi is not None:          # lost it in the window: search the whole table
            det = self.detect(img, None)
        accepted = self.update(None if det is None else det.xy, t)
        self.last_det = det if accepted else None
        self.timing_ms.append(1000 * (time.perf_counter() - t0))
        return self.last_det

    # -- outputs
    @property
    def ready(self) -> bool:
        return self.x is not None and self.n_meas >= self.min_meas

    def state_at(self, t: float) -> TrackState | None:
        """Filter estimate extrapolated (constant velocity) to time t, e.g. now = capture + latency."""
        if self.x is None:
            return None
        dt = t - self.t
        return TrackState(t, self.x[:2] + self.x[2:] * dt, self.x[2:].copy(), self.n_meas,
                          t - (self.t_last_meas if self.t_last_meas is not None else t))

    def predict_crossing(self, gate_x: float, t: float | None = None):
        """(y, time) where the ball's straight-line path crosses x = gate_x, or None."""
        if not self.ready or self.x[2] > -0.05:
            return None
        x, y, vx, vy = self.x
        dt = (gate_x - x) / vx
        return float(y + vy * dt), float(self.t + dt)
