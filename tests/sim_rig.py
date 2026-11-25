"""A simulated rig for running the calibration scripts without hardware: a fake camera that renders
real ArUco markers lying on a (slightly tilted) table, and a fake unpowered arm."""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from eaiproject import markers
from eaiproject import rig_calib as rc
from eaiproject.arm import ArmSample

TEX = 2000.0                                   # texture pixels per metre
X0, Y1 = -0.05, 0.50                           # texture origin: world (X0, Y1) is texture pixel (0, 0)
TEX_W, TEX_H = int(1.15 * TEX), int(1.0 * TEX)
LAYOUT = {"107": (0.30, -0.21, 0.3), "880": (0.33, 0.20, -1.2),      # near the arm
          "467": (0.72, -0.18, 2.0), "718": (0.86, 0.16, 0.6)}       # mid-table


def look_at(eye, target):
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z])
    return cv2.Rodrigues(R)[0].reshape(3), -R @ np.asarray(eye, float)


def intrinsics(width: int, height: int) -> np.ndarray:
    f = 910.0 * width / 1280
    return np.array([[f, 0, width / 2 - 0.5], [0, f, height / 2 - 0.5], [0, 0, 1.0]])


@dataclass
class Frame:
    img: np.ndarray
    frame_number: int
    hw_ts_ms: float
    t_host: float


class World:
    def __init__(self, plane=(0.012, 0.003, -0.002)):
        self.plane = np.array(plane)
        self.truth = {k: rc.marker_points(self.plane, pose, markers.SIZES[k]) for k, pose in LAYOUT.items()}
        self.texture = np.full((TEX_H, TEX_W, 3), 205, np.uint8)
        for k, P in self.truth.items():
            m = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(markers.DICT), int(k), 280)
            m = cv2.copyMakeBorder(m, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)           # quiet zone
            n = m.shape[0]
            src = np.float32([[39.5, 39.5], [n - 40.5, 39.5], [n - 40.5, n - 40.5], [39.5, n - 40.5]])
            H = cv2.getPerspectiveTransform(src, self.tex(P[:4, :2]).astype(np.float32))
            warped = cv2.warpPerspective(m, H, (TEX_W, TEX_H), flags=cv2.INTER_AREA, borderValue=0)
            mask = cv2.warpPerspective(np.full_like(m, 255), H, (TEX_W, TEX_H))
            self.texture[mask > 0] = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)[mask > 0]
        self.ball = None                           # (x, y) of the ball on the table, or None
        self.tip = None                            # physical fingertip position to draw the arm's fingers at, or None

    def tex(self, xy) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        return np.c_[(xy[:, 0] - X0) * TEX - 0.5, (Y1 - xy[:, 1]) * TEX - 0.5]      # pixel-centre coordinates

    def on_table(self, xy, h=0.0) -> np.ndarray:
        return rc.from_plane(self.plane, rc.to_plane(self.plane, np.c_[np.atleast_2d(xy), [[0.0]] * len(np.atleast_2d(xy))])[0], h)


class FakeCamera:
    def __init__(self, world: World, width=1280, height=720):
        self.world, self.width, self.height, self.fps = world, width, height, 30
        self.K, self.dist, self.serial = intrinsics(width, height), np.zeros(5), "SIM"
        self.rvec, self.tvec = look_at(np.array([1.45, 0.25, 0.75]), [0.55, 0.0, 0.0])
        self.n = 0
        self.rng = np.random.default_rng(0)

    def set_pose(self, eye, target=(0.55, 0.0, 0.0)):
        self.rvec, self.tvec = look_at(np.asarray(eye, float), target)

    SS = 3                                          # supersampling: pixels integrate light, as a sensor does

    def _render(self) -> np.ndarray:
        w, ss = self.world, self.SS
        Ks = np.diag([ss, ss, 1.0]) @ self.K
        Ks[:2, 2] += (ss - 1) / 2                   # pixel-centre convention at the fine resolution
        cw = np.array([[X0, Y1], [X0 + TEX_W / TEX, Y1], [X0 + TEX_W / TEX, Y1 - TEX_H / TEX], [X0, Y1 - TEX_H / TEX]])
        dst = rc.project(Ks, self.dist, self.rvec, self.tvec, w.on_table(cw))
        H = cv2.getPerspectiveTransform(np.float32([[0, 0], [TEX_W, 0], [TEX_W, TEX_H], [0, TEX_H]]) - 0.5, dst.astype(np.float32))
        img = cv2.warpPerspective(w.texture, H, (self.width * ss, self.height * ss), flags=cv2.INTER_LINEAR,
                                  borderValue=(90, 80, 70))
        if w.ball is not None:
            c = w.on_table([w.ball], rc.BALL_R)[0]
            uv = rc.project(Ks, self.dist, self.rvec, self.tvec, c)[0]
            depth = (cv2.Rodrigues(self.rvec)[0] @ c + self.tvec)[2]
            cv2.circle(img, (int(round(uv[0] * 16)), int(round(uv[1] * 16))), int(round(Ks[0, 0] * rc.BALL_R / depth * 16)),
                       (20, 130, 255), -1, cv2.LINE_AA, shift=4)
        if w.tip is not None:                       # the gripper as a grey wedge ending at the fingertips
            tip = np.asarray(w.tip, float)
            back = tip + [-0.13, 0.0, 0.075] if tip[0] > 0.13 else tip + [0.0, 0.0, 0.15]
            side = np.array([0.0, 0.035, 0.0])
            poly = rc.project(Ks, self.dist, self.rvec, self.tvec, [tip + side * 0.15, back + side, back - side, tip - side * 0.15])
            cv2.fillConvexPoly(img, np.round(poly * 16).astype(np.int32), (70, 72, 78), cv2.LINE_AA, shift=4)
        return cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)

    def read(self, timeout=1.0, newer_than=None) -> Frame:
        key = (self.rvec.tobytes(), self.tvec.tobytes(), self.world.ball, None if self.world.tip is None else tuple(self.world.tip))
        if getattr(self, "_key", None) != key:
            self._key, self._clean = key, self._render()
        img = np.clip(self._clean.astype(np.int16) + self.rng.integers(-3, 4, self._clean.shape, dtype=np.int16), 0, 255)
        self.n += 1
        return Frame(img.astype(np.uint8), self.n, time.monotonic() * 1000.0, time.monotonic())

    def close(self):
        pass

    def stop_server(self):
        pass


class FakeArm:
    """Stands in for eaiproject.arm.ArmReader. `bias` is the FK model error: the arm REPORTS physical
    position + bias, which is the kinematic frame the calibration should end up in."""

    def __init__(self, world: World, bias=(0.002, -0.001, 0.009), noise=0.0005):
        self.world, self.bias, self.noise = world, np.array(bias), noise
        self.p = np.array([0.3, 0.0, 0.2])
        self.q = np.zeros(6)                          # MOTOR readings (the arm step uses them)
        self.offset = np.zeros(6)                     # zero correction, as ArmReader applies it
        self.path = None                              # f(t) -> (x, y) on the table while sweeping
        self.rng = np.random.default_rng(1)

    def touch(self, point3d):
        self.p = np.asarray(point3d, float) + self.bias + self.rng.normal(0, self.noise, 3)

    def _sample(self, t):
        p = self.p
        if self.path is not None:
            p = self.world.on_table([self.path(t)])[0] + self.bias + self.rng.normal(0, self.noise, 3)
        return ArmSample(t, self.q + self.offset, p.copy())

    def latest(self, max_age=0.3):
        return self._sample(time.monotonic())

    def between(self, t0, t1):
        return [self._sample(t) for t in np.arange(t0 + 1e-4, t1, 0.002)]

    def steady(self, window=0.25):
        return self._sample(time.monotonic()), 0.0004

    def close(self):
        pass
