"""Table map: where the table markers and the table surface are in the arm's frame, and where the
camera is.

    once:    photos of the table markers from several camera positions + fingertip touches on
             marker corners + a fingertip sweep of the table  ->  solve_rig()  ->  RigMap
             (config/rig_map.json: marker corners and the table plane in the arm's frame)
    always:  locate(): one PnP on all visible marker corners against the map -> camera pose

Everything is in the arm's KINEMATIC frame: positions as eaiproject.kinematics.Fingertip reports
them (MuJoCo FK from joint readings). The model reads heights 8-15 mm off; measuring with the arm's
own FK absorbs that, because the planner commands targets through the same model.

Geometry conventions
    table plane      z = a + b x + c y in the base frame; the normal n points up (+z side)
    plane frame      origin (0, 0, a), e1 = base x projected into the plane, e2 = n x e1
    marker           (u, v, yaw) in the plane frame; corners in OpenCV ArUco order TL, TR, BR, BL
                     (marker x right, y up, z out of the marker = n); corner index 4 = centre
    camera           OpenCV: X_cam = R(rvec) X_base + tvec
    ball             centre on the plane table + r, on the camera's side (r > 0 along n)

Markers (IDs, sizes, detection) are defined in eaiproject/markers.py. Pure functions, no hardware.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from . import markers as _markers

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "config" / "rig_map.json"                 # committed: marker corners in the arm frame
OUT_DIR = ROOT / "outputs" / "calibration"                  # last camera pose, ball colour, check screenshots
SESSIONS_DIR = ROOT / "outputs" / "rig_calib"               # raw observations per table-map session (git-ignored)
GATE_X = 0.40                                               # intercept line of the "forward" mode

CORNER_NAMES = ("TL", "TR", "BR", "BL", "C")
CENTRE = 4
BALL_R = 0.022

SIGMA_PX = 0.5                    # averaged corner, incl. intrinsics / sub-pixel bias
SIGMA_TOUCH = 0.002               # fingertip resting on a marker corner (m)
SIGMA_SWEEP = 0.0015              # fingertip height while sliding on the table (m)
F_SCALE = 2.0                     # robust-loss scale, in sigmas

# Quality rules: breaking one gives a warning, never a refusal. Errors are kept for what cannot be computed.
MIN_VIEWS = 5
WANT_MARKERS = 3                  # map markers in view for a sturdy camera pose (one is enough to place it)
FAR_DIST = 0.50                   # m from the base: a marker beyond this counts as "far from the arm"
MAX_SCALE = 0.005                 # touched layout vs the camera's: more apart = wrong marker size

COARSE_LIFT = 0.008               # m off a plane through the raw sweep: gross error (arm lifted, bumped)
LIFT_MIN, LIFT_SIGMAS = 0.0015, 3.5     # above the table by more than max(3.5 sigma, 1.5 mm) = lifted
BELOW_MIN, BELOW_SIGMAS = 0.003, 4.0    # below it is impossible; only gross reading errors get there
LIFT_PAD_S = 0.12                 # s dropped around a lift: the tip leaves and lands gradually
TOUCH_LIFT = 0.004                # m; a corner touch this much above the others' table height was lifted
OK_PX, FAIL_PX = 1.5, 3.0
OK_MM, FAIL_MM = 5.0, 10.0


class CalibrationError(ValueError):
    """The data cannot give a trustworthy result (too few markers, wrong sign, ...)."""


# --------------------------------------------------------------------------------- markers
def local_corners(size: float) -> np.ndarray:
    """(5, 2): TL, TR, BR, BL, centre in the marker's own frame (x right, y up)."""
    h = size / 2
    return np.array([[-h, h], [h, h], [h, -h], [-h, -h], [0.0, 0.0]])


def guided_corners(gray: np.ndarray, rig_map: "RigMap", K, dist, rvec, tvec, have=(), min_side_px: float = 10.0,
                   max_shift: float = 3.0, min_contrast: float = 35.0) -> dict[str, np.ndarray]:
    """Corners of map markers the detector did NOT report, found from a pose we already have.

    The pose (from the markers that were detected, or from the last relocalisation) predicts where
    every other marker's outline is; the edge fit then locks onto the real edges near that
    prediction. This recovers small, far or blurred markers whose bit pattern the ArUco detector
    cannot read. A marker is accepted only if all four edges are found within `max_shift` px of
    the prediction with a clear white-outside / black-inside contrast (so a hand, the ball or the
    arm over a marker gives nothing).
    """
    h, w = gray.shape[:2]
    out = {}
    for k, P in rig_map.markers.items():
        if k in have:
            continue
        if np.any((cv2.Rodrigues(np.asarray(rvec, float))[0] @ P.T).T[:, 2] + np.asarray(tvec, float).reshape(3)[2] <= 0):
            continue                                                   # behind the camera
        pred = project(K, dist, rvec, tvec, P)
        sides = np.linalg.norm(pred - np.roll(pred, -1, axis=0), axis=1)
        if sides.min() < min_side_px or pred.min() < 6 or pred[:, 0].max() > w - 7 or pred[:, 1].max() > h - 7:
            continue
        c = _markers.fit_edges(gray, pred, max_shift=max_shift)
        if c is not None and _markers.edge_contrast(gray, c) >= min_contrast:
            out[k] = c
    return out


def detect_with_map(img_bgr: np.ndarray, rig_map: "RigMap", K, dist, pose=None) -> tuple[dict, list]:
    """markers.detect() plus guided_corners() for the map markers it missed. `pose` = (rvec, tvec) to
    predict from; None = a PnP on the markers detected in this image (needs 2 of them).
    Returns (key -> corners, keys that were found by prediction)."""
    found = {k: c for k, c in _markers.detect(img_bgr).items() if k in rig_map.markers}
    if pose is None:
        if len(found) < 2:
            return found, []
        try:
            pose = _pnp(np.concatenate([rig_map.markers[k] for k in found]), np.concatenate(list(found.values())), K, dist)
        except (CalibrationError, cv2.error):
            return found, []
    extra = guided_corners(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), rig_map, K, dist, *pose, have=found)
    return {**found, **extra}, sorted(extra)


def average_corners(frames: list[dict], min_frac: float = 0.7) -> tuple[dict, dict]:
    """Mean corners over frames for markers seen in >= min_frac of them, and their jitter (max std, px)."""
    acc: dict[str, list] = {}
    for found in frames:
        for k, c in found.items():
            acc.setdefault(k, []).append(np.asarray(c, float).reshape(4, 2))
    keep = {k: v for k, v in acc.items() if len(v) >= max(1, min_frac * len(frames))}
    return ({k: np.mean(v, axis=0) for k, v in keep.items()},
            {k: float(np.std(v, axis=0).max()) for k, v in keep.items()})


# ----------------------------------------------------------------------------------- plane
def plane_z(plane, x, y):
    a, b, c = plane
    return a + b * np.asarray(x, float) + c * np.asarray(y, float)


def plane_frame(plane) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Origin, in-plane axes e1, e2 and the upward unit normal n of z = a + b x + c y."""
    a, b, c = (float(v) for v in plane)
    n = np.array([-b, -c, 1.0])
    n /= np.linalg.norm(n)
    e1 = np.array([1.0, 0.0, b])
    e1 /= np.linalg.norm(e1)
    return np.array([0.0, 0.0, a]), e1, np.cross(n, e1), n


def to_plane(plane, P) -> tuple[np.ndarray, np.ndarray]:
    """Base-frame points -> in-plane coordinates (N, 2) and signed height above the plane (N,)."""
    o, e1, e2, n = plane_frame(plane)
    d = np.asarray(P, float).reshape(-1, 3) - o
    return np.c_[d @ e1, d @ e2], d @ n


def from_plane(plane, uv, h=0.0) -> np.ndarray:
    o, e1, e2, n = plane_frame(plane)
    uv = np.asarray(uv, float).reshape(-1, 2)
    return o + uv[:, :1] * e1 + uv[:, 1:] * e2 + np.reshape(h, (-1, 1)) * n


def marker_points(plane, pose, size: float) -> np.ndarray:
    """(5, 3): the 4 corners + centre of a marker with in-plane pose (u, v, yaw), base frame."""
    u, v, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    return from_plane(plane, np.array([u, v]) + local_corners(size) @ np.array([[c, s], [-s, c]]))


def rigid_fit_2d(local: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """(u, v, yaw) of the rotation + translation taking `local` points onto `uv` (least squares)."""
    lm, um = local.mean(axis=0), uv.mean(axis=0)
    l, w = local - lm, uv - um
    yaw = np.arctan2(np.sum(l[:, 0] * w[:, 1] - l[:, 1] * w[:, 0]), np.sum(l[:, 0] * w[:, 0] + l[:, 1] * w[:, 1]))
    c, s = np.cos(yaw), np.sin(yaw)
    t = um - np.array([[c, -s], [s, c]]) @ lm
    return np.array([t[0], t[1], yaw])


def fit_plane(P: np.ndarray, iters: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Robust z = a + b x + c y (trimmed least squares). Returns (coef, residuals of all points)."""
    P = np.asarray(P, float).reshape(-1, 3)
    A = np.c_[np.ones(len(P)), P[:, 0], P[:, 1]]
    keep = np.ones(len(P), bool)
    for _ in range(iters):
        coef = np.linalg.lstsq(A[keep], P[keep, 2], rcond=None)[0]
        res = P[:, 2] - A @ coef
        sig = max(1.4826 * float(np.median(np.abs(res - np.median(res)))), 5e-4)
        new = np.abs(res) < 3 * sig
        if new.sum() < 3 or np.array_equal(new, keep):
            break
        keep = new
    return coef, res




# ----------------------------------------------------------------------------------- sweep
def expand_in_time(t, flag, pad: float) -> np.ndarray:
    """`flag` widened to every sample within `pad` seconds of a flagged one."""
    t, flag = np.asarray(t, float), np.asarray(flag, bool)
    if not flag.any() or pad <= 0:
        return flag.copy()
    ft = np.sort(t[flag])
    i = np.searchsorted(ft, t)
    near = np.minimum(np.abs(t - ft[np.clip(i - 1, 0, len(ft) - 1)]), np.abs(ft[np.clip(i, 0, len(ft) - 1)] - t))
    return near <= pad


def count_in_time(t, flag, half: float) -> np.ndarray:
    """Number of flagged samples within +-`half` seconds of every sample (itself included)."""
    t, flag = np.asarray(t, float), np.asarray(flag, bool)
    o = np.argsort(t, kind="stable")
    ts, cs = t[o], np.r_[0, np.cumsum(flag[o])]
    n = np.empty(len(t), int)
    n[o] = cs[np.searchsorted(ts, ts + half, side="right")] - cs[np.searchsorted(ts, ts - half, side="left")]
    return n


def lifted_samples(h, xy, t, use, pad: float = LIFT_PAD_S) -> tuple[np.ndarray, np.ndarray, float]:
    """One-sided outlier test for the sweep. h = contact height above the fitted table (m) per sample.

    The tip cannot go below the table, but it can leave it: samples clearly ABOVE the table are
    "lifted" (threshold max(3.5 sigma, 1.5 mm)). Where that lasts (most samples within +-0.05 s above
    half the threshold) the samples within `pad` seconds go too: the tip leaves and lands gradually; a
    lone noise spike is dropped alone. Below the table only gross errors are dropped (max(4 sigma, 3 mm)). A
    smooth surface (quadratic in x, y) is taken out first, so the arm model's smooth height error is
    not mistaken for lifting. Returns (lifted, below, sigma) over all samples; `use` = candidates.
    """
    h, xy, use = np.asarray(h, float), np.asarray(xy, float).reshape(-1, 2), np.asarray(use, bool)
    r = h.copy()
    if use.sum() >= 30:
        A = np.c_[np.ones(len(h)), xy, xy ** 2, xy[:, 0] * xy[:, 1]]
        keep = use.copy()
        for _ in range(4):
            coef = np.linalg.lstsq(A[keep], h[keep], rcond=None)[0]
            r = h - A @ coef
            sig = max(1.4826 * float(np.median(np.abs(r[use] - np.median(r[use])))), 2e-4)
            new = use & (np.abs(r) < 3 * sig)
            if np.array_equal(new, keep) or new.sum() < 30:
                break
            keep = new
    sig = max(1.4826 * float(np.median(np.abs(r[use] - np.median(r[use])))), 2e-4) if use.any() else 0.0
    below = use & (r < -max(BELOW_SIGMAS * sig, BELOW_MIN))
    thr = max(LIFT_SIGMAS * sig, LIFT_MIN)
    lifted = use & (r > thr)
    near = np.maximum(count_in_time(t, use, 0.05), 1)
    lasting = lifted & (count_in_time(t, use & (r > thr / 2), 0.05) >= 0.5 * near) & (near >= 2)
    lifted = (lifted | expand_in_time(t, lasting, pad)) & use & ~below
    return lifted, below, sig


# ------------------------------------------------------------------------------ projection
def undistort_px(K, dist, uv) -> np.ndarray:
    """Distorted pixels -> pixels of the ideal pinhole camera with the same K."""
    uv = np.asarray(uv, np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(uv, np.asarray(K, float), np.asarray(dist, float), P=np.asarray(K, float)).reshape(-1, 2)


def project(K, dist, rvec, tvec, P) -> np.ndarray:
    p, _ = cv2.projectPoints(np.asarray(P, np.float64).reshape(-1, 1, 3), np.asarray(rvec, float),
                             np.asarray(tvec, float), np.asarray(K, float), np.asarray(dist, float))
    return p.reshape(-1, 2)


def camera_centre(rvec, tvec) -> np.ndarray:
    return -cv2.Rodrigues(np.asarray(rvec, float))[0].T @ np.asarray(tvec, float).reshape(3)


def backproject_to_plane(K, dist, rvec, tvec, plane, uv, offset: float = 0.0) -> np.ndarray:
    """Pixels -> base-frame points on the table plane raised by `offset` along its normal."""
    K = np.asarray(K, float)
    und = undistort_px(K, dist, uv)
    rays = np.c_[(und[:, 0] - K[0, 2]) / K[0, 0], (und[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(und))]
    R = cv2.Rodrigues(np.asarray(rvec, float))[0]
    d = rays @ R                                      # rows: R^T ray
    C = camera_centre(rvec, tvec)
    o, _, _, n = plane_frame(plane)
    s = (offset - n @ (C - o)) / (d @ n)
    return C + s[:, None] * d


def pixel_to_ball_xy(K, dist, rvec, tvec, plane, uv, r: float = BALL_R) -> np.ndarray:
    """Ball-centre (x, y) in the base frame for blob-centre pixels: the ray meets the plane of the
    ball centre, table + r on the camera's side. Fails loudly if the sign convention is broken."""
    if not r > 0:
        raise CalibrationError(f"ball radius must be positive (got {r}): the ball centre is ABOVE the table, "
                               "on the camera's side")
    o, _, _, n = plane_frame(plane)
    height = float(n @ (camera_centre(rvec, tvec) - o))
    if height <= r:
        raise CalibrationError(f"the camera is {height * 1000:.0f} mm above the table plane along +n: it must look "
                               "at the table from above (wrong plane sign or a mirrored pose)")
    return backproject_to_plane(K, dist, rvec, tvec, plane, uv, offset=r)[:, :2]


def _pnp(obj, img, K, dist):
    """SQPNP start, then Levenberg-Marquardt refinement on all points."""
    obj = np.asarray(obj, np.float64).reshape(-1, 1, 3)
    img = np.asarray(img, np.float64).reshape(-1, 1, 2)
    K, dist = np.asarray(K, float), np.asarray(dist, float)
    ok, rv, tv = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise CalibrationError("solvePnP failed")
    rv, tv = cv2.solvePnPRefineLM(obj, img, K, dist, rv, tv)
    return rv.reshape(3), tv.reshape(3)


# ------------------------------------------------------------------------------ data model
@dataclass
class View:
    name: str
    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int
    corners: dict                 # key -> (4, 2) averaged corner pixels


@dataclass
class Touch:
    key: str
    corner: int                   # 0..3 = TL, TR, BR, BL; 4 = centre
    p: np.ndarray                 # Fingertip(q), base frame


@dataclass
class Sweep:
    """Fingertip samples recorded while the tip slides on the table, in any gripper orientation.
    Every `grid` cell counts once in the solve, however long the tip stayed there."""
    p: np.ndarray                 # (S, 3) Fingertip(q)
    t: np.ndarray                 # (S,) time, s
    grid: float = 0.02

    @staticmethod
    def empty() -> "Sweep":
        return Sweep(np.zeros((0, 3)), np.zeros(0))

    @staticmethod
    def of_points(P, dt: float = 0.02) -> "Sweep":
        P = np.asarray(P, float).reshape(-1, 3)
        return Sweep(P, np.arange(len(P)) * dt)

    def __len__(self) -> int:
        return len(self.p)

    def cells(self, use) -> tuple[np.ndarray, int]:
        """Cell index of every used sample, and the number of cells."""
        ij = np.floor(self.p[use, :2] / self.grid).astype(int)
        if len(ij) == 0:
            return np.zeros(0, int), 0
        _, idx = np.unique(ij, axis=0, return_inverse=True)
        return idx.ravel(), int(idx.max()) + 1


@dataclass
class Observations:
    views: list
    touches: list
    sweep: Sweep = field(default_factory=Sweep.empty)
    sizes: dict = field(default_factory=lambda: dict(_markers.SIZES))
    sources: dict = field(default_factory=dict)


@dataclass
class RigMap:
    markers: dict                 # key -> (4, 3) corners in the base frame (TL, TR, BR, BL)
    plane: np.ndarray             # table plane z = a + b x + c y
    sizes: dict
    touched: dict = field(default_factory=dict)          # key -> touched corner indices
    far_dist: float = FAR_DIST
    time: str = ""
    stats: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    sources: dict = field(default_factory=dict)

    def centre(self, key: str) -> np.ndarray:
        return self.markers[key].mean(axis=0)

    def is_far(self, key: str) -> bool:
        return float(np.linalg.norm(self.centre(key)[:2])) >= self.far_dist

    def to_json(self) -> dict:
        mk = {}
        for k, c in self.markers.items():
            yaw = np.degrees(np.arctan2(c[1, 1] - c[0, 1], c[1, 0] - c[0, 0]))
            mk[k] = {"size": self.sizes[k], "corners": np.round(c, 5).tolist(),
                     "centre": np.round(c.mean(axis=0), 5).tolist(), "yaw_deg": round(float(yaw), 2),
                     "touched_corners": [CORNER_NAMES[i] for i in self.touched.get(k, [])],
                     "far": self.is_far(k)}
        return {"version": 1, "time": self.time,
                "frame": "arm kinematic base frame (eaiproject.kinematics.Fingertip), metres",
                "corner_order": "TL, TR, BR, BL (OpenCV ArUco)",
                "plane_z=a+bx+cy": [float(v) for v in self.plane], "far_dist": self.far_dist, "markers": mk,
                "stats": self.stats, "warnings": self.warnings, "sources": self.sources}

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")

    @staticmethod
    def from_json(d: dict) -> "RigMap":
        mk = {k: np.array(m["corners"], float) for k, m in d["markers"].items()}
        return RigMap(markers=mk, plane=np.array(d["plane_z=a+bx+cy"], float),
                      sizes={k: float(d["markers"][k]["size"]) for k in mk},
                      touched={k: [CORNER_NAMES.index(n) for n in d["markers"][k]["touched_corners"]] for k in mk
                               if d["markers"][k].get("touched_corners")},
                      far_dist=float(d.get("far_dist", FAR_DIST)), time=d.get("time", ""),
                      stats=d.get("stats", {}), warnings=d.get("warnings", []), sources=d.get("sources", {}))

    @staticmethod
    def load(path) -> "RigMap":
        return RigMap.from_json(json.loads(Path(path).read_text()))


def load_observations(session_dir, grid: float = 0.02) -> Observations:
    """Read what scripts/real/calibrate.py saved (views.json, touches.json, sweep.npz)."""
    d = Path(session_dir)
    sizes = dict(_markers.SIZES)
    views, touches, sweep = [], [], Sweep.empty()
    if (d / "views.json").exists():
        for v in json.loads((d / "views.json").read_text())["views"]:
            views.append(View(v["name"], np.array(v["K"], float), np.array(v["dist"], float), v["width"], v["height"],
                              {k: np.array(c, float) for k, c in v["corners"].items() if k in sizes}))
    if (d / "touches.json").exists():
        for t in json.loads((d / "touches.json").read_text())["touches"]:
            if t["key"] in sizes:
                touches.append(Touch(t["key"], int(t["corner"]), np.array(t["p"], float)))
    if (d / "sweep.npz").exists():
        z = np.load(d / "sweep.npz")
        sweep = Sweep.of_points(z["p"])
        if "t" in z:
            sweep.t = np.asarray(z["t"], float)
        sweep.grid = grid
    src = {"session": str(d), "views": "views.json", "touches": "touches.json", "sweep": "sweep.npz"}
    return Observations(views, touches, sweep, sizes, src)


# ---------------------------------------------------------------------------------- solver
def _rodrigues(r: np.ndarray) -> np.ndarray:
    """(V, 3) rotation vectors -> (V, 3, 3) matrices."""
    th = np.linalg.norm(r, axis=1)
    k = r / np.where(th < 1e-12, 1.0, th)[:, None]
    Kx = np.zeros((len(r), 3, 3))
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    Kx[:, 1, 2], Kx[:, 2, 0], Kx[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    return np.eye(3) + s * Kx + (1 - c) * (Kx @ Kx)




class _Problem:
    """Index arrays for the joint problem. x = [plane (3) | marker u, v, yaw (3M) | rvec, tvec (6V)]."""

    def __init__(self, obs: Observations, keys: list[str], touches: list[Touch], sweep_use=None, touch_z=None):
        self.keys, self.views = keys, obs.views
        self.M, self.V = len(keys), len(obs.views)
        ki = {k: i for i, k in enumerate(keys)}
        self.local = np.array([local_corners(obs.sizes[k]) for k in keys])          # (M, 5, 2)
        vi, mi, ci, und, f, c = [], [], [], [], [], []
        for v_idx, v in enumerate(obs.views):
            for k, px in v.corners.items():
                if k not in ki:
                    continue
                u = undistort_px(v.K, v.dist, px)
                for j in range(4):
                    vi.append(v_idx); mi.append(ki[k]); ci.append(j); und.append(u[j])
                    f.append([v.K[0, 0], v.K[1, 1]]); c.append([v.K[0, 2], v.K[1, 2]])
        self.vi, self.mi, self.ci = np.array(vi), np.array(mi), np.array(ci)
        self.und, self.f, self.c = np.array(und), np.array(f), np.array(c)
        self.t_mi = np.array([ki[t.key] for t in touches], int)
        self.t_ci = np.array([t.corner for t in touches], int)
        self.t_p = np.array([t.p for t in touches], float).reshape(-1, 3)
        self.t_zw = np.ones(len(touches)) if touch_z is None else np.asarray(touch_z, float)   # 0 = height ignored
        sw = obs.sweep
        use = np.ones(len(sw), bool) if sweep_use is None else np.asarray(sweep_use, bool)
        self.s_p = sw.p[use]
        self.s_cell, self.n_cells = sw.cells(use)
        self.s_count = np.maximum(np.bincount(self.s_cell, minlength=self.n_cells), 1).astype(float)

    def unpack(self, x):
        return x[:3], x[3:3 + 3 * self.M].reshape(self.M, 3), x[3 + 3 * self.M:].reshape(self.V, 6)

    def pack(self, plane, poses, cams):
        return np.r_[plane, np.ravel(poses), np.ravel(cams)]

    def points(self, plane, poses) -> np.ndarray:
        """(M, 5, 3) corners + centre of every marker in the base frame."""
        o, e1, e2, _ = plane_frame(plane)
        c, s = np.cos(poses[:, 2])[:, None], np.sin(poses[:, 2])[:, None]
        u = poses[:, 0:1] + c * self.local[:, :, 0] - s * self.local[:, :, 1]
        v = poses[:, 1:2] + s * self.local[:, :, 0] + c * self.local[:, :, 1]
        return o + u[..., None] * e1 + v[..., None] * e2

    def cell_points(self) -> np.ndarray:
        """(cells, 3) mean fingertip position per sweep cell."""
        return np.stack([np.bincount(self.s_cell, self.s_p[:, i], self.n_cells) for i in range(3)], axis=1) / \
            self.s_count[:, None]

    def parts(self, x):
        """Unweighted residuals: reprojection (N, 2) px, touches (T, 3) m, sweep cell heights (cells,) m."""
        plane, poses, cams = self.unpack(x)
        P = self.points(plane, poses)
        R = _rodrigues(cams[:, :3])
        Xc = np.einsum("nij,nj->ni", R[self.vi], P[self.mi, self.ci]) + cams[self.vi, 3:]
        reproj = self.f * Xc[:, :2] / Xc[:, 2:3] + self.c - self.und
        touch = self.t_p - P[self.t_mi, self.t_ci] if len(self.t_p) else np.zeros((0, 3))
        h = self.s_p[:, 2] - plane_z(plane, self.s_p[:, 0], self.s_p[:, 1])
        sweep = np.bincount(self.s_cell, h, self.n_cells) / self.s_count
        return reproj, touch, sweep

    def residuals(self, x):
        reproj, touch, sweep = self.parts(x)
        touch = touch * np.c_[np.ones((len(touch), 2)), self.t_zw]
        return np.concatenate([reproj.ravel() / SIGMA_PX, touch.ravel() / SIGMA_TOUCH, sweep / SIGMA_SWEEP])


def _initial_guess(obs: Observations, prob: _Problem, touches: list[Touch]):
    """Plane from the sweep (else the touches), touched markers from their touched points, untouched
    markers by back-projection through views whose pose comes from PnP on already-known markers."""
    if len(prob.s_p) >= 10:
        plane, _ = fit_plane(prob.s_p)
    else:
        tp = np.array([t.p for t in touches])
        plane = fit_plane(tp)[0] if len(tp) >= 3 else np.array([float(np.median(tp[:, 2])), 0.0, 0.0])
    poses: dict[str, np.ndarray] = {}
    for k in prob.keys:
        tk = [t for t in touches if t.key == k]
        if len(tk) >= 2:
            uv, _ = to_plane(plane, [t.p for t in tk])
            poses[k] = rigid_fit_2d(local_corners(obs.sizes[k])[[t.corner for t in tk]], uv)
    if not poses:
        raise CalibrationError("no marker has at least two touched points: the map cannot be placed in the arm frame")

    def view_pose(v):
        known = [k for k in v.corners if k in poses]
        obj = np.concatenate([marker_points(plane, poses[k], obs.sizes[k])[:4] for k in known])
        return _pnp(obj, np.concatenate([v.corners[k] for k in known]), v.K, v.dist)

    def place(todo_only: bool):
        est: dict[str, list] = {}
        for v in obs.views:
            if v.name not in cams:
                continue
            for k, px in v.corners.items():
                if k in prob.keys and (k not in poses if todo_only else k not in touched):
                    est.setdefault(k, []).append(to_plane(plane, backproject_to_plane(v.K, v.dist, *cams[v.name], plane, px))[0])
        for k, l in est.items():
            poses[k] = rigid_fit_2d(local_corners(obs.sizes[k])[:4], np.mean(l, axis=0))
        return bool(est)

    touched = set(poses)
    cams: dict[str, tuple] = {}
    progressed = True
    while progressed:
        progressed = False
        todo = [v for v in obs.views if v.name not in cams and any(k in poses for k in v.corners)]
        for v in sorted(todo, key=lambda v: -sum(k in poses for k in v.corners)):
            cams[v.name] = view_pose(v)
            progressed = True
        progressed |= place(todo_only=True)
    lost_v = [v.name for v in obs.views if v.name not in cams]
    lost_m = [k for k in prob.keys if k not in poses]
    if lost_v or lost_m:
        raise CalibrationError(f"cannot link everything to the touched markers: views {lost_v}, markers {lost_m}. "
                               "Each view must show a touched marker, or share a marker with a view that does.")
    for _ in range(2):                                   # polish: all markers in every PnP, all views per marker
        cams = {v.name: view_pose(v) for v in obs.views}
        place(todo_only=False)
    return prob.pack(plane, [poses[k] for k in prob.keys], [np.r_[cams[v.name]] for v in obs.views])




@dataclass
class RigSolution:
    rig_map: RigMap
    cams: dict                    # view name -> (rvec, tvec)
    report: dict
    warnings: list
    touch_points: np.ndarray      # (T, 3) touched fingertip positions
    touch_residuals: np.ndarray   # (T, 3) touch - fitted corner (m)
    sweep_points: np.ndarray      # (cells, 3) mean fingertip position per 2 cm cell (used samples)
    sweep_residuals: np.ndarray   # (cells,) cell height - plane (m)
    sweep_rejected: np.ndarray    # (n, 3) samples left out (lifted / gross)


def _linked(obs: Observations, keys: list[str], touches: list[Touch]) -> tuple[list, list]:
    """The views and markers a solve can place: from the markers with two or more touched points,
    through the views that see a placed marker, to the markers those views see, and so on."""
    placed = {k for k in keys if sum(t.key == k for t in touches) >= 2}
    names: set[str] = set()
    grow = bool(placed)
    while grow:
        grow = False
        for v in obs.views:
            if v.name not in names and any(k in placed for k in v.corners):
                names.add(v.name)
                placed |= {k for k in v.corners if k in keys}
                grow = True
    return [v for v in obs.views if v.name in names], [k for k in keys if k in placed]


def _run(prob: _Problem, x0: np.ndarray) -> np.ndarray:
    return least_squares(prob.residuals, x0, loss="soft_l1", f_scale=F_SCALE, x_scale="jac", max_nfev=300).x


def solve_rig(obs: Observations, far_dist: float = FAR_DIST) -> RigSolution:
    """Joint least squares for the table plane, every table marker's pose in the plane and one camera
    pose per view, in the arm's kinematic frame. Sweep samples (and touch heights) where the tip had
    left the table are found from a first solution and dropped, then it is solved again. Data that
    only weakens the map gives a warning (views and markers not linked to a touched marker are left
    out); CalibrationError only when nothing can be placed in the arm frame."""
    warn: list[str] = []
    if not obs.views:
        raise CalibrationError("no camera views")
    keys = sorted({k for v in obs.views for k in v.corners if k in obs.sizes})
    if not keys:
        raise CalibrationError("no table marker in the views")
    touches = [t for t in obs.touches if t.key in keys]
    views, linked = _linked(obs, keys, touches)
    if not linked:
        raise CalibrationError("no marker seen by the camera has two or more touched points: nothing ties the map "
                               "to the arm frame. Touch 2 or more corners of one marker")
    if len(views) < len(obs.views) or len(linked) < len(keys):
        lost_v = [v.name for v in obs.views if v.name not in {u.name for u in views}]
        lost_m = [k for k in keys if k not in linked]
        warn.append(f"left out, not linked to a touched marker: views {lost_v}, markers {lost_m} (each view must show "
                    "a touched marker, or share a marker with a view that does)")
        obs, keys = replace(obs, views=views), linked
        touches = [t for t in touches if t.key in keys]
    if len(keys) < WANT_MARKERS:
        warn.append(f"only {len(keys)} table marker(s) in the map ({keys}): the camera wants {WANT_MARKERS} or more in "
                    "view for a sturdy pose")
    if len(obs.views) < MIN_VIEWS:
        warn.append(f"only {len(obs.views)} views (at least {MIN_VIEWS} wanted)")
    if len({t.key for t in touches}) < 2:
        warn.append("only one marker touched: the direction of the map is weak; touch 2 or more")

    # sweep: gross errors (far off a plane through the raw samples), then solve, then the one-sided
    # lift test against the solved table and the touch heights, and solve again
    sw = obs.sweep
    base = np.ones(len(sw), bool)
    if len(sw) >= 10:
        base &= ~expand_in_time(sw.t, np.abs(fit_plane(sw.p)[1]) > COARSE_LIFT, LIFT_PAD_S)
    use, touch_z = base.copy(), np.ones(len(touches))
    lifted = below = np.zeros(len(sw), bool)
    sig = 0.0
    prob = _Problem(obs, keys, touches, use, touch_z)
    x = _run(prob, _initial_guess(obs, prob, touches))
    for _ in range(3):
        plane = prob.unpack(x)[0]
        new_use, new_z = use, touch_z
        if len(sw) >= 10:
            h = sw.p[:, 2] - plane_z(plane, sw.p[:, 0], sw.p[:, 1])
            lifted, below, sig = lifted_samples(h, sw.p[:, :2], sw.t, base)
            new_use = base & ~lifted & ~below
        dz = prob.parts(x)[1][:, 2]
        if len(dz) >= 3:
            new_z = (dz <= max(0.0, float(np.median(dz))) + TOUCH_LIFT).astype(float)
        if np.array_equal(new_use, use) and np.array_equal(new_z, touch_z):
            break
        use, touch_z = new_use, new_z
        prob = _Problem(obs, keys, touches, use, touch_z)
        x = _run(prob, x)

    plane, poses, cams = prob.unpack(x)
    reproj, touch, sweep = prob.parts(x)
    P = prob.points(plane, poses)
    o, _, _, n = plane_frame(plane)
    for v, cam in zip(obs.views, cams):
        if n @ (camera_centre(cam[:3], cam[3:]) - o) <= 0:
            raise CalibrationError(f"view {v.name} ends up below the table: mirrored solution, check the touches")

    err = np.linalg.norm(reproj, axis=1)
    rms = lambda e: float(np.sqrt(np.mean(np.square(e)))) if len(e) else 0.0
    pressed = touch_z > 0
    report = {
        "n_views": len(obs.views), "n_markers": len(keys), "n_touches": len(touches), "n_sweep_cells": prob.n_cells,
        "reproj_rms_px": round(rms(err), 3), "reproj_max_px": round(float(err.max()), 3),
        "per_marker_reproj_px": {k: round(rms(err[prob.mi == i]), 3) for i, k in enumerate(keys)},
        "per_view_reproj_px": {v.name: round(rms(err[prob.vi == i]), 3) for i, v in enumerate(obs.views)},
        "touch_rms_mm": round(rms(np.linalg.norm(touch[pressed], axis=1)) * 1000, 2),
        "touch_xy_rms_mm": round(rms(np.linalg.norm(touch[:, :2], axis=1)) * 1000, 2),
        "touches": [{"marker": t.key, "corner": CORNER_NAMES[t.corner], "dx_mm": round(float(r[0]) * 1000, 2),
                     "dy_mm": round(float(r[1]) * 1000, 2), "dz_mm": round(float(r[2]) * 1000, 2),
                     "err_mm": round(float(np.linalg.norm(r)) * 1000, 2), "lifted": bool(zw == 0)}
                    for t, r, zw in zip(touches, touch, touch_z)],
        "sweep": {"samples": len(sw), "gross": int((~base).sum()), "lifted": int(lifted.sum()), "below": int(below.sum()),
                  "used": int(use.sum()), "cells": prob.n_cells, "sigma_mm": round(sig * 1000, 2)},
        "sweep_rms_mm": round(rms(sweep) * 1000, 2),
        "plane_z=a+bx+cy": [float(v) for v in plane],
        "table_on_line_mm": {f"{y:+.2f}": round(float(plane_z(plane, GATE_X, y)) * 1000, 2) for y in (-0.24, 0.0, 0.24)},
        "far_markers": [k for i, k in enumerate(keys) if np.linalg.norm(P[i, 4, :2]) >= far_dist],
    }
    # Scale check: the camera measures distances in units of the printed marker size, the arm directly.
    fit_xy, t_xy = (prob.t_p - touch)[:, :2], prob.t_p[:, :2]
    fc, tc = fit_xy - fit_xy.mean(axis=0), t_xy - t_xy.mean(axis=0)
    scale = float(np.sum(fc * tc) / max(np.sum(fc * fc), 1e-12))
    report["touch_scale_percent"] = round((scale - 1) * 100, 3)
    if abs(scale - 1) > MAX_SCALE and float(np.ptp(t_xy, axis=0).max()) > 0.2:
        warn.append(f"the touched points are {abs(scale - 1) * 100:.1f} % {'further apart' if scale > 1 else 'closer together'} "
                    "than the camera's layout: measure a printed marker (must match markers.py)")
    if report["reproj_rms_px"] > OK_PX:
        warn.append(f"reprojection RMS {report['reproj_rms_px']} px is above {OK_PX} px")
    if report["touch_rms_mm"] > OK_MM:
        warn.append(f"touch residual RMS {report['touch_rms_mm']} mm is above {OK_MM} mm: a corner was touched wrongly?")
    for t, zw in zip(touches, touch_z):
        if zw == 0:
            warn.append(f"touch {t.key} {CORNER_NAMES[t.corner]} sits above the table (tip lifted?): its height is ignored")
    if len(sw) and (len(sw) - use.sum()) > 0.25 * len(sw):
        warn.append(f"{(len(sw) - use.sum()) / len(sw) * 100:.0f} % of the sweep was off the table and dropped: "
                    "keep the tip pressed on the table while recording")
    if not prob.n_cells:
        warn.append("no table sweep: the table plane comes from the corner touches only")
    if not report["far_markers"]:
        warn.append(f"no marker is farther than {far_dist:.2f} m from the base: locating the camera needs one")

    touched = {k: sorted({t.corner for t in touches if t.key == k}) for k in sorted({t.key for t in touches})}
    rig_map = RigMap(markers={k: P[i, :4].copy() for i, k in enumerate(keys)}, plane=np.array(plane, float),
                     sizes={k: obs.sizes[k] for k in keys}, touched=touched, far_dist=far_dist,
                     time=time.strftime("%Y-%m-%d %H:%M:%S"),
                     stats={k: report[k] for k in ("n_views", "n_markers", "n_touches", "n_sweep_cells", "reproj_rms_px",
                                                   "per_marker_reproj_px", "touch_rms_mm", "touch_xy_rms_mm",
                                                   "touch_scale_percent", "sweep_rms_mm", "sweep", "table_on_line_mm")},
                     warnings=list(warn), sources=dict(obs.sources))
    report["warnings"] = warn
    return RigSolution(rig_map, {v.name: (c[:3].copy(), c[3:].copy()) for v, c in zip(obs.views, cams)}, report, warn,
                       prob.t_p.copy(), touch, prob.cell_points(), sweep, sw.p[~use] if len(sw) else np.zeros((0, 3)))


# -------------------------------------------------------------------------- relocalisation
@dataclass
class Reloc:
    rvec: np.ndarray
    tvec: np.ndarray
    used: list
    rejected: list
    per_marker_px: dict
    per_marker_mm: dict
    reproj_rms_px: float
    table_rms_mm: float
    camera_position: np.ndarray
    guided: list = field(default_factory=list)      # markers located by prediction (guided_corners)
    corners: dict = field(default_factory=dict)     # the corner pixels the pose was fitted to
    jitter_px: float = 0.0                          # largest corner movement while averaging (locate)
    warnings: list = field(default_factory=list)    # what makes this pose weaker (it is still used)

    @property
    def status(self) -> str:
        return status(self.reproj_rms_px, self.table_rms_mm)


def status(px: float | None = None, mm: float | None = None) -> str:
    """OK / WARN / FAIL from a reprojection error (px) and / or a position error (mm)."""
    px, mm = px or 0.0, mm or 0.0
    if px > FAIL_PX or mm > FAIL_MM:
        return "FAIL"
    return "OK" if px < OK_PX and mm < OK_MM else "WARN"


def marker_errors(rig_map: RigMap, K, dist, rvec, tvec, detections: dict) -> tuple[dict, dict]:
    """Per-marker error of a pose against detected corners: mean reprojection (px) and mean
    distance on the table between the detected corners and the map (mm)."""
    px, mm = {}, {}
    for k, c in detections.items():
        if k not in rig_map.markers:
            continue
        c = np.asarray(c, float).reshape(4, 2)
        px[k] = float(np.linalg.norm(project(K, dist, rvec, tvec, rig_map.markers[k]) - c, axis=1).mean())
        on_table = backproject_to_plane(K, dist, rvec, tvec, rig_map.plane, c)
        mm[k] = float(np.linalg.norm(on_table - rig_map.markers[k], axis=1).mean() * 1000)
    return px, mm


def relocalize(rig_map: RigMap, K, dist, detections: dict, guided: dict | None = None) -> Reloc:
    """Camera pose in the arm frame from ONE PnP over all visible table-marker corners.

    `detections`: key -> (4, 2) corners, ideally averaged over ~30 frames (average corners, not
    poses). One map marker places the camera. Fewer than WANT_MARKERS, or none far from the arm
    (near markers alone leave the pose loose down the table), give a warning in `Reloc.warnings`.
    `guided`: extra corners from guided_corners() (markers located by prediction, not identified by
    their bit pattern). They join the fit but are not counted for those warnings.
    CalibrationError only when no map marker is visible or the pose comes out mirrored.
    """
    seen = {k: np.asarray(c, float).reshape(4, 2) for k, c in detections.items() if k in rig_map.markers}
    extra = {k: np.asarray(c, float).reshape(4, 2) for k, c in (guided or {}).items() if k in rig_map.markers and k not in seen}
    unknown = sorted(k for k in detections if k not in rig_map.markers)
    ident = sorted(seen)
    if not ident:
        raise CalibrationError(f"no map marker visible (map: {sorted(rig_map.markers)}). Aim the camera at the table, "
                               "check lighting / occlusion." + (f" Markers not in the map: {unknown}." if unknown else ""))
    warn = []
    if len(ident) < WANT_MARKERS:
        warn.append(f"only {len(ident)} map marker(s) in view ({ident}): the camera pose is weaker; "
                    f"{WANT_MARKERS} or more are better")
    far_seen = any(rig_map.is_far(k) for k in ident)
    if not far_seen:
        far = sorted(k for k in rig_map.markers if rig_map.is_far(k))
        warn.append(f"no marker far from the arm in view (beyond {rig_map.far_dist:.2f} m: {far or 'none in the map'}): "
                    "the pose is loose far down the table; aim the camera further down")

    def fit(keys):
        rv, tv = _pnp(np.concatenate([rig_map.markers[k] for k in keys]), np.concatenate([seen[k] for k in keys]), K, dist)
        return rv, tv, marker_errors(rig_map, K, dist, rv, tv, {k: seen[k] for k in keys})

    def can_drop(k):            # only while WANT_MARKERS identified ones, and a far one if one is seen, remain
        rest = [j for j in ident if j != k]
        return k in extra or (len(rest) >= WANT_MARKERS and (not far_seen or any(rig_map.is_far(j) for j in rest)))

    seen.update(extra)
    keys = sorted(seen)
    rvec, tvec, (px, mm) = fit(keys)
    rejected = []
    if len(ident) > WANT_MARKERS and np.mean(list(px.values())) > OK_PX:
        # A bad fit with markers to spare: does dropping ONE marker (moved on the table, misread) fix it?
        # (A fit on one marker alone always looks good, hence the markers kept by can_drop.)
        best = None
        for k in keys:
            if not can_drop(k):
                continue
            rest = [j for j in keys if j != k]
            cand = fit(rest)
            if best is None or np.mean(list(cand[2][0].values())) < np.mean(list(best[1][2][0].values())):
                best = (k, cand, rest)
        if best is not None and np.mean(list(best[1][2][0].values())) < min(OK_PX, 0.4 * np.mean(list(px.values()))):
            (bad, (rvec, tvec, (px, mm)), keys), rejected = best, [best[0]]
            warn.append(f"marker {bad} does not fit the map (moved on the table?): left out")
    C = camera_centre(rvec, tvec)
    o, _, _, n = plane_frame(rig_map.plane)
    if n @ (C - o) <= 0:
        raise CalibrationError("the pose puts the camera below the table: mirrored PnP solution (camera too flat / "
                               "markers too small in the image). Raise the camera and retry.")
    allc = np.concatenate([seen[k] for k in keys])
    e_px = np.linalg.norm(project(K, dist, rvec, tvec, np.concatenate([rig_map.markers[k] for k in keys])) - allc, axis=1)
    e_mm = np.linalg.norm(backproject_to_plane(K, dist, rvec, tvec, rig_map.plane, allc)
                          - np.concatenate([rig_map.markers[k] for k in keys]), axis=1) * 1000
    return Reloc(rvec, tvec, keys, rejected, px, mm, float(np.sqrt(np.mean(e_px ** 2))),
                 float(np.sqrt(np.mean(e_mm ** 2))), C, sorted(k for k in keys if k in extra), {k: seen[k] for k in keys},
                 warnings=warn)


def locate(rig_map: RigMap, K, dist, images: list) -> Reloc:
    """Camera pose from a short burst of frames (BGR) of a camera that stands still.

    Corners are averaged over the frames (corners, not poses), one PnP gives a first pose from the
    markers the detector read, the remaining map markers are then looked for where that pose
    predicts them, and the PnP is repeated with everything. Raises CalibrationError when it cannot
    be computed (see relocalize) or when the pose would break the ball-plane convention.
    """
    mean, jitter = average_corners([_markers.detect(im) for im in images])
    res = relocalize(rig_map, K, dist, mean)
    grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images]
    extra, j2 = average_corners([guided_corners(g, rig_map, K, dist, res.rvec, res.tvec, have=mean) for g in grays])
    if extra:
        res = relocalize(rig_map, K, dist, mean, guided=extra)
    res.jitter_px = max([0.0, *jitter.values(), *j2.values()])
    h, w = grays[0].shape
    pixel_to_ball_xy(K, dist, res.rvec, res.tvec, rig_map.plane, [[w / 2, h / 2]])       # sign convention check
    return res


def pose_dict(res: Reloc, rig_map: RigMap, K, dist, width: int, height: int, serial: str | None = None) -> dict:
    """A relocalisation as JSON, for the log (outputs/calibration/camera_pose.json)."""
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "rig_map_time": rig_map.time, "camera_serial": serial,
            "width": int(width), "height": int(height), "K": np.asarray(K, float).tolist(),
            "dist": np.asarray(dist, float).ravel().tolist(), "rvec": res.rvec.tolist(), "tvec": res.tvec.tolist(),
            "camera_position": res.camera_position.tolist(), "status": res.status,
            "reproj_rms_px": round(res.reproj_rms_px, 3), "table_rms_mm": round(res.table_rms_mm, 2),
            "per_marker_reproj_px": {k: round(v, 3) for k, v in res.per_marker_px.items()},
            "markers_used": res.used, "markers_guided": res.guided, "markers_rejected": res.rejected}


class PoseMonitor:
    """Re-check a camera pose against the map while tracking: feed it marker detections every few
    seconds; it reports the median per-marker reprojection error and warns after two bad checks in a
    row (one check can be spoiled by a hand or the ball over a marker)."""

    def __init__(self, rig_map: RigMap, K, dist, rvec, tvec, warn_px: float = OK_PX, period: float = 3.0):
        self.map, self.K, self.dist, self.rvec, self.tvec = rig_map, K, dist, rvec, tvec
        self.warn_px, self.period = warn_px, period
        self.t_last, self.bad, self.median_px, self.n_markers = None, 0, None, 0

    def due(self, now: float) -> bool:
        return self.t_last is None or now - self.t_last >= self.period

    def detect(self, img_bgr: np.ndarray) -> dict:
        """Markers for update(): detected ones plus those found where the pose predicts them."""
        return detect_with_map(img_bgr, self.map, self.K, self.dist, (self.rvec, self.tvec))[0]

    def update(self, detections: dict, now: float) -> str | None:
        """`detections` from self.detect(img). Returns a warning text, or None while the pose still agrees with the map."""
        self.t_last = now
        px, _ = marker_errors(self.map, self.K, self.dist, self.rvec, self.tvec, detections)
        self.n_markers = len(px)
        if len(px) < 2:
            self.median_px = None
            return None                                  # not enough markers in view to judge
        self.median_px = float(np.median(list(px.values())))
        self.bad = self.bad + 1 if self.median_px > self.warn_px else 0
        return self.warning

    @property
    def warning(self) -> str | None:
        if self.bad >= 2:
            return f"CAMERA MOVED? markers are {self.median_px:.1f} px off the map: press c"
        return None


def latest_session() -> Path | None:
    ds = [d for d in SESSIONS_DIR.glob("*") if d.is_dir() and any((d / f).exists() for f in ("views.json", "touches.json"))]
    return max(ds, key=lambda d: d.stat().st_mtime) if ds else None
