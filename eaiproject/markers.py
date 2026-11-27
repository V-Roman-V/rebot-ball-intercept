"""The ArUco markers taped on the table, and their detection.

This file is the single place that defines them. To add a marker, print it from DICT at the size
you want and add one line to SIZES. Marker keys are the ID as a string, e.g. "107".
(The markers on the arm's mounting plate are not used.)
"""
from __future__ import annotations

import cv2
import numpy as np

DICT = cv2.aruco.DICT_ARUCO_ORIGINAL
MODULES = 7                 # marker width in cells, black border included (5x5 bits + border)
SIZES = {                   # ID -> side of the black square, metres (measured: printed at 71.5 mm)
    "107": 0.0715,          # where each one lies: config/rig_map.json (107 is the far one, ~0.76 m out)
    "880": 0.0715,
    "467": 0.0715,
    "718": 0.0715,
}

UPSCALE_BELOW = 1000        # px: an image narrower than this is searched at twice its size


def _detector() -> cv2.aruco.ArucoDetector:
    p = cv2.aruco.DetectorParameters()
    p.perspectiveRemovePixelPerCell = 8      # bits are read from 8 px per cell (default 4): steadier for cells ~3 px wide
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT), p)


_DETECTOR = _detector()


def fit_edges(gray: np.ndarray, corners, max_shift: float = 3.0, passes: int = 2) -> np.ndarray | None:
    """Sub-pixel marker corners from straight lines fitted to the marker's four outer edges.

    Start from approximate corners (4, 2) in OpenCV order (TL, TR, BR, BL). Edge positions are found
    across each side (gradient centroid, searched no deeper than the black border), a line is
    fitted per side, and the corners are the line intersections. OpenCV's own cornerSubPix pulls
    the corner of a blurred black square inwards by 0.3-0.8 px, which makes markers look 1-3 % too
    small; this does not. Returns None if an edge is not found or a corner would move more than
    `max_shift` px.
    """
    c0 = np.asarray(corners, np.float64).reshape(4, 2)
    c = c0
    for _ in range(passes):
        ctr, lines = c.mean(axis=0), []
        for i in range(4):
            a, b = c[i], c[(i + 1) % 4]
            L = float(np.linalg.norm(b - a))
            if L < 10:
                return None
            t = (b - a) / L
            nrm = np.array([-t[1], t[0]])
            if nrm @ (ctr - a) > 0:
                nrm = -nrm                                             # pointing out of the marker
            depth = 2 * abs(float(nrm @ (ctr - a)))                    # marker extent across this side (foreshortened)
            half = float(np.clip(0.6 * depth / MODULES, 1.0, 4.0))
            offs = np.arange(-half - 1, half + 1.01, 0.5)              # samples across the edge
            base = a + np.linspace(0.15, 0.85, max(6, int(L * 0.7 / 1.5)))[:, None] * (b - a)
            pts = (base[:, None, :] + offs[None, :, None] * nrm).astype(np.float32)
            prof = cv2.remap(gray, pts[..., 0], pts[..., 1], cv2.INTER_LINEAR).astype(np.float64)
            g = np.clip(np.diff(prof, axis=1), 0, None)                # black -> white going outwards
            wsum = g.sum(axis=1)
            ok = wsum > 0.25 * np.median(wsum)
            if np.median(wsum) <= 0 or ok.sum() < 4:
                return None
            e = (g[ok] * (0.5 * (offs[:-1] + offs[1:]))).sum(axis=1) / wsum[ok]
            vx, vy, x0, y0 = cv2.fitLine((base[ok] + e[:, None] * nrm).astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            lines.append((np.array([x0, y0], float), np.array([vx, vy], float)))
        out = np.empty((4, 2))
        for i in range(4):
            (p1, d1), (p2, d2) = lines[i - 1], lines[i]
            A = np.column_stack([d1, -d2])
            if abs(np.linalg.det(A)) < 1e-6:
                return None
            out[i] = p1 + np.linalg.solve(A, p2 - p1)[0] * d1
        if not np.all(np.isfinite(out)) or np.abs(out - c0).max() > max_shift:
            return None
        c = out
    return c


def edge_contrast(gray: np.ndarray, corners) -> float:
    """Grey level just outside the marker outline minus just inside it (the weakest of the 4 sides):
    large for a visible marker (white margin, black border), small when something covers it."""
    c = np.asarray(corners, np.float64).reshape(4, 2)
    ctr, diff = c.mean(axis=0), []
    for i in range(4):
        a, b = c[i], c[(i + 1) % 4]
        t = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
        nrm = np.array([-t[1], t[0]])
        if nrm @ (ctr - a) > 0:
            nrm = -nrm
        d = max(1.0, abs(float(nrm @ (ctr - a))) / MODULES)           # half a border width
        base = a + np.linspace(0.2, 0.8, 8)[:, None] * (b - a)
        pts = np.stack([base + d * nrm, base - d * nrm]).astype(np.float32)
        v = cv2.remap(gray, pts[..., 0], pts[..., 1], cv2.INTER_LINEAR).astype(float)
        diff.append(v[0].mean() - v[1].mean())
    return float(min(diff))


def detect(img_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Known table markers in the image: ID -> (4, 2) corner pixels, OpenCV order TL, TR, BR, BL.
    A small image (640x480) is searched at twice its size: a far marker is ~20 px wide there and
    flickered (read in 58 % of positions under 20 px, 94 % this way; offline study of the 27 Nov 2026
    views). The corners are always fitted on the image itself, so their accuracy does not change."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    up = 2 if gray.shape[1] < UPSCALE_BELOW else 1
    search = gray if up == 1 else cv2.resize(gray, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    corners, ids, _ = _DETECTOR.detectMarkers(search)
    out = {}
    if ids is not None:
        for c, i in zip(corners, ids.ravel()):
            if str(int(i)) in SIZES:
                c = (c.reshape(4, 2).astype(np.float64) + 0.5) / up - 0.5     # pixel centres of the searched image -> this one
                fine = fit_edges(gray, c)
                out[str(int(i))] = c if fine is None else fine
    return out


def draw(img: np.ndarray, found: dict[str, np.ndarray]) -> None:
    for key, c in found.items():
        pts = c.round().astype(int)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        cv2.circle(img, tuple(pts[0]), 3, (0, 0, 255), -1)
        cv2.putText(img, key, tuple(pts[0] + [0, -6]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
