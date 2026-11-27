"""Drawing helpers shared by the rig calibration scripts (OpenCV only, no hardware)."""
from __future__ import annotations

import cv2
import numpy as np

from . import rig_calib as rc

GREEN, YELLOW, GREY, RED, MAGENTA, WHITE, CYAN = ((0, 255, 0), (0, 220, 255), (160, 160, 160), (0, 0, 255),
                                                  (255, 0, 255), (255, 255, 255), (255, 200, 0))
STATUS_COLOR = {"OK": GREEN, "WARN": (0, 165, 255), "FAIL": RED}


def put(img, text, org, color=WHITE, scale=0.55, thick=1):
    """Text with a dark halo. The halo is the same text shifted by a pixel: a thicker stroke would
    not line up, because OpenCV 5 changes the glyph spacing with the thickness."""
    x, y = int(org[0]), int(org[1])
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, 2), (2, 0)):
        cv2.putText(img, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def open_window(win: str, width: int, height: int, note: str = "starting ...") -> None:
    """Create the window and put it on screen at once, with `note` on a dark frame. macOS shows an
    OpenCV window only when it is first drawn, so a task that first locates the camera, opens the CAN
    adapter or moves the arm would run for seconds with an app icon and no window."""
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, width, height)
    show_note(win, note, width, height)


def show_note(win: str, note: str, width: int = 1280, height: int = 720) -> None:
    """`note` on a dark frame, drawn now (for the waits before a window has its picture)."""
    img = np.full((height // 2, width // 2, 3), 40, np.uint8)
    put(img, note, (16, img.shape[0] // 2), YELLOW, 0.6)
    cv2.imshow(win, img)
    cv2.pollKey()                                   # draws it; no key is read


def window_closed(win: str) -> bool:
    try:
        return cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def ipt(p) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def outline(img, corners_px, color, thick=1, label: str | None = None):
    pts = np.asarray(corners_px, float).round().astype(np.int32)
    cv2.polylines(img, [pts], True, color, thick, cv2.LINE_AA)
    cv2.circle(img, ipt(pts[0]), 3, color, -1)                      # TL
    if label:
        put(img, label, pts.min(axis=0) + [0, -6], color, 0.45)


def draw_map(img, rig_map: rc.RigMap, K, dist, rvec, tvec, color=MAGENTA, labels: dict | None = None):
    """Project the map's marker outlines into the image."""
    for k, P in rig_map.markers.items():
        outline(img, rc.project(K, dist, rvec, tvec, P), color, 1, None if labels is None else labels.get(k))


def draw_table_grid(img, K, dist, rvec, tvec, rig_map: rc.RigMap, gate_x=0.40, step=0.1):
    """10 cm grid on the map's table plane in its table axes (x down the table from the base), and the
    intercept line."""
    pr = lambda P: rc.project(K, dist, rvec, tvec, np.asarray(P, float))
    on = lambda x, y: rig_map.frame.line_point(rig_map.plane, y, 0.0, x)
    for x in np.arange(0.1, 1.01, step):
        a, b = pr([on(x, -0.3), on(x, 0.3)])
        cv2.line(img, ipt(a), ipt(b), CYAN, 1, cv2.LINE_AA)
        put(img, f"{x:.1f}", b + [4, 0], CYAN, 0.38)
    for y in np.arange(-0.3, 0.31, step):
        a, b = pr([on(0.1, y), on(1.0, y)])
        cv2.line(img, ipt(a), ipt(b), CYAN, 1, cv2.LINE_AA)
    a, b = pr([on(gate_x, -0.3), on(gate_x, 0.3)])
    cv2.line(img, ipt(a), ipt(b), WHITE, 2, cv2.LINE_AA)


def diverging(v: float, vmax: float) -> tuple[int, int, int]:
    """Blue (negative) - white - red (positive), BGR."""
    t = float(np.clip(v / vmax, -1, 1))
    k = int(255 * (1 - abs(t)))
    return (k, k, 255) if t > 0 else (255, k, k)


class TopDown:
    """Top view of the table in table axes (x down the table from the base, to the right; y up).
    px / cell / markers take arm-frame points; canvas, rect and the ranges x, y are in table axes."""

    def __init__(self, frame: rc.TableFrame, x=(-0.05, 1.05), y=(-0.42, 0.42), scale=640.0, margin=34):
        self.frame = frame
        self.x, self.y, self.s, self.m = x, y, scale, margin
        self.w = int((x[1] - x[0]) * scale) + 2 * margin
        self.h = int((y[1] - y[0]) * scale) + 2 * margin

    def pt(self, xy) -> np.ndarray:
        """Pixels of table-axes points."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        return np.c_[self.m + (xy[:, 0] - self.x[0]) * self.s, self.m + (self.y[1] - xy[:, 1]) * self.s]

    def px(self, xy) -> np.ndarray:
        """Pixels of arm-frame points."""
        return self.pt(self.frame.table_xy(np.asarray(xy, float).reshape(-1, 2)))

    def canvas(self, gate_x: float | None = 0.40) -> np.ndarray:
        img = np.full((self.h, self.w, 3), 32, np.uint8)
        for gx in np.arange(0.0, self.x[1] + 1e-6, 0.1):
            a, b = self.pt([[gx, self.y[0]], [gx, self.y[1]]])
            cv2.line(img, ipt(a), ipt(b), (64, 64, 64), 1)
            put(img, f"{gx:.1f}", (a[0] - 10, self.h - 10), GREY, 0.38)
        for gy in np.arange(-0.4, 0.41, 0.1):
            a, b = self.pt([[self.x[0], gy], [self.x[1], gy]])
            cv2.line(img, ipt(a), ipt(b), (64, 64, 64), 1)
            put(img, f"{gy:+.1f}", (2, a[1] + 4), GREY, 0.38)
        o, ex, ey = self.px([[0, 0], [0.08, 0], [0, 0.08]])             # the arm's x (red) and y (green)
        cv2.arrowedLine(img, ipt(o), ipt(ex), RED, 2)
        cv2.arrowedLine(img, ipt(o), ipt(ey), GREEN, 2)
        put(img, "base", o + [4, 18], GREY, 0.4)
        if gate_x is not None:
            a, b = self.pt([[gate_x, -0.30], [gate_x, 0.30]])
            cv2.line(img, ipt(a), ipt(b), WHITE, 1, cv2.LINE_AA)
            put(img, f"intercept line x={gate_x:.2f}", a + [6, 14], WHITE, 0.4)
        return img

    def cell(self, img, xy, size, color):
        """A square of `size` (table axes) around the arm-frame point xy."""
        c = self.frame.table_xy(np.asarray(xy, float)[:2])
        a = self.pt([[c[0] - size / 2, c[1] + size / 2]])[0]
        b = self.pt([[c[0] + size / 2, c[1] - size / 2]])[0]
        cv2.rectangle(img, ipt(a), ipt(b), color, -1)

    def rect(self, img, x, y, color, thick=1):
        a, b = self.pt([[x[0], y[1]], [x[1], y[0]]])
        cv2.rectangle(img, ipt(a), ipt(b), color, thick)

    def markers(self, img, rig_map: rc.RigMap):
        for k, P in rig_map.markers.items():
            pts = self.px(P[:, :2])
            col = YELLOW if k in rig_map.touched else GREY
            cv2.polylines(img, [pts.round().astype(np.int32)], True, col, 2, cv2.LINE_AA)
            cv2.circle(img, ipt(pts[0]), 4, col, -1)
            put(img, k, pts.mean(axis=0) + [-12, 5], col, 0.42)


def colorbar(img, vmax_mm: float, org=(0, 0), h=140):
    x0, y0 = org
    for i in range(h):
        cv2.line(img, (x0, y0 + i), (x0 + 12, y0 + i), diverging(vmax_mm * (1 - 2 * i / h), vmax_mm), 1)
    put(img, f"+{vmax_mm:.0f} mm", (x0 + 16, y0 + 8), WHITE, 0.38)
    put(img, "0", (x0 + 16, y0 + h // 2 + 4), WHITE, 0.38)
    put(img, f"-{vmax_mm:.0f} mm", (x0 + 16, y0 + h), WHITE, 0.38)
