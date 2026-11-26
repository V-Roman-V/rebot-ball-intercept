"""The report of a table-map solve: text, top-view layout, sweep heat map, reprojection overlays."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import rig_calib as rc
from .rig_draw import GREEN, GREY, MAGENTA, RED, WHITE, YELLOW, TopDown, colorbar, diverging, ipt, outline, put


def report_text(sol: rc.RigSolution) -> str:
    r, sw = sol.report, sol.report["sweep"]
    a, b, c = r["plane_z=a+bx+cy"]
    L = [f"=== table map ({sol.rig_map.time}) ===",
         f"{r['n_views']} views, {r['n_markers']} table markers, {r['n_touches']} corner touches, "
         f"{sw['used']} sweep samples in {r['n_sweep_cells']} cells of 2 cm",
         "",
         f"reprojection: RMS {r['reproj_rms_px']} px, max {r['reproj_max_px']} px   (accept: RMS < {rc.OK_PX} px)",
         "  per marker: " + "  ".join(f"{k} {v}" for k, v in r["per_marker_reproj_px"].items()),
         "  per view:   " + "  ".join(f"{k} {v}" for k, v in r["per_view_reproj_px"].items()),
         "",
         f"corner touches: RMS {r['touch_rms_mm']} mm, in the table plane {r['touch_xy_rms_mm']} mm   "
         f"(accept: RMS < {rc.OK_MM} mm)"]
    for t in r["touches"]:
        flag = "   <-- check" if t["err_mm"] > 2 * rc.OK_MM else ("   lifted: height ignored" if t["lifted"] else "")
        L.append(f"  {t['marker']:9s} {t['corner']:2s}  dx {t['dx_mm']:+6.2f}  dy {t['dy_mm']:+6.2f}  dz {t['dz_mm']:+6.2f}  "
                 f"|{t['err_mm']:5.2f}| mm{flag}")
    L += ["",
          f"scale check: the touched points are {r['touch_scale_percent']:+.2f} % wider apart than the camera's layout",
          f"table plane: z = {a * 1000:.1f} mm at the base, slope {np.degrees(np.arctan(b)):+.2f} deg along x, "
          f"{np.degrees(np.arctan(c)):+.2f} deg along y",
          f"sweep: {sw['samples']} samples; dropped {sw['lifted']} lifted off the table, {sw['below']} below it, "
          f"{sw['gross']} far off (> {rc.COARSE_LIFT * 1000:.0f} mm); scatter {sw['sigma_mm']} mm; "
          f"per-cell RMS {r['sweep_rms_mm']} mm",
          "table height on the intercept line (mm): " + "  ".join(f"y{y}: {v:.1f}" for y, v in r["table_on_line_mm"].items()),
          f"far markers (one must be in view to locate the camera): {r['far_markers']}",
          "", "WARNINGS:" if sol.warnings else "no warnings"] + [f"  - {w}" for w in sol.warnings]
    ok = not sol.warnings and r["reproj_rms_px"] < rc.OK_PX and r["touch_rms_mm"] < rc.OK_MM
    L += ["", f"RESULT: {'OK' if ok else 'CHECK THE WARNINGS'}"]
    return "\n".join(L)


def layout_image(sol: rc.RigSolution) -> np.ndarray:
    """Top view: fitted markers, touches (errors drawn 20x), camera positions, swept area."""
    top = TopDown()
    img = top.canvas(rc.GATE_X)
    for p in sol.sweep_points:
        top.cell(img, p[:2], 0.02, (60, 80, 60))
    for p in sol.sweep_rejected:
        cv2.circle(img, ipt(top.px(p[:2])[0]), 1, RED, -1)
    top.markers(img, sol.rig_map)
    for p, r in zip(sol.touch_points, sol.touch_residuals):
        a = top.px(p[:2] - r[:2])[0]                      # fitted corner
        b = top.px(p[:2] - r[:2] + 20 * r[:2])[0]
        cv2.circle(img, ipt(a), 3, GREEN, -1)
        cv2.line(img, ipt(a), ipt(b), MAGENTA, 1, cv2.LINE_AA)
    for name, (rv, tv) in sol.cams.items():
        C = rc.camera_centre(rv, tv)
        c = top.px(np.clip(C[:2], [top.x[0], top.y[0]], [top.x[1], top.y[1]]))[0]
        cv2.drawMarker(img, ipt(c), WHITE, cv2.MARKER_TRIANGLE_UP, 10, 1)
        put(img, f"{name.split('_')[-1]} h{C[2]:.2f}", c + [6, 4], GREY, 0.35)
    put(img, "table map, arm frame (m). yellow = touched markers, magenta = touch error x20, triangles = camera views, "
             "red dots = sweep samples dropped", (8, 16), WHITE, 0.42)
    put(img, f"reproj {sol.report['reproj_rms_px']} px   touches {sol.report['touch_rms_mm']} mm   "
             f"sweep {sol.report['sweep_rms_mm']} mm", (8, 34), WHITE, 0.42)
    return img


def heatmap_image(sol: rc.RigSolution, vmax_mm: float = 3.0) -> np.ndarray:
    """Sweep height residuals against the fitted plane, over the table."""
    top = TopDown(x=(0.10, 0.75), y=(-0.40, 0.40), scale=900.0)
    img = top.canvas(rc.GATE_X)
    for p, r in zip(sol.sweep_points, sol.sweep_residuals):
        top.cell(img, p[:2], 0.02, diverging(r * 1000, vmax_mm))
    for p in sol.sweep_rejected:
        cv2.circle(img, ipt(top.px(p[:2])[0]), 1, WHITE, -1)
    top.markers(img, sol.rig_map)
    colorbar(img, vmax_mm, (img.shape[1] - 80, 50))
    put(img, f"sweep height minus the table plane (mm); RMS {sol.report['sweep_rms_mm']} mm, "
             f"{len(sol.sweep_points)} cells of 2 cm; white dots = samples dropped (tip lifted)", (8, 16), WHITE, 0.45)
    put(img, "a smooth trend (one side red, one blue, or a bowl) = arm model error, not noise", (8, 34), WHITE, 0.42)
    return img


def reproj_images(sol: rc.RigSolution, obs: rc.Observations, sess: Path, out: Path) -> int:
    n = 0
    for v in obs.views:
        f = sess / f"{v.name}.png"
        if not f.exists() or v.name not in sol.cams:             # not in the solve: not linked to a touched marker
            continue
        img = cv2.imread(str(f))
        rv, tv = sol.cams[v.name]
        for c in v.corners.values():
            outline(img, c, GREEN, 1)
        for k, P in sol.rig_map.markers.items():
            px = rc.project(v.K, v.dist, rv, tv, P)
            outline(img, px, MAGENTA, 1, f"{k} {sol.report['per_marker_reproj_px'][k]}px" if k in v.corners else k)
        put(img, f"{v.name}: green = detected, magenta = fitted map; RMS {sol.report['per_view_reproj_px'][v.name]} px",
            (10, 24), YELLOW)
        cv2.imwrite(str(out / f"reproj_{v.name}.png"), img)
        n += 1
    return n


def save_report(sol: rc.RigSolution, obs: rc.Observations, sess: Path) -> tuple[Path, str]:
    """Write report.txt / report.json / layout.png / sweep_heatmap.png / reproj_<view>.png to <sess>/report."""
    out = Path(sess) / "report"
    out.mkdir(exist_ok=True)
    text = report_text(sol)
    (out / "report.txt").write_text(text + "\n")
    (out / "report.json").write_text(json.dumps(sol.report, indent=2))
    cv2.imwrite(str(out / "layout.png"), layout_image(sol))
    if len(sol.sweep_points):
        cv2.imwrite(str(out / "sweep_heatmap.png"), heatmap_image(sol))
    reproj_images(sol, obs, Path(sess), out)
    return out, text
