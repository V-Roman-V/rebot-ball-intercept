"""Live ball tracking on the real rig, arm in SHADOW mode (nothing is sent to the motors).

Shows the RealSense image with: detection, track, velocity, predicted path, the intercept line and
the predicted crossing point, and where the gripper WOULD go (fingertip target). A side panel renders
the MuJoCo arm in the pose the planner would command. Each roll is scored: the predicted crossing
point at 0.5 / 0.3 / 0.15 s before the ball reaches the line vs where it actually crossed.

    ./rig track                         # the camera is switched to 640x480 @ 60

The camera pose is found at start from the table markers and config/rig_map.json (hands and arm away
from the markers for the first second). It is re-checked every few seconds in the background; a red
warning means the camera was moved: press c.

Keys: c  locate the camera again (after it was moved)    r  reset the tracker    q  quit (prints the roll statistics)

The ball colour is the one saved in ./rig calibrate, step 5 (with the colour mask and sliders there).
"""
from __future__ import annotations

import signal
import sys
signal.signal(signal.SIGINT, lambda *_: sys.exit(130))     # Ctrl+C while starting up: quiet (cli.run takes over)
import json
import threading
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eaiproject import cli  # noqa: E402
from eaiproject.real_intercept import (HostClock, InterceptPlanner, camera_calib, hsv_saved_time,  # noqa: E402
                                       locate_camera, make_tracker, table_z)
from eaiproject import rig_calib  # noqa: E402
from eaiproject.rig_draw import put, window_closed  # noqa: E402

WIN = "real tracking (shadow arm)"
PRED_AT = (0.5, 0.3, 0.15)          # seconds before crossing at which the prediction is scored


class Roll:
    """One roll: prediction history and the actual crossing."""

    def __init__(self):
        self.preds: list[tuple[float, float]] = []      # (time to cross, predicted y)
        self.actual = None
        self.speed = None


def setup(cam, rig_map):
    """Locate the camera on the marker map; returns (calibration, tracker, pose monitor)."""
    res = locate_camera(cam, rig_map)
    print(f"camera located: {res.status}, {res.reproj_rms_px:.2f} px / {res.table_rms_mm:.1f} mm on {len(res.used)} markers "
          f"{res.used}; position x {res.camera_position[0]:.2f} y {res.camera_position[1]:.2f} z {res.camera_position[2]:.2f} m",
          flush=True)
    for w in res.warnings:
        print(f"  CHECK: {w}", flush=True)
    calib = camera_calib(cam, res)
    return calib, make_tracker(calib, rig_map.plane), rig_calib.PoseMonitor(rig_map, calib.K, calib.dist, calib.rvec, calib.tvec)


def main() -> int:
    from eaiproject.realsense_client import RealSenseClient, switch_mode
    if not rig_calib.MAP_PATH.exists():
        print("config/rig_map.json not found: run ./rig calibrate first")
        return 2
    rig_map = rig_calib.RigMap.load(rig_calib.MAP_PATH)
    plane = rig_map.plane
    try:
        cam = switch_mode(RealSenseClient(), 640, 480, 60)
    except OSError:
        print("Camera server not running: ./rig camera start")
        return 2
    try:
        calib, trk, monitor = setup(cam, rig_map)
    except rig_calib.CalibrationError as e:
        print(f"CANNOT LOCATE THE CAMERA: {e}")
        cam.close()
        return 1
    planner = InterceptPlanner(plane)
    gx = planner.gate_x
    clock = HostClock()
    mon_thread, mon_warned = None, False

    # MuJoCo shadow render
    m, d = planner.env.model, mujoco.MjData(planner.env.model)
    rend = mujoco.Renderer(m, 360, 360)
    vcam = mujoco.MjvCamera()
    vcam.lookat[:], vcam.azimuth, vcam.elevation, vcam.distance = (0.3, 0, 0.1), -150, -25, 1.1
    arm_adr = planner.env.arm_qadr

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    when = hsv_saved_time()
    print(f"ball colour: {'saved ' + when if when else 'the default (./rig calibrate, step 5)'}", flush=True)

    def proj(P):
        return calib.project(np.atleast_2d(P)).round().astype(int)

    ball_z = lambda x, y: table_z(plane, x, y) + 0.022          # height of the ball centre
    last_fn, rolls, roll, t_seen = None, [], None, time.monotonic()
    frame_i, sim = 0, None
    fps, t_prev = 0.0, time.monotonic()
    msg, msg_t = "", 0.0
    cli.ctrl_c_quits()
    while True:
        f = cam.read(timeout=2.0, newer_than=last_fn)
        last_fn = f.frame_number
        t_cap = clock(f.hw_ts_ms, f.t_host)
        now = time.monotonic()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
        t_prev = now

        det = trk.process(f.img, t_cap)
        if det is not None:
            t_seen = now
        if monitor.due(now) and (mon_thread is None or not mon_thread.is_alive()):
            monitor.t_last = now
            mon_thread = threading.Thread(target=lambda im=f.img, t=now: monitor.update(monitor.detect(im), t), daemon=True)
            mon_thread.start()
        if bool(monitor.warning) != mon_warned:
            mon_warned = bool(monitor.warning)
            print(monitor.warning or "camera pose agrees with the map again", flush=True)
        if trk.x is not None and now - (trk.t_last_meas or now) > 0.6:
            trk.reset()                                   # lost for a while: start over
        p = planner.plan(trk, now)

        # ---- roll bookkeeping
        st = trk.state_at(now) if trk.ready else None
        moving = st is not None and rig_calib.table_xy(st.vel)[0] < -0.15     # rolling down the table toward the arm
        if moving and roll is None:
            roll = Roll()
            roll.t0 = now
        if roll is not None and now - roll.t0 > 8.0:          # never crossed (stopped / rolled off): drop
            roll = None
        if roll is not None:
            if p.phase == "tracking" and p.crossing_y is not None:
                roll.preds.append((p.t_cross, p.crossing_y))
                roll.speed = float(np.linalg.norm(st.vel))
            hist = [(t, rig_calib.table_xy(xy)) for t, xy in trk.history[-2:]]     # table axes
            if len(hist) >= 2 and hist[-1][1][0] < gx <= hist[-2][1][0]:     # crossed between the last 2 measurements
                (t1, a), (t2, b) = hist[-2], hist[-1]
                s = (a[0] - gx) / (a[0] - b[0])
                roll.actual = float(a[1] + s * (b[1] - a[1]))
            if roll.actual is not None or (not moving and trk.x is None):
                if roll.actual is not None and roll.preds:
                    errs = {}
                    for tb in PRED_AT:
                        cand = [yy for tc, yy in roll.preds if tc <= tb]
                        if cand:
                            errs[tb] = (cand[0] - roll.actual) * 1000
                    rolls.append({"actual_y": roll.actual, "speed": roll.speed, "err_mm": errs})
                    e = "  ".join(f"{tb:.2f}s:{v:+.0f}mm" for tb, v in errs.items())
                    msg, msg_t = f"roll {len(rolls)}: crossed y={roll.actual*100:+.1f} cm at {roll.speed:.2f} m/s; pred err {e}", now
                    print(msg, flush=True)
                roll = None

        # ---- drawing
        img = f.img.copy()
        line_pts = proj([planner.line_point(y) for y in (-0.30, 0.30)])
        cv2.line(img, tuple(line_pts[0]), tuple(line_pts[1]), (255, 255, 255), 1)
        if det is not None:
            cv2.circle(img, tuple(np.round(det.uv).astype(int)), int(max(det.radius_px, 4)) + 3, (0, 255, 0), 2)
        if len(trk.history) > 1:
            pts = proj([[xy[0], xy[1], ball_z(*xy)] for _, xy in trk.history[-60:]])
            cv2.polylines(img, [pts], False, (0, 200, 255), 1)
        if st is not None:
            c = proj([[*st.pos, ball_z(*st.pos)]])[0]
            tipv = proj([[*(st.pos + st.vel * 0.25), ball_z(*st.pos)]])[0]
            cv2.arrowedLine(img, tuple(c), tuple(tipv), (255, 0, 255), 2)
            if p.crossing_y is not None:
                cp = proj([planner.line_point(p.crossing_y, 0.022)])[0]
                cv2.line(img, tuple(c), tuple(cp), (255, 0, 255), 1)
                cv2.drawMarker(img, tuple(cp), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        jw = proj([p.jaw])[0]
        along = np.r_[rig_calib.arm_xy([[0.0, 0.05]])[0], 0.0]          # 5 cm along the line
        jl = proj([p.jaw - along])[0]
        jr = proj([p.jaw + along])[0]
        cv2.line(img, tuple(jl), tuple(jr), (0, 255, 255), 3)          # open jaw, 10 cm
        cv2.circle(img, tuple(jw), 3, (0, 255, 255), -1)
        status = ("TRACKING" if det is not None else "COASTING") if trk.ready else ("ACQUIRING" if trk.x is not None else "NO BALL")
        col = {"TRACKING": (0, 255, 0), "COASTING": (0, 200, 255), "ACQUIRING": (0, 200, 255), "NO BALL": (160, 160, 160)}[status]
        put(img, f"{status}  {fps:4.1f} fps  frame age {(now - f.t_host)*1000:3.0f} ms  CV {np.mean(trk.timing_ms[-30:]):.1f} ms",
            (8, 18), col)
        if st is not None:
            bx, by = rig_calib.table_xy(st.pos)
            put(img, f"ball {bx*100:5.1f} cm down the table, {by*100:+5.1f} cm across  v {np.linalg.norm(st.vel):.2f} m/s", (8, 38))
        if p.crossing_y is not None:
            put(img, f"crossing y {p.crossing_y*100:+5.1f} cm in {p.t_cross*1000:4.0f} ms  -> jaw y {p.jaw_y*100:+5.1f} cm",
                (8, 58), (255, 0, 255))
        put(img, f"SHADOW: arm not powered | jaw target z {(p.jaw[2]-table_z(planner.plane, p.jaw[0], p.jaw[1]))*1000:.0f} mm above table"
                 f" | IK err {p.ik_err*1000:.1f} mm", (8, img.shape[0] - 10), (0, 255, 255), 0.45)
        if monitor.warning:
            put(img, monitor.warning, (8, 82), (0, 0, 255), 0.5, 2)
        elif monitor.median_px is not None:
            put(img, f"camera pose check: {monitor.median_px:.1f} px on {monitor.n_markers} markers", (8, 78), (160, 160, 160), 0.4)
        if msg and now - msg_t < 6:
            put(img, msg, (8, img.shape[0] - 30), (255, 255, 0), 0.45)
        if now - t_seen > 10:
            put(img, "no ball seen for 10 s: ball on the table? colour: ./rig calibrate --step ball", (8, 102), (0, 200, 255), 0.45)
        # shadow arm render: every 3rd frame (it is the slow part of the loop)
        frame_i += 1
        if sim is None or frame_i % 3 == 0:
            d.qpos[:] = m.qpos0
            d.qpos[arm_adr] = p.q
            d.qpos[arm_adr[0]] -= planner.j1_turn          # the sim scene's table is in front: undo the base's turn
            for jn in ("joint7", "joint_left", "joint_right"):
                d.qpos[m.joint(jn).qposadr[0]] = 0.05
            mujoco.mj_kinematics(m, d)
            rend.update_scene(d, vcam)
            sim = cv2.cvtColor(rend.render(), cv2.COLOR_RGB2BGR)
            put(sim, f"shadow arm: {p.phase}", (8, 20), (0, 255, 255))
            sim = cv2.resize(sim, (img.shape[0], img.shape[0]))
        cv2.imshow(WIN, np.hstack([img, sim]))

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q") or window_closed(WIN) or cli.ctrl_c():
            break
        if k == ord("c"):
            try:
                calib, trk, monitor = setup(cam, rig_map)
                planner.reset()
                roll = None
                msg, msg_t = "camera located again", now
            except rig_calib.CalibrationError as e:
                msg, msg_t = f"cannot locate the camera: {e}"[:110], now
                print(msg, flush=True)
        if k == ord("r"):
            trk.reset()
            planner.reset()
            roll = None
    cv2.destroyAllWindows()
    cam.close()
    if rolls:
        print(f"\n{len(rolls)} rolls scored")
        for tb in PRED_AT:
            e = np.array([r["err_mm"][tb] for r in rolls if tb in r["err_mm"]])
            if len(e):
                print(f"  crossing prediction {tb:.2f} s before the line: mean |err| {np.abs(e).mean():.1f} mm, "
                      f"max {np.abs(e).max():.1f} mm (gripper margin +-28 mm)")
        out = ROOT / "outputs" / "hardware" / f"tracking_rolls_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rolls, indent=2))
        print("saved", out)
    return 0


if __name__ == "__main__":
    cli.run(main)
