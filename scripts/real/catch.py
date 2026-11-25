"""Powered interception: the arm blocks the rolling ball with its open jaw.

    ./rig catch                         # the camera is switched to 640x480 @ 60

Before: the table map and today's arm and ball steps (./rig calibrate); open the jaw by hand (the
gripper motor is not powered). The camera is located from the table markers (hands off them).
ENTER powers the arm: rest -> initial pose -> the waiting pose at the intercept line; then the jaw
follows the planner (the simulation's policy on the tracked ball). Roll the ball.

Keys (window): q finish (initial pose -> rest -> release), r reset the tracker.
Ctrl+C in the terminal holds the arm where it is; ENTER then releases it (support the arm first).
The motors' own speed limit and current cap are the safety.
Commands and ball states are saved to outputs/hardware/catch_<time>.npz.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eaiproject import arm as arm_mod  # noqa: E402
from eaiproject import rig_calib as rc  # noqa: E402
from eaiproject.real_intercept import (BALL_R, HostClock, InterceptPlanner, camera_calib, locate_camera,  # noqa: E402
                                       make_tracker, table_z)
from eaiproject.rig_draw import GREEN, GREY, MAGENTA, WHITE, YELLOW, ipt, put, window_closed  # noqa: E402

WIN = "catch (q finish, r reset)"
PHASES = ("home", "tracking", "arrived")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint-speed", type=float, default=1.5, help="firmware speed limit of every joint (rad/s)")
    args = ap.parse_args()
    from eaiproject.realsense_client import RealSenseClient, switch_mode
    if not rc.MAP_PATH.exists():
        print("config/rig_map.json not found: run ./rig calibrate first")
        return 2
    rig_map = rc.RigMap.load(rc.MAP_PATH)
    plane = rig_map.plane
    try:
        cam = switch_mode(RealSenseClient(), 640, 480, 60)
    except OSError:
        print("Camera server not running: ./rig camera start")
        return 2
    try:
        res = locate_camera(cam, rig_map)
    except rc.CalibrationError as e:
        print(f"CANNOT LOCATE THE CAMERA: {e}")
        cam.close()
        return 1
    print(f"camera located: {res.status}, {res.reproj_rms_px:.2f} px on markers {res.used}", flush=True)
    calib = camera_calib(cam, res)
    trk, planner, clock = make_tracker(calib, plane), InterceptPlanner(plane), HostClock()
    gx = planner.gate_x
    proj = lambda P: calib.project(np.atleast_2d(np.asarray(P, float))).round().astype(int)
    log = {k: [] for k in ("t", "ball", "jaw", "q", "phase")}
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, cam.width * 2, cam.height * 2)
    try:
        with arm_mod.powered(speed=args.joint_speed) as arm:
            arm.go(planner.plan(trk, time.monotonic()).q)          # the waiting pose at the intercept line
            trk.reset()
            planner.reset()
            last_fn, last_phase, last_cross = None, "home", None
            print("following the ball: roll it. q in the window finishes, Ctrl+C in the terminal holds", flush=True)
            while True:
                f = cam.read(timeout=2.0, newer_than=last_fn)
                last_fn = f.frame_number
                now = time.monotonic()
                det = trk.process(f.img, clock(f.hw_ts_ms, f.t_host))
                if trk.x is not None and now - (trk.t_last_meas or now) > 0.6:
                    trk.reset()                                    # lost for a while: start over
                p = planner.plan(trk, now)
                arm.command(p.q)
                st = trk.state_at(now) if trk.ready else None
                log["t"].append(now)
                log["ball"].append(np.r_[st.pos, st.vel] if st is not None else np.full(4, np.nan))
                log["jaw"].append(p.jaw)
                log["q"].append(p.q)
                log["phase"].append(PHASES.index(p.phase))
                if p.phase == "tracking":
                    last_cross = p.crossing_y
                if p.phase == "arrived" and last_phase == "tracking":
                    print(f"ball at the line: predicted y {last_cross * 100:+.1f} cm, jaw at y {p.jaw[1] * 100:+.1f} cm", flush=True)
                last_phase = p.phase

                img = f.img.copy()
                a, b = proj([[gx, y, table_z(plane, gx, y)] for y in (-0.30, 0.30)])
                cv2.line(img, ipt(a), ipt(b), WHITE, 1)
                if det is not None:
                    cv2.circle(img, ipt(det.uv), int(max(det.radius_px, 4)) + 3, GREEN, 2)
                if st is not None and p.crossing_y is not None:
                    c = proj([[*st.pos, table_z(plane, *st.pos) + BALL_R]])[0]
                    cp = proj([[gx, p.crossing_y, table_z(plane, gx, p.crossing_y) + BALL_R]])[0]
                    cv2.line(img, ipt(c), ipt(cp), MAGENTA, 1)
                    cv2.drawMarker(img, ipt(cp), MAGENTA, cv2.MARKER_TILTED_CROSS, 14, 2)
                jl, jr = proj([p.jaw + [0, -0.03, 0], p.jaw + [0, 0.03, 0]])
                cv2.line(img, ipt(jl), ipt(jr), YELLOW, 3)
                put(img, f"POWERED  {p.phase}" + (f"  crossing y {p.crossing_y * 100:+.1f} cm in {p.t_cross * 1000:.0f} ms"
                                                  if p.crossing_y is not None else ""), (8, 18), YELLOW)
                put(img, "q finish   r reset   Ctrl+C (terminal) holds", (8, img.shape[0] - 10), GREY, 0.45)
                cv2.imshow(WIN, img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or window_closed(WIN):
                    break
                if key == ord("r"):
                    trk.reset()
                    planner.reset()
    finally:
        cv2.destroyAllWindows()
        cam.close()
        if log["t"]:
            log_dir = rc.OUT_DIR.parent / "hardware"                  # outputs/hardware
            log_dir.mkdir(parents=True, exist_ok=True)
            out = log_dir / f"catch_{time.strftime('%Y%m%d_%H%M%S')}.npz"
            np.savez(out, **{k: np.array(v) for k, v in log.items()}, plane=plane)
            print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
