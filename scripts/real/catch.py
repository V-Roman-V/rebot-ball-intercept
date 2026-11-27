"""Powered interception: the arm blocks the rolling ball with its open jaw.

    ./rig catch                         # the camera is switched to 640x480 @ 60

Before: the table map and today's arm and ball steps (./rig calibrate). The camera is located from
the table markers (hands off them). ENTER powers the arm: rest -> initial pose, where the gripper
opens (--grip, motor rad; --grip 0 leaves it alone: open it by hand) -> the waiting pose at the
intercept line; then the jaw
follows the planner (the simulation's policy on the tracked ball). Roll the ball.

Keys (window): q finish (initial pose -> rest -> release), r reset the tracker.
Ctrl+C in the terminal holds the arm where it is; ENTER then takes it back to rest and releases it,
a second Ctrl+C drops it at once and stops.
The crossing point is the tracker's straight-line prediction; the arrival time counts the ball's
measured slowdown (real_intercept.crossing).
--joint-speed is the fastest joint motion the planner asks for: it shortens its steps along the
intercept line instead of letting a limit cut the targets, so the jaw keeps its height.
--control mit (default): the motors' MIT mode with the simulation's PD gains and gravity feed-forward,
the controller of the system-identification run, so its fitted sim parameters describe this arm; the
torque is capped in the motors. --control position: the motors' own position mode and current cap.
In both, the arm's own speed limit is a backstop 25 % above --joint-speed.
Each roll is scored when the ball reaches the line: where it crossed against where the jaw was (the
measured joints in MIT control, the commanded pose otherwise), and a summary is printed at the end.
Commands, ball states and the roll scores are saved to outputs/hardware/catch_<time>.npz.
"""
from __future__ import annotations

import signal
import sys
signal.signal(signal.SIGINT, lambda *_: sys.exit(130))     # Ctrl+C while starting up: quiet (cli.run takes over)
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eaiproject import cli  # noqa: E402
from eaiproject import arm as arm_mod  # noqa: E402
from eaiproject import rig_calib as rc  # noqa: E402
from eaiproject.kinematics import Fingertip  # noqa: E402
from eaiproject.real_intercept import (BALL_R, HostClock, InterceptPlanner, camera_calib, locate_camera,  # noqa: E402
                                       make_tracker, table_z)
from eaiproject.rig_draw import (GREEN, GREY, MAGENTA, WHITE, YELLOW, ipt, open_window, put, show_note,  # noqa: E402
                                 window_closed)

WIN = "catch (q finish, r reset)"
PHASES = ("home", "tracking", "arrived")
JAW_HALF = 0.028             # m: the open jaw catches a ball within this of its middle
BACKSTOP = 1.25              # the arm's own speed limit / the planner's: the planner shapes the motion, not the limit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--height", type=float, default=20.0, help="fingertips above the table at the catch, mm (default 20; the ball is 44 mm tall: above its centre, 22 mm, it may squeeze under the fingers)")
    ap.add_argument("--grip", type=float, default=arm_mod.GRIP_OPEN,
                    help="gripper opening for the catch, motor rad (0.1 .. 5 = open; 0 = not powered, open it by hand)")
    ap.add_argument("--control", choices=("mit", "position"), default="mit",
                    help="mit: the sim's PD gains + gravity feed-forward (as identified); position: the motors' own loop")
    ap.add_argument("--joint-speed", type=float, default=3.0,
                    help="fastest joint motion the planner asks for (rad/s); the arm's own limit is set 25 %% above it")
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
    size = cam.width * 2, cam.height * 2
    open_window(WIN, *size, "locating the camera from the table markers (hands off them) ...")
    try:
        res = locate_camera(cam, rig_map)
    except rc.CalibrationError as e:
        print(f"CANNOT LOCATE THE CAMERA: {e}")
        cam.close()
        return 1
    print(f"camera located: {res.status}, {res.reproj_rms_px:.2f} px on markers {res.used}", flush=True)
    for w in res.warnings:
        print(f"  CHECK: {w}", flush=True)
    calib = camera_calib(cam, res)
    trk, clock = make_tracker(calib, rig_map), HostClock()
    planner = InterceptPlanner(plane, rig_map.frame, jaw_height=args.height / 1000, max_joint_speed=args.joint_speed)
    gx = planner.gate_x
    proj = lambda P: calib.project(np.atleast_2d(np.asarray(P, float))).round().astype(int)
    log = {k: [] for k in ("t", "ball", "meas", "ball_px", "jaw", "q", "q_sent", "q_meas", "phase")}
    rolls, tip = [], Fingertip()          # per roll: time, ball y at the line, jaw y, ball speed, crossing seen (1) or predicted (0)
    show_note(WIN, "ENTER in the terminal powers the arm; the picture starts when it waits at the line", *size)
    try:
        print(f"control: {args.control}", flush=True)
        with arm_mod.powered(speed=BACKSTOP * args.joint_speed, grip=args.grip or None, control=args.control) as arm:
            arm.go(planner.plan(trk, time.monotonic()).q)          # the waiting pose at the intercept line
            trk.reset()
            planner.reset()
            last_fn, last_phase, last_cross, last_meas_t = None, "home", None, None
            seen, speed = None, 0.0               # this roll: (ball y, jaw y) when the ball was seen crossing; its speed

            def jaw_y(p):                         # where the jaw is, along the line (table y)
                q = arm.measured() if args.control == "mit" else p.q
                return float(planner.frame.table_xy(tip(q if np.all(np.isfinite(q)) else p.q))[1])
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
                new = trk.history[-1] if trk.history and trk.history[-1][0] != last_meas_t else None
                last_meas_t = trk.history[-1][0] if trk.history else None
                log["meas"].append(np.r_[new[0], new[1]] if new is not None else np.full(3, np.nan))   # the tracker's raw measurement
                log["ball_px"].append(det.radius_px if det is not None else np.nan)   # its apparent size in the image
                log["jaw"].append(p.jaw)
                log["q"].append(p.q)
                log["q_sent"].append(arm.last.copy())
                log["q_meas"].append(arm.measured() if args.control == "mit" else np.full(6, np.nan))
                log["phase"].append(PHASES.index(p.phase))
                if p.phase == "tracking":
                    last_cross, speed = p.crossing_y, float(np.linalg.norm(st.vel))
                if new is not None and len(trk.history) >= 2:        # seen on both sides of the line: where it crossed
                    a, b = (planner.frame.table_xy(xy) for _, xy in trk.history[-2:])
                    if b[0] < gx <= a[0]:
                        seen = (float(a[1] + (a[0] - gx) / (a[0] - b[0]) * (b[1] - a[1])), jaw_y(p))
                if p.phase == "arrived" and last_phase == "tracking":
                    ball_y, jaw = seen if seen is not None else (last_cross, jaw_y(p))
                    off = (jaw - ball_y) * 1000
                    rolls.append((now, ball_y, jaw, speed, float(seen is not None)))
                    print(f"ball at the line: roll {len(rolls)}, {speed:.2f} m/s, crossed y {ball_y * 100:+.1f} cm"
                          f"{'' if seen is not None else ' (predicted)'}, jaw at y {jaw * 100:+.1f} cm: {off:+.0f} mm, "
                          + ("in the jaw" if abs(off) <= JAW_HALF * 1000 else
                             "beyond the reach" if abs(ball_y) > planner.max_y + JAW_HALF else "MISSED"), flush=True)
                if p.phase != "tracking" and last_phase != "tracking":
                    seen = None
                last_phase = p.phase

                img = f.img.copy()
                a, b = proj([planner.line_point(y) for y in (-0.30, 0.30)])
                cv2.line(img, ipt(a), ipt(b), WHITE, 1)
                if det is not None:
                    cv2.circle(img, ipt(det.uv), int(max(det.radius_px, 4)) + 3, GREEN, 2)
                if st is not None and p.crossing_y is not None:
                    c = proj([[*st.pos, table_z(plane, *st.pos) + BALL_R]])[0]
                    cp = proj([planner.line_point(p.crossing_y, BALL_R)])[0]
                    cv2.line(img, ipt(c), ipt(cp), MAGENTA, 1)
                    cv2.drawMarker(img, ipt(cp), MAGENTA, cv2.MARKER_TILTED_CROSS, 14, 2)
                along = np.r_[planner.frame.arm_xy([[0.0, 0.03]])[0], 0.0]   # 3 cm along the line
                jl, jr = proj([p.jaw - along, p.jaw + along])
                cv2.line(img, ipt(jl), ipt(jr), YELLOW, 3)
                put(img, f"POWERED  {p.phase}" + (
                    "" if p.crossing_y is None else f"  crossing y {p.crossing_y * 100:+.1f} cm " +
                    ("(stops before the line)" if p.t_cross is None else f"in {p.t_cross * 1000:.0f} ms")), (8, 18), YELLOW)
                put(img, "q finish   r reset   Ctrl+C (terminal) holds, twice drops", (8, img.shape[0] - 10), GREY, 0.45)
                cv2.imshow(WIN, img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or window_closed(WIN):
                    break
                if key == ord("r"):                          # forget the ball; the jaw goes back to the middle
                    trk.reset()
                    planner.recentre()
    finally:
        cv2.destroyAllWindows()
        cam.close()
        if log["t"]:
            log_dir = rc.OUT_DIR.parent / "hardware"                  # outputs/hardware
            log_dir.mkdir(parents=True, exist_ok=True)
            out = log_dir / f"catch_{time.strftime('%Y%m%d_%H%M%S')}.npz"
            np.savez(out, **{k: np.array(v) for k, v in log.items()}, plane=plane, rolls=np.array(rolls).reshape(-1, 5))
            if rolls:
                R = np.array(rolls)
                off, reach = np.abs(R[:, 2] - R[:, 1]), np.abs(R[:, 1]) <= planner.max_y + JAW_HALF
                ok = off <= JAW_HALF
                print(f"{len(R)} rolls: {int((ok & reach).sum())} of the {int(reach.sum())} inside the reach in the jaw "
                      f"(within {JAW_HALF * 1000:.0f} mm), {int((~reach).sum())} beyond the reach")
                for lo, hi in ((0.0, 0.5), (0.5, 0.8), (0.8, 9.9)):
                    m = reach & (R[:, 3] >= lo) & (R[:, 3] < hi)
                    if m.any():
                        print(f"  {lo:.1f}-{hi:.1f} m/s: {int((ok & m).sum())}/{int(m.sum())}, median miss {np.median(off[m]) * 1000:.0f} mm"
                              .replace("-9.9 m/s", " m/s and faster"))
            print("saved", out)
    return 0


if __name__ == "__main__":
    cli.run(main)
