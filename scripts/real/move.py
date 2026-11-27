"""Powered test: the open jaw passes along the intercept line at a few heights above the table.

    ./rig move                          # 60, 35, 20 mm above the table, y from -0.20 to +0.20 m
    ./rig move --heights 40 --y 0.10 --speed 0.05

The jaw is posed as when blocking (fingers forward, tilted down, facing the ball's side of the
table). The heights are above the measured table plane (config/rig_map.json): check them with a
ruler. Route: rest -> initial pose -> the passes, highest first, alternating direction -> lift ->
initial pose -> rest, then the motors are released.

The motors' own speed limit and current cap are the safety. ENTER powers the arm; Ctrl+C holds it
where it is, then ENTER releases it (support the arm first); a second Ctrl+C drops it at once and stops.
"""
from __future__ import annotations

import signal
import sys
signal.signal(signal.SIGINT, lambda *_: sys.exit(130))     # Ctrl+C while starting up: quiet (cli.run takes over)
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eaiproject import cli  # noqa: E402
from eaiproject import arm as arm_mod  # noqa: E402
from eaiproject import rig_calib as rc  # noqa: E402
from eaiproject.ball_intercept_env import BallInterceptEnv  # noqa: E402
from eaiproject.kinematics import Fingertip  # noqa: E402
from eaiproject.real_intercept import home_pose  # noqa: E402

MAX_IK_ERR = 0.003           # m: passes the arm reaches worse than this run anyway, with a warning


def plan(env: BallInterceptEnv, rig_map: rc.RigMap, heights, y_max: float, speed: float):
    """[(height, joint path)] along the intercept line, highest first, alternating direction; the
    path of each pass is eased, peaking at 1.5x `speed`. Also returns the largest IK error (m).
    y runs along the line in the map's table axes (rig_calib.TableFrame: the arm base's turn on the table)."""
    plane, frame = rig_map.plane, rig_map.frame
    q, err, out = home_pose(env, plane, frame, max(heights) - env.jaw_height), 0.0, []
    for i, h in enumerate(sorted(heights, reverse=True)):
        y0, y1 = (-y_max, y_max) if i % 2 == 0 else (y_max, -y_max)
        n = max(int(np.ceil(abs(y1 - y0) * 1.5 / speed * arm_mod.RATE)), 1)
        path = []
        for j, y in enumerate(np.r_[y0, y0 + arm_mod.ease(n) * (y1 - y0)]):
            x, yy, z = frame.line_point(plane, y, h, env.gate_x)
            q = env.ik_jaw(x, yy, z, yaw=frame.turn, q0=q, iters=400 if j == 0 else 40)
            err = max(err, env.jaw_error(q, x, yy, z))
            path.append(q)
        out.append((h, np.array(path)))
    return out, err


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heights", type=float, nargs="+", default=[60, 35, 20], help="fingertip heights above the table (mm)")
    ap.add_argument("--y", type=float, default=0.20, help="the passes go from -y to +y (m)")
    ap.add_argument("--speed", type=float, default=0.10, help="mean fingertip speed along the line (m/s)")
    args = ap.parse_args()
    if not rc.MAP_PATH.exists():
        print("config/rig_map.json not found: run ./rig calibrate first")
        return 2
    rig_map = rc.RigMap.load(rc.MAP_PATH)
    plane = rig_map.plane
    env, tip = BallInterceptEnv("forward"), Fingertip()
    passes, err = plan(env, rig_map, [h / 1000 for h in args.heights], args.y, args.speed)
    if err > MAX_IK_ERR:
        print(f"CHECK: the arm reaches the passes only within {err * 1000:.1f} mm (the heights printed below are what "
              "it really does); smaller --y or higher --heights fit better")
    cfg = arm_mod.load()
    secs = sum(len(p) for _, p in passes) / arm_mod.RATE
    print(f"{len(passes)} passes along the intercept line ({env.gate_x:.2f} m down the table from the base), "
          f"{-args.y:+.2f} .. {args.y:+.2f} m along it, "
          f"{', '.join(f'{h * 1000:.0f}' for h, _ in passes)} mm above the table, {secs:.0f} s of passes")
    print("route: " + " -> ".join(["rest" if cfg.rest is not None else "here"] + (["initial"] if cfg.initial is not None else [])
                                  + ["passes"] + (["initial"] if cfg.initial is not None else [])
                                  + ["rest" if cfg.rest is not None else "start"]) + ", then release")

    def height(q) -> float:
        p = tip(q)
        return (p[2] - rc.plane_z(plane, p[0], p[1])) * 1000

    with arm_mod.powered(cfg) as arm:
        for h, path in passes:
            q = arm.go(path[0])
            print(f"pass at {h * 1000:.0f} mm: start {height(q):.1f} mm above the table", flush=True)
            arm.follow(path[1:])
            q = arm.go(path[-1], settle=0.5)
            print(f"                end   {height(q):.1f} mm above the table", flush=True)
        x, y = tip(path[-1])[:2]
        z = rc.plane_z(plane, x, y) + max(passes[0][0], 0.06)
        arm.go(env.ik_jaw(x, y, z, yaw=rig_map.frame.turn, q0=path[-1], iters=400))        # lift straight up first
    return 0


if __name__ == "__main__":
    cli.run(main)
