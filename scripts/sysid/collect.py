"""Module 1 - record the arm's joint response to chirp position targets (PACE data collection).

All six arm joints run in the motors' MIT mode (kp, kd = our sim gains, gravity feed-forward from the
model). One run (~3 min of motion): enable at the rest pose -> pose A: two chirps + a held-out multisine
-> pose B: one chirp -> back to the rest pose -> disable. Everything is logged; a data-quality verdict
is printed at the end. Guide: docs/sysid.md.

Usage
    .venv/bin/python scripts/sysid/collect.py --sim                 # whole run on the MuJoCo stand-in (no hardware)
    .venv/bin/python scripts/sysid/collect.py --sim --inject track  # rehearse an abort + slow return
    .venv/bin/python scripts/sysid/collect.py --dry-run             # real arm: preflight + plan check, nothing enabled
    .venv/bin/python scripts/sysid/collect.py                       # real run (asks before enabling)

Output: outputs/sysid/<date_time>[_sim]/  log.npz  meta.json  quality.txt  segment_*.png
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eaiproject.sysid.collect import (HardAbort, Recorder, Runner, Safety, SoftAbort, base_meta,  # noqa: E402
                                      preflight, print_status, run_plan)
from eaiproject.sysid.excitation import EXCITE_NAMES, build_plan, check_plan  # noqa: E402
from eaiproject.sysid.model import (KD_DEFAULT, KP_DEFAULT, MOTOR_TORQUE_MAX, TORQUE_CAP_DEFAULT,  # noqa: E402
                                    ArmModel)
from eaiproject.sysid.report import plot_run, quality  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="run on the MuJoCo stand-in arm instead of the hardware")
    ap.add_argument("--dry-run", action="store_true", help="real arm: preflight and plan check only")
    ap.add_argument("--can", default="auto", choices=["auto", "pcan", "socketcan"],
                    help="CAN backend (auto: pcan on macOS, socketcan on Linux)")
    ap.add_argument("--channel", default=None, help="default PCAN_USBBUS1 (pcan) / can0 (socketcan)")
    ap.add_argument("--rate", type=float, default=200.0, help="command rate, Hz")
    ap.add_argument("--fmax", type=float, default=4.0, help="highest chirp frequency, Hz")
    ap.add_argument("--scale", type=float, default=1.0, help="excitation amplitude scale (0.2 .. 1.2)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="scale of the joint speed and acceleration the excitation reaches, 0.5 .. 2 (1 = up to 1.0 / "
                         "0.8 / 0.8 / 1.5 / 1.5 / 2.0 rad/s on joints 1-6). Faster motion needs more torque: the plan "
                         "check says if --torque-cap must be raised")
    ap.add_argument("--only", nargs="+", choices=EXCITE_NAMES, help="run only these excitation segments")
    ap.add_argument("--torque-cap", type=float, nargs=6, default=TORQUE_CAP_DEFAULT.tolist(),
                    help="limit_torque written to the motors for the run, Nm (joints 1-6)")
    ap.add_argument("--no-torque-cap", action="store_true", help="leave the motors' limit_torque unchanged")
    ap.add_argument("--yes", action="store_true", help="real arm: skip the confirmation prompts")
    ap.add_argument("--sim-start", type=float, nargs=6, default=[0, 0, 0, 0, 0, 0], help="sim: start pose, rad")
    ap.add_argument("--inject", default="none", choices=["none", "track", "lost"],
                    help="sim: inject a disturbance (track: push on joint 2) or lost feedback at t = 100 s")
    ap.add_argument("--joint-offset", action="append", default=[], metavar="J=DEG",
                    help="real arm: motor joint J reads DEG in the model's zero pose (repeatable). Default: from "
                         "the rig's calibration (wrist zero, base turn, the turn each motor counts)")
    ap.add_argument("--out", default=None, help="output directory (default outputs/sysid/<time>)")
    args = ap.parse_args()
    args.scale = min(max(args.scale, 0.2), 1.2)
    args.speed = min(max(args.speed, 0.5), 2.0)      # x2 stays below the run's speed abort (Safety.vel_max)
    args.fmax = min(max(args.fmax, 0.5), 8.0)
    kp, kd = KP_DEFAULT.copy(), KD_DEFAULT.copy()
    offset = np.zeros(6)
    cap = MOTOR_TORQUE_MAX.copy() if args.no_torque_cap else np.minimum(np.array(args.torque_cap), MOTOR_TORQUE_MAX)

    arm = ArmModel()
    if args.sim:
        from eaiproject.sysid.arm_io import SIM_TRUTH, SimArm
        io = SimArm(arm, args.sim_start, kp, kd, cap, inject=args.inject)
    else:
        from eaiproject.sysid.arm_io import RealArm, rig_offset
        try:
            io = RealArm(None if args.can == "auto" else args.can, args.channel, allow_motion=not args.dry_run,
                        offset=offset)
        except Exception as e:  # noqa: BLE001
            print(f"Cannot open the CAN adapter: {e}\n"
                  "  macOS: PCAN-USB plugged in, libPCBUSB in ~/lib (docs/real_rig.md).\n"
                  "  Linux: sudo ip link set can0 up type can bitrate 1000000")
            return 2
    print(f"CAN link: {io.link}")
    if not args.sim:
        # joint offsets: from the rig's calibration, then any typed with --joint-offset
        raw, note = io.raw_positions(), "no joint answered: no calibration offsets"
        if raw is not None:
            from eaiproject import arm as arm_mod
            from eaiproject import rig_calib
            cfg = arm_mod.load()
            yaw = rig_calib.RigMap.load(rig_calib.MAP_PATH).frame.yaw_deg if rig_calib.MAP_PATH.exists() else 0.0
            offset = rig_offset(raw, cfg, yaw)
            note = (f"wrist zero {cfg.info.get('level_time', 'not measured')}, base turn {yaw:+.0f} deg"
                    + ("" if rig_calib.MAP_PATH.exists() else " (no table map)"))
        for item in args.joint_offset:
            j, deg = item.split("=")
            offset[int(j) - 1] = np.radians(float(deg))
        io.offset = offset
        print(f"joint offsets (deg): {np.degrees(offset).round(2).tolist()}  ({note}"
              + (f"; typed: {', '.join(args.joint_offset)}" if args.joint_offset else "") + ")")

    try:
        # ------------------------------------------------ 1. preflight (nothing enabled)
        status = io.read_status()
        print_status(status)
        q_start, problems, warns = preflight(status, arm, args.sim)
        for w in warns:
            print("warning:", w)
        if problems:
            print("\nPREFLIGHT FAILED, nothing was enabled:")
            for p in problems:
                print("  -", p)
            return 2
        print(f"preflight OK. start pose (deg): {np.round(np.degrees(q_start), 1).tolist()}")

        # ------------------------------------------------ 2. plan + offline check in the model
        segs = build_plan(q_start, fmax=args.fmax, scale=args.scale, only=args.only, speed=args.speed)
        print(f"\nplan check in MuJoCo (nominal model, {args.rate:.0f} Hz) ...")
        chk = check_plan(arm, segs, kp, kd, cap, args.rate)
        print(chk.table(cap))
        print(f"plan: {len(segs)} segments, {chk.duration:.0f} s of motion: "
              + ", ".join(s.name for s in segs if s.kind == "excite"))
        if not chk.ok:
            print("\nPLAN CHECK FAILED, nothing was enabled:")
            for p in chk.problems:
                print("  -", p)
            return 2
        # excitation: allowed deviation from the nominal model's predicted motion = the whole predicted
        # PD lag (i.e. up to 100 % model error) + 3 deg, at least 7 deg
        safety = Safety(track_excite=np.maximum(0.12, chk.track_err + 0.05))
        print("abort if the measured motion deviates from the prediction by more than (deg): "
              + " ".join(f"{math.degrees(x):.0f}" for x in safety.track_excite))
        if args.dry_run:
            print("\ndry run: preflight and plan OK, nothing was enabled.")
            return 0

        # ------------------------------------------------ 3. confirm
        if not args.sim and not args.yes:
            print("\nBefore enabling:\n"
                  "  * the arm rests in its rest pose; nothing within its reach (~0.7 m), the jaw closed\n"
                  "  * one person keeps a hand near the 48 V switch for the whole run\n"
                  "  * joints 2-3 fall if the motors are disabled: on a HARD ABORT support the arm\n"
                  "  * Ctrl+C = stop safely (slow return); Ctrl+C again = disable at once")
            if input("Type 'yes' to enable the six arm motors: ").strip().lower() != "yes":
                print("cancelled, nothing was enabled")
                return 1

        out = Path(args.out) if args.out else ROOT / "outputs/sysid" / (
            time.strftime("%Y%m%d_%H%M%S") + ("_sim" if args.sim else ""))
        out.mkdir(parents=True, exist_ok=True)
        rec = Recorder()
        meta = base_meta(args, args.sim, io.link)
        meta["joint_offset_deg"] = np.degrees(offset).round(3).tolist()
        meta.update(kp=kp, kd=kd, torque_cap=cap, q_start=q_start, preflight=status, safety=vars(safety),
                    planned_excitation=[s.name for s in segs if s.kind == "excite"],
                    plan_check={"peak_tau": chk.peak_tau, "peak_vel": chk.peak_vel, "track_err": chk.track_err})
        if args.sim:
            meta["sim_truth"] = SIM_TRUTH.to_json()
            meta["sim_inject"] = args.inject
        runner = Runner(io, arm, kp, kd, cap, args.rate, safety, rec, pred=chk.pred)
        checkpoint = lambda: rec.checkpoint(out, meta)

        # ------------------------------------------------ 4. run
        abort = None
        try:
            io.prepare(None if args.no_torque_cap else cap)
            io.enable(q_start, kp, kd)
            print(f"\nmotors enabled, running ({chk.duration:.0f} s) ...")
            abort = run_plan(runner, segs, q_start, checkpoint)
            if abort is None or abort.startswith("soft"):
                runner.hold_until_enter(q_start, 0.0, "\nArm is back at the rest pose. Press Enter to disable "
                                        "the motors (support the arm if it does not rest by itself) ", len(segs) + 2)
        except (SoftAbort, HardAbort) as e:
            abort = f"{type(e).__name__}: {e}"
        except Exception as e:  # noqa: BLE001
            abort = f"error: {type(e).__name__}: {e}"
            raise
        finally:
            io.disable()
            for m in io.restore():
                print("warning:", m)
            meta["abort"] = abort
            meta["overruns"] = runner.overruns
            rec.save(out, meta)
            print("motors disabled" + ("" if abort is None else f"   ({abort})"))
            if abort and "hard abort" in abort:
                print("HARD ABORT: joints 2-3 were released. Lower the arm to its rest pose by hand.")

        # ------------------------------------------------ 5. data quality
        ok, text = quality(out)
        plots = plot_run(out)
        print("\n" + text)
        print(f"\nsaved: {out}" + (f"  ({len(plots)} plots)" if plots else ""))
        return 0 if ok else 3
    finally:
        io.close()


if __name__ == "__main__":
    sys.exit(main())
