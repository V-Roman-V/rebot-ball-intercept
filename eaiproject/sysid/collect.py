"""Module 1: run the excitation plan on the arm (real or sim), watch it, log everything.

Safety model
    HardAbort  fault bits, lost feedback (> 50 ms), second Ctrl+C, error while returning
               -> disable all motors at once (joints 2-3 drop: someone must support the arm).
    SoftAbort  tracking error, torque near the cap, over-temperature, joint range, speed, Ctrl+C
               -> keep the motors powered, move slowly back to the start pose, ramp the gravity
                  feed-forward down, then disable.
"""
from __future__ import annotations

import gc
import json
import math
import platform
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..robstride import GRIPPER_ID
from .arm_io import Feedback
from .excitation import Segment, hold, move, _ramp
from .model import MOTOR_IDS, N, ArmModel

ROOT = Path(__file__).resolve().parents[2]


class SoftAbort(RuntimeError):
    pass


class HardAbort(RuntimeError):
    pass


@dataclass
class Safety:
    track_excite: np.ndarray                 # rad, per joint (from the plan check)
    track_hold: float = 0.15                 # rad, holds and moves
    track_return: float = 0.30               # rad, while returning after an abort
    sustain: float = 0.25                    # s an error must last before it counts
    tau_frac: float = 0.92                   # of the cap
    tau_sustain: float = 0.30
    tau_sustain_return: float = 1.50         # while returning: ride out a short bump, still stop a real block
    temp_max: float = 75.0                   # deg C
    fb_timeout: float = 0.05                 # s without feedback from a joint
    q_margin: float = 0.03                   # rad outside the model range
    vel_max: np.ndarray = field(default_factory=lambda: np.array([3.0, 3.0, 3.0, 4.0, 4.0, 5.0]))


class Recorder:
    KEYS = ("t_cmd", "q_cmd", "tau_ff", "ff_scale", "seg", "fb_t", "fb_q", "fb_v", "fb_tau", "fb_temp", "fb_fault")

    def __init__(self):
        self.rows = {k: [] for k in self.KEYS}
        self.segments: list[dict] = []
        self._saver: threading.Thread | None = None

    def __len__(self) -> int:
        return len(self.rows["t_cmd"])

    def add(self, t, q_cmd, tau_ff, ff, seg, fb: Feedback) -> None:
        r = self.rows
        r["t_cmd"].append(t)
        r["q_cmd"].append(np.array(q_cmd, float))
        r["tau_ff"].append(np.array(tau_ff, float))
        r["ff_scale"].append(ff)
        r["seg"].append(seg)
        r["fb_t"].append(fb.t)
        r["fb_q"].append(fb.q)
        r["fb_v"].append(fb.v)
        r["fb_tau"].append(fb.tau)
        r["fb_temp"].append(fb.temp)
        r["fb_fault"].append(fb.fault)

    def arrays(self, rows: dict | None = None) -> dict[str, np.ndarray]:
        return {k: np.array(v) for k, v in (rows or self.rows).items()}

    def save(self, out: Path, meta: dict, rows: dict | None = None, segments: list | None = None) -> None:
        if self._saver is not None and rows is None:       # final save: let a checkpoint finish first
            self._saver.join()
        a = self.arrays(rows)
        n = len(a["t_cmd"])
        np.savez_compressed(out / "log.npz", **a, qd_cmd=np.zeros_like(a["q_cmd"]) if n else np.zeros((0, N)))
        meta = dict(meta, segments=segments if segments is not None else self.segments, n_cycles=n)
        (out / "meta.json").write_text(json.dumps(meta, indent=2, default=_json_default))

    def checkpoint(self, out: Path, meta: dict) -> None:
        """Save in a background thread (a compressed save takes ~0.2 s, longer than 40 control periods)."""
        if self._saver is not None and self._saver.is_alive():
            return
        rows = {k: list(v) for k, v in self.rows.items()}     # shallow snapshot, fast
        segs = [dict(s) for s in self.segments]
        self._saver = threading.Thread(target=self.save, args=(out, meta, rows, segs), daemon=True)
        self._saver.start()


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


class Runner:
    def __init__(self, io, arm: ArmModel, kp, kd, cap, rate: float, safety: Safety, rec: Recorder,
                 pred: list | None = None):
        self.io, self.arm, self.rate, self.safety, self.rec = io, arm, rate, safety, rec
        self.pred = pred or []
        self.q_start = arm.range.mean(axis=1)      # set by run_plan
        self.kp, self.kd, self.cap = np.asarray(kp, float), np.asarray(kd, float), np.asarray(cap, float)
        self.zero = np.zeros(N)
        self.last_ok = np.full(N, np.inf)
        self.err_since = np.full(N, np.nan)
        self.tau_since = np.full(N, np.nan)
        self.last_q = None                 # last measured pose (all joints valid)
        self.last_ff = 0.0
        self.overruns = 0
        self.returning = False

    # ------------------------------------------------------------------ one control cycle
    def cycle(self, q_cmd, ff: float, seg: int, level: str, q_ref=None) -> Feedback:
        tau = ff * self.arm.gravity(q_cmd)
        t = self.io.now()
        fb = self.io.command(q_cmd, self.zero, self.kp, self.kd, tau)
        self.rec.add(t, q_cmd, tau, ff, seg, fb)
        self.last_ff = ff
        if np.all(np.isfinite(fb.q)):
            self.last_q = fb.q.copy()
        self._check(t, np.asarray(q_cmd), fb, level, q_ref)
        return fb

    def _check(self, t: float, q_cmd: np.ndarray, fb: Feedback, level: str, q_ref=None) -> None:
        """q_ref: the nominal model's predicted response (excitation segments). The tracking check is
        then on the deviation from the prediction, not on the (large, expected) PD lag itself."""
        s = self.safety
        ok = np.isfinite(fb.q)
        if np.isinf(self.last_ok).any():
            self.last_ok = np.where(np.isinf(self.last_ok), t, self.last_ok)
        self.last_ok[ok] = t
        lost = np.where(t - self.last_ok > s.fb_timeout)[0]
        if lost.size:
            raise HardAbort(f"no feedback from joint{lost[0] + 1} for {1000 * (t - self.last_ok[lost[0]]):.0f} ms")
        if np.any(fb.fault[ok] != 0):
            j = int(np.where(ok & (fb.fault != 0))[0][0])
            raise HardAbort(f"joint{j + 1} reports fault bits 0x{int(fb.fault[j]):02x}")
        q = np.where(ok, fb.q, q_cmd)
        if np.nanmax(fb.temp) > s.temp_max:
            raise SoftAbort(f"motor temperature {np.nanmax(fb.temp):.0f} C > {s.temp_max:.0f} C")
        lo = np.minimum(self.arm.range[:, 0], self.q_start) - s.q_margin
        hi = np.maximum(self.arm.range[:, 1], self.q_start) + s.q_margin
        bad = np.where((q < lo) | (q > hi))[0]
        if bad.size:
            j = bad[0]
            raise SoftAbort(f"joint{j + 1} at {math.degrees(q[j]):+.1f} deg is outside the model range")
        v = np.where(ok, np.abs(fb.v), 0.0)
        if np.any(v > s.vel_max):
            j = int(np.argmax(v - s.vel_max))
            raise SoftAbort(f"joint{j + 1} speed {v[j]:.2f} rad/s > {s.vel_max[j]:.1f}")
        thr = {"excite": s.track_excite, "return": np.full(N, s.track_return)}.get(level, np.full(N, s.track_hold))
        ref = q_cmd if q_ref is None else q_ref
        err = np.abs(ref - q)
        over = err > thr
        self.err_since = np.where(over, np.where(np.isnan(self.err_since), t, self.err_since), np.nan)
        long = np.where(over & (t - self.err_since > s.sustain))[0]
        if long.size:
            j = long[0]
            what = "tracking error" if q_ref is None else "deviation from the predicted motion"
            raise SoftAbort(f"joint{j + 1} {what} {math.degrees(err[j]):.1f} deg > "
                            f"{math.degrees(thr[j]):.1f} deg for {s.sustain} s")
        tau = np.where(ok, np.abs(fb.tau), 0.0)
        over = tau > s.tau_frac * self.cap
        self.tau_since = np.where(over, np.where(np.isnan(self.tau_since), t, self.tau_since), np.nan)
        sustain = s.tau_sustain_return if level == "return" else s.tau_sustain
        long = np.where(over & (t - self.tau_since > sustain))[0]
        if long.size:
            j = long[0]
            raise SoftAbort(f"joint{j + 1} torque {tau[j]:.1f} Nm at the {self.cap[j]:.1f} Nm cap (blocked?)")

    # ------------------------------------------------------------------ segments
    def run_segment(self, seg: Segment, seg_idx: int, level: str | None = None) -> dict:
        level = level or ("excite" if seg.kind == "excite" else "hold")
        i0 = len(self.rec)
        t0 = self.io.now()
        info = {"name": seg.name, "kind": seg.kind, "role": seg.role, "i0": i0, "t0": t0, "duration": seg.duration,
                "complete": False}
        self.rec.segments.append(info)
        period = 1.0 / self.rate
        pred = self.pred[seg_idx] if level == "excite" and seg_idx < len(self.pred) else None
        k = 0
        try:
            while True:
                self.io.sleep_until(t0 + k * period)
                s = self.io.now() - t0
                if s > seg.duration:
                    break
                q_ref = None
                if pred is not None and len(pred[0]):
                    q_ref = np.array([np.interp(s, pred[0], pred[1][:, j]) for j in range(N)])
                self.cycle(seg.q(s), seg.ff(s), seg_idx, level, q_ref)
                k += 1
                late = self.io.now() - (t0 + k * period)
                if late > period:                       # fell behind by a whole period: skip ahead
                    self.overruns += 1
                    k = int((self.io.now() - t0) / period) + 1
            info["complete"] = True
        except KeyboardInterrupt:
            raise (HardAbort("Ctrl+C while returning") if self.returning else SoftAbort("Ctrl+C"))
        finally:
            info["i1"] = len(self.rec)
            info["t1"] = self.io.now()
        return info

    def hold_until_enter(self, q, ff: float, prompt: str, seg_idx: int) -> None:
        """Keep holding q (and checking) while waiting for Enter in the terminal."""
        if self.io.is_sim:
            return
        done = threading.Event()

        def wait():
            try:
                input(prompt)
            except EOFError:
                pass
            done.set()
        threading.Thread(target=wait, daemon=True).start()
        period = 1.0 / self.rate
        t = self.io.now()
        try:
            while not done.is_set():
                self.cycle(q, ff, seg_idx, "hold")
                t += period
                self.io.sleep_until(t)
        except KeyboardInterrupt:
            raise SoftAbort("Ctrl+C")

    def safe_return(self, q_start, seg_idx: int) -> None:
        """After a SoftAbort: slow move from where the arm is to the start pose, then ramp ff to 0."""
        self.returning = True
        self.err_since[:] = np.nan                 # timers from before the abort must not carry over
        self.tau_since[:] = np.nan
        q_from = self.last_q if self.last_q is not None else np.asarray(q_start)
        ff0 = self.last_ff
        back = move("abort_return", q_from, q_start, v_peak=0.25)
        back.ff = lambda t: ff0
        self.run_segment(back, seg_idx, "return")
        down = hold("abort_disable_hold", q_start, 2.0, ff=lambda t: ff0 * (1.0 - _ramp(t, 0.2, 1.5)))
        self.run_segment(down, seg_idx + 1, "return")


# ====================================================================================== preflight
def preflight(status: dict, arm: ArmModel, sim: bool) -> tuple[np.ndarray | None, list[str], list[str]]:
    """Returns (q_start, problems, warnings). Nothing is enabled."""
    problems, warns = [], []
    missing = [mid for mid in MOTOR_IDS if status[mid]["pos"] is None]
    if len(missing) == N:
        return None, ["No arm motor answered. Is the 48 V supply on and the CAN cable plugged into the arm?"], []
    if missing:
        problems.append(f"no answer from joint(s) {missing}")
    if status.get(GRIPPER_ID, {}).get("pos") is None:
        warns.append("gripper (ID 7) did not answer; it is not used, continuing")
    q = np.array([status[m]["pos"] if status[m]["pos"] is not None else np.nan for m in MOTOR_IDS])
    for j, mid in enumerate(MOTOR_IDS):
        s = status[mid]
        if s["pos"] is None:
            continue
        if s.get("faults"):
            problems.append(f"joint{mid} reports fault bits 0x{s['faults']:02x}")
        if s.get("fb_pos") is None:
            problems.append(f"joint{mid}: no feedback frame (cannot check scaling / temperature)")
        elif abs(s["fb_pos"] - s["pos"]) > 0.05:
            problems.append(f"joint{mid}: feedback position {s['fb_pos']:.3f} != mechPos {s['pos']:.3f} (scaling/ID?)")
        if s.get("drift") is not None:
            if abs(s["drift"]) > math.radians(0.1):
                problems.append(f"joint{mid} moved {math.degrees(s['drift']):+.2f} deg in 0.5 s; the arm must be at rest")
        elif s["vel"] is not None and abs(s["vel"]) > 0.05:
            problems.append(f"joint{mid} is moving ({s['vel']:+.2f} rad/s); the arm must be at rest")
        if s["iq"] is not None and abs(s["iq"]) > 0.3:
            problems.append(f"joint{mid} draws {s['iq']:+.2f} A: is it already enabled by another program?")
        if s["vbus"] is not None and not (40.0 <= s["vbus"] <= 55.0):
            problems.append(f"joint{mid} bus voltage {s['vbus']:.1f} V (expected ~48 V)")
        if s.get("temp") is not None and s["temp"] > 60.0:
            problems.append(f"joint{mid} is hot ({s['temp']:.0f} C); let it cool")
        lo, hi = arm.range[j]
        if not (lo - 0.1 <= s["pos"] <= hi + 0.1):
            problems.append(f"joint{mid} reads {math.degrees(s['pos']):+.1f} deg, outside the model range "
                            f"[{math.degrees(lo):+.1f}, {math.degrees(hi):+.1f}] -> is the arm in its rest pose, and "
                            f"the wrist zero today's (./rig calibrate --step arm)? ./rig arm shows the joints; "
                            f"--joint-offset {mid}=<reading in the zero pose, deg> sets it by hand")
    return (None if np.isnan(q).any() else q), problems, warns


def print_status(status: dict) -> None:
    f = lambda v, fmt: "   ----" if v is None else format(v, fmt)
    print(f"{'joint':8s} {'pos deg':>8s} {'vel rad/s':>10s} {'iq A':>7s} {'vbus V':>7s} {'temp C':>7s} "
          f"{'mode':>5s} {'limit_torque Nm':>16s}")
    for mid, s in status.items():
        name = "gripper" if mid == GRIPPER_ID else f"joint{mid}"
        deg = None if s["pos"] is None else math.degrees(s["pos"])
        print(f"{name:8s} {f(deg, '8.2f')} {f(s['vel'], '10.3f')} {f(s['iq'], '7.2f')} {f(s['vbus'], '7.1f')} "
              f"{f(s.get('temp'), '7.1f')} {f(s.get('run_mode'), '5d')} {f(s.get('limit_torque'), '16.1f')}")


def git_hash() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=2).stdout.strip() + (
            "-dirty" if subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True,
                                       text=True, timeout=2).stdout.strip() else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def base_meta(args, sim: bool, link: str) -> dict:
    return {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "sim" if sim else "real", "link": link,
            "host": platform.platform(), "python": sys.version.split()[0], "git": git_hash(),
            "argv": sys.argv, "rate_hz": args.rate, "fmax_hz": args.fmax, "scale": args.scale,
            "speed": getattr(args, "speed", 1.0)}


def run_plan(runner: Runner, segs: list[Segment], q_start: np.ndarray, checkpoint) -> str | None:
    """Runs all segments; returns the abort reason or None. Leaves the arm at q_start with ff 0."""
    io = runner.io
    runner.q_start = np.asarray(q_start, float)
    gc.disable()
    try:
        for i, seg in enumerate(segs):
            info = runner.run_segment(seg, i)
            if seg.name == "enable_hold":
                steady_check(runner, info, limit=0.08, what="after enabling at the start pose")
            elif seg.name.startswith("settle_"):
                steady_check(runner, info, limit=0.05, what=f"holding pose {seg.name[-1]}")
            elif seg.kind == "excite":
                print(f"  {seg.name:14s} done  {segment_line(runner, info)}")
                checkpoint()
            elif seg.kind == "move":
                print(f"  {seg.name:14s} done")
        return None
    except SoftAbort as e:
        print(f"\nSOFT ABORT: {e}\n  -> returning slowly to the start pose, motors stay powered")
        try:
            runner.safe_return(q_start, len(segs))
            return f"soft abort: {e}"
        except (SoftAbort, HardAbort) as e2:
            io.disable()
            return f"soft abort: {e}; then hard abort while returning: {e2}"
    except HardAbort as e:
        io.disable()
        return f"hard abort: {e}"
    finally:
        gc.enable()


def steady_check(runner: Runner, info: dict, limit: float, what: str) -> None:
    a = runner.rec.rows
    i0 = max(info["i0"], info["i1"] - int(runner.rate))          # last second of the segment
    q_cmd = np.array(a["q_cmd"][i0:info["i1"]])
    q = np.array(a["fb_q"][i0:info["i1"]])
    tau = np.array(a["fb_tau"][i0:info["i1"]])
    ff = np.array(a["tau_ff"][i0:info["i1"]])
    err = np.nanmean(q_cmd - q, axis=0)
    print(f"  steady state {what}:  " + "  ".join(
        f"j{j + 1} {math.degrees(err[j]):+.2f}deg/{np.nanmean(tau[:, j]):+.1f}Nm(ff {np.mean(ff[:, j]):+.1f})"
        for j in range(N)))
    bad = np.where(np.abs(err) > limit)[0]
    if bad.size:
        j = bad[0]
        raise SoftAbort(f"steady error joint{j + 1} {math.degrees(err[j]):+.1f} deg {what} "
                        f"(> {math.degrees(limit):.1f}): gravity feed-forward wrong or joint blocked")


def segment_line(runner: Runner, info: dict) -> str:
    a = runner.rec.rows
    sl = slice(info["i0"], info["i1"])
    t = np.array(a["t_cmd"][sl])
    q_cmd, q = np.array(a["q_cmd"][sl]), np.array(a["fb_q"][sl])
    miss = np.isnan(q).mean() * 100
    rms = np.sqrt(np.nanmean((q_cmd - q) ** 2, axis=0))
    tau = np.nanmax(np.abs(np.array(a["fb_tau"][sl])), axis=0)
    rate = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 else 0
    return (f"{rate:5.0f} Hz, missing {miss:.2f} %, rms err deg [{' '.join(f'{math.degrees(x):.2f}' for x in rms)}], "
            f"peak tau/cap [{' '.join(f'{x / c:.2f}' for x, c in zip(tau, runner.cap))}]")
