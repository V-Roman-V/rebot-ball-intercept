"""Excitation plan: holds, smooth moves, chirps and a held-out multisine, plus the offline plan check.

PACE data collection: fixed base, no contacts, all joints excited at the same time by chirp *position
targets* tracked by the motors' PD loop. The chirp amplitude falls with frequency so that velocity and
acceleration stay bounded: A(f) = min(A_max, V_max / (2 pi f), Acc_max / (2 pi f)^2).

Plan (default, ~2.7 min of motion):
    enable_hold     hold the start (rest) pose, ramp gravity feed-forward 0 -> 1
    to_A, settle_A  slow move to pose A, check the steady tracking error
    chirp_A_large   train   0.1 -> fmax Hz, 30 s, full amplitude
    chirp_A_small   train   same sweep at 35 % amplitude (friction / low-speed regime)
    multisine_A     val     held-out: 12 sines with random phases, 25 s
    to_B, settle_B
    chirp_B         train   second configuration (different inertia / gravity load), 70 % amplitude
    to_start, disable_hold  back to the start pose, ramp the feed-forward 1 -> 0
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .model import N, ArmModel, SimParams
from .replay import Replayer, Window

POSE_A = np.array([0.0, 1.2, 1.6, -0.6, 0.0, 0.0])        # jaw ~0.46 m above the table, in front of the base
POSE_B = np.array([0.4, 0.9, 1.2, 0.3, 0.3, 0.5])         # jaw ~0.62 m up, turned and wrist bent
A_MAX = np.array([0.35, 0.25, 0.25, 0.40, 0.40, 0.50])    # rad, amplitude at low frequency (scale 1)
V_MAX = np.array([1.0, 0.8, 0.8, 1.5, 1.5, 2.0])          # rad/s
ACC_MAX = np.array([8.0, 8.0, 8.0, 15.0, 15.0, 20.0])     # rad/s^2
SIGNS = np.array([1, -1, 1, -1, 1, -1], float)            # alternate directions between neighbouring joints
MOVE_SPEED = 0.3                                          # rad/s, peak joint speed of transfer moves
EXCITE_NAMES = ("chirp_A_large", "chirp_A_small", "multisine_A", "chirp_B")


@dataclass
class Segment:
    name: str
    kind: str                                    # hold / move / excite
    role: str                                    # train / val / "" (not used for fitting)
    duration: float
    q: Callable[[float], np.ndarray]             # t in [0, duration] -> joint targets
    q_end: np.ndarray
    ff: Callable[[float], float] = field(default=lambda t: 1.0)   # gravity feed-forward scale


def _ramp(t: float, t0: float, t1: float) -> float:
    s = min(max((t - t0) / (t1 - t0), 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * s)


def hold(name: str, q, T: float, ff: Callable[[float], float] | None = None) -> Segment:
    q = np.array(q, float)
    return Segment(name, "hold", "", T, lambda t: q, q, ff or (lambda t: 1.0))


def move(name: str, q0, q1, v_peak: float = MOVE_SPEED) -> Segment:
    """Minimum-jerk move; peak joint speed = 1.875 |dq| / T <= v_peak."""
    q0, q1 = np.array(q0, float), np.array(q1, float)
    T = max(1.0, 1.875 * float(np.max(np.abs(q1 - q0))) / v_peak)

    def q(t: float) -> np.ndarray:
        s = min(max(t / T, 0.0), 1.0)
        return q0 + (q1 - q0) * (10 * s**3 - 15 * s**4 + 6 * s**5)
    return Segment(name, "move", "", T, q, q1)


def amplitude(f, scale: float) -> np.ndarray:
    w = 2 * np.pi * np.maximum(f, 1e-6)
    return np.minimum(np.minimum(A_MAX * scale, V_MAX / w), ACC_MAX / w**2)


def _window(t: float, T: float, fade: float) -> float:
    return _ramp(t, 0.0, fade) * (1.0 - _ramp(t, T - fade, T))


def chirp(name: str, role: str, center, T: float = 30.0, f0: float = 0.1, f1: float = 4.0, scale: float = 1.0,
          pre: float = 1.5, post: float = 1.0, fade: float = 1.5) -> Segment:
    """Exponential sweep f(t) = f0 (f1/f0)^(t/T), all joints at once, with holds before and after."""
    c = np.array(center, float)
    k = f1 / f0

    def q(t: float) -> np.ndarray:
        s = t - pre
        if s <= 0 or s >= T:
            return c
        f = f0 * k ** (s / T)
        ph = 2 * np.pi * f0 * T / math.log(k) * (k ** (s / T) - 1)
        return c + SIGNS * amplitude(f, scale) * math.sin(ph) * _window(s, T, fade)
    return Segment(name, "excite", role, pre + T + post, q, c)


def multisine(name: str, role: str, center, T: float = 25.0, f_lo: float = 0.2, f_hi: float = 4.0, n: int = 12,
              scale: float = 1.0, seed: int = 7, pre: float = 1.5, post: float = 1.0, fade: float = 1.5) -> Segment:
    """Sum of n log-spaced sines with random phases per joint (a signal the fit never sees)."""
    c = np.array(center, float)
    rng = np.random.default_rng(seed)
    freqs = np.geomspace(f_lo, f_hi, n)
    amps = np.stack([amplitude(f, scale) for f in freqs]) * (1.6 / n)       # (n, 6); the plan check bounds it
    phases = rng.uniform(0, 2 * np.pi, (n, N))

    def q(t: float) -> np.ndarray:
        s = t - pre
        if s <= 0 or s >= T:
            return c
        x = (amps * np.sin(2 * np.pi * freqs[:, None] * s + phases)).sum(axis=0)
        return c + x * _window(s, T, fade)
    return Segment(name, "excite", role, pre + T + post, q, c)


def build_plan(q_start, fmax: float = 4.0, scale: float = 1.0, only: list[str] | None = None) -> list[Segment]:
    q_start = np.array(q_start, float)
    want = lambda n: only is None or n in only
    segs = [hold("enable_hold", q_start, 4.0, ff=lambda t: _ramp(t, 0.5, 2.5))]
    q = q_start
    a_group = [s for s in (
        chirp("chirp_A_large", "train", POSE_A, f1=fmax, scale=scale),
        chirp("chirp_A_small", "train", POSE_A, f1=fmax, scale=0.35 * scale),
        multisine("multisine_A", "val", POSE_A, f_hi=fmax, scale=scale),
    ) if want(s.name)]
    b_group = [s for s in (chirp("chirp_B", "train", POSE_B, f1=fmax, scale=0.7 * scale),) if want(s.name)]
    for pose, tag, group in ((POSE_A, "A", a_group), (POSE_B, "B", b_group)):
        if not group:
            continue
        segs += [move(f"to_{tag}", q, pose), hold(f"settle_{tag}", pose, 2.0)]
        segs += group
        q = pose
    segs += [move("to_start", q, q_start), hold("disable_hold", q_start, 2.5, ff=lambda t: 1.0 - _ramp(t, 0.5, 2.0))]
    return segs


def sample_plan(segs: list[Segment], rate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(t, q (K,6), ff (K,), segment index (K,)) on a uniform grid."""
    ts, qs, ffs, ids = [], [], [], []
    t0 = 0.0
    for i, s in enumerate(segs):
        n = max(1, int(round(s.duration * rate)))
        for k in range(n):
            t = k / rate
            ts.append(t0 + t)
            qs.append(s.q(t))
            ffs.append(s.ff(t))
            ids.append(i)
        t0 += n / rate
    return np.array(ts), np.array(qs), np.array(ffs), np.array(ids)


@dataclass
class PlanCheck:
    ok: bool
    problems: list[str]
    duration: float
    peak_vel: np.ndarray          # predicted |qd| max per joint
    peak_tau: np.ndarray          # predicted |tau| max per joint (before saturation)
    track_err: np.ndarray         # predicted max |q_cmd - q| per joint over the excitation segments
    q_lo: np.ndarray
    q_hi: np.ndarray
    pred: list = field(default_factory=list)   # per segment: (t from segment start, predicted q (T, 6))

    def table(self, cap: np.ndarray) -> str:
        lines = [f"{'joint':8s} {'q min deg':>10s} {'q max deg':>10s} {'peak vel rad/s':>15s} "
                 f"{'peak tau Nm':>12s} {'cap Nm':>7s} {'track err deg':>14s}"]
        for j in range(N):
            lines.append(f"joint{j + 1:<3d} {math.degrees(self.q_lo[j]):10.1f} {math.degrees(self.q_hi[j]):10.1f} "
                         f"{self.peak_vel[j]:15.2f} {self.peak_tau[j]:12.2f} {cap[j]:7.1f} "
                         f"{math.degrees(self.track_err[j]):14.2f}")
        return "\n".join(lines)


def check_plan(arm: ArmModel, segs: list[Segment], kp, kd, cap, rate: float,
               tau_frac: float = 0.7, margin: float = 0.05) -> PlanCheck:
    """Joint limits, collisions (kinematic, along the targets) and a nominal-model dynamic rollout."""
    problems: list[str] = []
    t, q, ff, ids = sample_plan(segs, rate)
    q_start = segs[0].q(0.0)                                   # the rest pose may sit at/near a hard stop
    lo = np.minimum(arm.range[:, 0] + margin, q_start - 1e-6)
    hi = np.maximum(arm.range[:, 1] - margin, q_start + 1e-6)
    for j in range(N):
        bad = (q[:, j] < lo[j]) | (q[:, j] > hi[j])
        if bad.any():
            k = int(np.argmax(bad))
            problems.append(f"joint{j + 1} target {math.degrees(q[k, j]):+.1f} deg in '{segs[ids[k]].name}' "
                            f"is outside the model range +-margin [{math.degrees(lo[j]):+.1f}, {math.degrees(hi[j]):+.1f}]")
    start_pairs = arm.contacts(q_start)                        # resting contacts at the start pose are allowed
    step = max(1, int(rate / 25))
    for k in range(0, len(t), step):
        new = arm.contacts(q[k]) - start_pairs
        if new:
            problems.append(f"model predicts contact {sorted(new)[0]} in '{segs[ids[k]].name}' at t = {t[k]:.1f} s")
            break

    tau_ff = np.stack([f * arm.gravity(qk) for f, qk in zip(ff, q)])
    w = Window("plan", "plan", t, q, np.zeros_like(q), tau_ff, q[0], np.full((1, N), np.nan), np.zeros((1, N)))
    rep = Replayer(arm, [w], kp, kd, cap, nbatch=1, nthread=1, preroll=0.2)
    nominal = arm.nominal
    qs, qds = rep.simulate([nominal], 0)
    pr = rep.prep[0]
    ctrl = rep._ctrl(w, pr["t_ctrl"], nominal)[:, arm.act]
    tau = np.asarray(kp) * (ctrl - qs[0]) - np.asarray(kd) * qds[0]
    peak_tau = np.abs(tau).max(axis=0)
    peak_vel = np.abs(qds[0]).max(axis=0)
    # tracking error on the excitation segments, sim state vs command at the same time
    q_cmd_grid = np.stack([np.interp(pr["t_state"], t, q[:, j]) for j in range(N)], axis=1)
    seg_grid = np.interp(pr["t_state"], t, ids).round().astype(int)
    exc = np.isin(seg_grid, [i for i, s in enumerate(segs) if s.kind == "excite"])
    err = np.abs(q_cmd_grid - qs[0])
    track_err = err[exc].max(axis=0) if exc.any() else err.max(axis=0)
    cap = np.asarray(cap, float)
    for j in range(N):
        if peak_tau[j] > tau_frac * cap[j]:
            problems.append(f"joint{j + 1} predicted torque {peak_tau[j]:.1f} Nm > {tau_frac:.0%} of the "
                            f"{cap[j]:.1f} Nm cap: lower --scale / --fmax or raise the cap")
    rep.close()
    pred = []
    for i in range(len(segs)):
        sel = seg_grid == i
        t0 = t[ids == i][0] if np.any(ids == i) else 0.0
        pred.append((pr["t_state"][sel] - t0, qs[0][sel]))
    return PlanCheck(not problems, problems, float(t[-1]), peak_vel, peak_tau, track_err,
                     q.min(axis=0), q.max(axis=0), pred)
