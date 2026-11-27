"""Replay logged MIT commands in MuJoCo for a batch of parameter sets (mujoco.rollout, multithreaded C).

The command stream (time, q_cmd, qd_cmd, tau_ff) is applied as zero-order hold, shifted by each
candidate's delay, at the physics rate (1 kHz). The simulated joint positions are compared with the
measured ones at the real feedback timestamps.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass

import mujoco
import numpy as np
from mujoco import rollout

from .model import N, ArmModel, SimParams, mit_ctrl


@dataclass
class Window:
    """One excitation segment of a log (or of a plan), ready for replay."""
    name: str
    role: str                 # "train" / "val" / "plan"
    t_cmd: np.ndarray         # (K,)   command send times, s
    q_cmd: np.ndarray         # (K, 6)
    qd_cmd: np.ndarray        # (K, 6)
    tau_ff: np.ndarray        # (K, 6)
    q0: np.ndarray            # (6,)   measured pose at the window start (arm at rest under PD)
    t_fb: np.ndarray          # (M, 6) feedback receive times (NaN = missing)
    q_fb: np.ndarray          # (M, 6) measured positions

    @property
    def duration(self) -> float:
        return float(self.t_cmd[-1] - self.t_cmd[0])


class Replayer:
    def __init__(self, arm: ArmModel, windows: list[Window], kp, kd, torque_cap, nbatch: int,
                 nthread: int | None = None, preroll: float = 0.5, chunk_s: float = 5.0):
        self.arm, self.windows, self.nbatch = arm, windows, nbatch
        self.kp, self.kd = np.asarray(kp, float), np.asarray(kd, float)
        self.base = arm.make_dynamics_model(self.kp, self.kd, np.asarray(torque_cap, float))
        self.models = [copy.copy(self.base) for _ in range(nbatch)]
        nthread = max(1, min(nthread or os.cpu_count() or 1, nbatch))
        self.datas = [mujoco.MjData(self.base) for _ in range(nthread)]
        self.dt = self.base.opt.timestep
        self.nq, self.nu = self.base.nq, self.base.nu
        self.chunk = max(1, int(round(chunk_s / self.dt)))
        self.preroll_steps = int(round(preroll / self.dt))
        self.prep = [self._prepare(w) for w in windows]

    def _prepare(self, w: Window) -> dict:
        nstep = self.preroll_steps + int(np.ceil(w.duration / self.dt)) + 1
        t_ctrl = w.t_cmd[0] - self.preroll_steps * self.dt + np.arange(nstep) * self.dt
        t_state = t_ctrl + self.dt                            # state after step k
        # linear-interpolation indices from the uniform state grid to every feedback sample
        idx, wts, mask = [], [], []
        for j in range(N):
            t = w.t_fb[:, j]
            ok = np.isfinite(t) & np.isfinite(w.q_fb[:, j]) & (t >= t_state[0]) & (t <= t_state[-1])
            u = (t[ok] - t_state[0]) / self.dt
            lo = np.clip(np.floor(u).astype(int), 0, nstep - 2)
            idx.append(lo)
            wts.append(u - lo)
            mask.append(ok)
        init = self.arm.initial_state(self.base, w.q0)
        return dict(nstep=nstep, t_ctrl=t_ctrl, t_state=t_state, idx=idx, wts=wts, mask=mask, init=init,
                    n_samples=sum(int(m.sum()) for m in mask))

    def _ctrl(self, w: Window, t_ctrl: np.ndarray, p: SimParams) -> np.ndarray:
        k = np.clip(np.searchsorted(w.t_cmd, t_ctrl - p.delay, side="right") - 1, 0, len(w.t_cmd) - 1)
        c = np.zeros((len(t_ctrl), self.nu))
        c[:, self.arm.act] = mit_ctrl(w.q_cmd[k], w.qd_cmd[k], w.tau_ff[k], self.kp, self.kd, p.bias)
        return c

    def simulate(self, params: list[SimParams], wi: int) -> tuple[np.ndarray, np.ndarray]:
        """Joint positions and velocities on the state grid: (B, nstep, 6) each."""
        assert len(params) <= self.nbatch
        w, pr = self.windows[wi], self.prep[wi]
        B = len(params)
        for b, p in enumerate(params):
            p.apply_to(self.models[b])
        ctrl = np.stack([self._ctrl(w, pr["t_ctrl"], p) for p in params])
        q = np.empty((B, pr["nstep"], N))
        qd = np.empty((B, pr["nstep"], N))
        state0 = np.tile(pr["init"], (B, 1))
        qa = 1 + self.arm.qadr                                # FULLPHYSICS state: [time, qpos, qvel, ...]
        va = 1 + self.nq + self.arm.dadr
        for s in range(0, pr["nstep"], self.chunk):
            e = min(s + self.chunk, pr["nstep"])
            st, _ = rollout.rollout(self.models[:B], self.datas, state0, ctrl[:, s:e], persistent_pool=True)
            q[:, s:e] = st[:, :, qa]
            qd[:, s:e] = st[:, :, va]
            state0 = st[:, -1, :].copy()
        return q, qd

    def at_feedback(self, q: np.ndarray, wi: int) -> list[np.ndarray]:
        """Interpolate simulated positions to the feedback samples, per joint: list of (B, M_j)."""
        pr = self.prep[wi]
        return [q[:, pr["idx"][j], j] * (1 - pr["wts"][j]) + q[:, pr["idx"][j] + 1, j] * pr["wts"][j]
                for j in range(N)]

    def errors(self, params: list[SimParams], wi: int) -> tuple[np.ndarray, np.ndarray]:
        """Sum of squared position errors per joint (B, 6) and sample counts (6,) for one window."""
        q, _ = self.simulate(params, wi)
        sims = self.at_feedback(q, wi)
        w, pr = self.windows[wi], self.prep[wi]
        sse = np.stack([((s - w.q_fb[pr["mask"][j], j]) ** 2).sum(axis=1) for j, s in enumerate(sims)], axis=1)
        cnt = np.array([m.sum() for m in pr["mask"]], float)
        if not np.all(np.isfinite(sse)):                      # an unstable candidate: huge loss, not NaN
            sse = np.where(np.isfinite(sse), sse, 1e6)
        return sse, cnt

    def loss(self, params: list[SimParams]) -> np.ndarray:
        """PACE objective: time-averaged squared joint-position error over all windows, (B,) in rad^2."""
        sse_tot, cnt_tot = 0.0, 0.0
        for wi in range(len(self.windows)):
            sse, cnt = self.errors(params, wi)
            sse_tot = sse_tot + sse.sum(axis=1)
            cnt_tot += cnt.sum()
        return sse_tot / max(cnt_tot, 1.0)

    def rmse(self, p: SimParams, wi: int) -> np.ndarray:
        sse, cnt = self.errors([p], wi)
        return np.sqrt(sse[0] / np.maximum(cnt, 1))

    def close(self) -> None:
        if hasattr(rollout, "shutdown_persistent_pool"):
            rollout.shutdown_persistent_pool()
