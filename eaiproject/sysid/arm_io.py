"""Arm I/O for the collector: the real arm over CAN (RealArm) and a MuJoCo stand-in (SimArm).

Both have the same methods, so scripts/sysid/collect.py runs the identical sequence with --sim.

    read_status()            all joints: position, velocity, current, bus voltage, mode, temperature, faults
    prepare(torque_cap)      motors disabled: MIT run mode, limit_torque cap (RAM only)
    enable(q, kp, kd)        queue a hold at q, then enable joints 1-6
    command(q, qd, kp, kd, tau) -> Feedback    one MIT frame per joint, wait for the six replies
    disable(), restore(), close()
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

from ..robstride import (GRIPPER_ID, P_IQF, P_LIMIT_TORQUE, P_MECHPOS, P_MECHVEL, P_RUN_MODE, P_VBUS, P_MAX,
                         T_FEEDBACK, RobStrideBus)
from .model import MOTOR_IDS, MOTOR_TORQUE_MAX, N, ArmModel, SimParams, mit_ctrl, set_pd_actuator

RUN_MODE_MIT = 0


def rig_offset(raw, cfg, yaw_deg: float = 0.0) -> np.ndarray:
    """The offset (motor reading = model angle + offset, rad) of the arm as the rig is calibrated, from
    the motors' readings `raw` now. It replaces --joint-offset typed by hand.

    cfg (arm.ArmConfig): the joint's angle inside its limits, whatever turn the motor counts, plus
    the wrist zero. yaw_deg (the table map's base turn): this model has its table in front of joint
    1's zero, so the real base's turn is taken out of joint 1; the plan check then sees the table
    where it is. Joint 6: the two-finger gripper is the same rolled half a turn, so it takes the
    half turn nearest the model's zero, where the plan has room on both sides."""
    q = cfg.to_model(raw)
    q[0] -= np.radians(yaw_deg)           # joint 1's axis points down: facing along the arm's +y is -90 deg
    q[5] -= np.pi * np.round(q[5] / np.pi)
    return np.asarray(raw, float) - q


@dataclass
class Feedback:
    q: np.ndarray       # rad          NaN where a joint did not answer this cycle
    v: np.ndarray       # rad/s
    tau: np.ndarray     # N m (motor's estimate)
    temp: np.ndarray    # deg C
    fault: np.ndarray   # fault bits (int)
    t: np.ndarray       # receive time, s (same clock as now())


def _empty() -> Feedback:
    nan = np.full(N, np.nan)
    return Feedback(nan.copy(), nan.copy(), nan.copy(), nan.copy(), np.zeros(N, int), nan.copy())


# ============================================================================================ real
class RealArm:
    is_sim = False

    def __init__(self, interface: str | None = None, channel: str | None = None, allow_motion: bool = True,
                 offset=None):
        self.rs = RobStrideBus(channel, interface=interface, movable=MOTOR_IDS if allow_motion else ())
        # motor reading = model angle + offset (rad); applied to every read and every command
        self.offset = np.zeros(N) if offset is None else np.asarray(offset, float)
        self.orig: dict[int, dict] = {}
        self.enabled = False

    @property
    def link(self) -> str:
        return f"{self.rs.interface}:{self.rs.channel}"

    def now(self) -> float:
        return time.monotonic()

    def sleep_until(self, t: float) -> None:
        while True:
            rem = t - time.monotonic()
            if rem <= 0:
                return
            if rem > 0.002:
                time.sleep(rem - 0.0015)      # OS sleep, then spin the last ~1.5 ms for an even period

    def raw_positions(self) -> np.ndarray | None:
        """The six motors' own position readings (rad), or None if one does not answer."""
        q = [self.rs.read_param_f32(mid, P_MECHPOS) for mid in MOTOR_IDS]
        return None if any(v is None for v in q) else np.array(q, float)

    def read_status(self) -> dict:
        rs = self.rs
        st: dict = {}
        for mid in (*MOTOR_IDS, GRIPPER_ID):
            st[mid] = {"pos": rs.read_param_f32(mid, P_MECHPOS), "vel": rs.read_param_f32(mid, P_MECHVEL),
                       "iq": rs.read_param_f32(mid, P_IQF), "vbus": rs.read_param_f32(mid, P_VBUS)}
        time.sleep(0.5)   # at-rest check from the position change (the velocity reading is noisy, ~0.08 rad/s)
        for mid in MOTOR_IDS:
            p2 = rs.read_param_f32(mid, P_MECHPOS)
            st[mid]["drift"] = None if p2 is None or st[mid]["pos"] is None else p2 - st[mid]["pos"]
        for mid in MOTOR_IDS:
            st[mid]["run_mode"] = rs.read_param_u8(mid, P_RUN_MODE)
            st[mid]["limit_torque"] = rs.read_param_f32(mid, P_LIMIT_TORQUE)
        rs.last_fb.clear()
        for mid in (*MOTOR_IDS, GRIPPER_ID):   # a stop frame to a disabled motor returns one feedback frame
            rs.stop(mid)
            time.sleep(0.002)
        rs.drain(0.05)
        for mid in (*MOTOR_IDS, GRIPPER_ID):
            fb = rs.last_fb.get(mid)
            st[mid].update(fb_pos=None if fb is None else fb.pos, temp=None if fb is None else fb.temp_c,
                           faults=None if fb is None else fb.faults, mode=None if fb is None else fb.mode)
        for j, mid in enumerate(MOTOR_IDS):
            st[mid]["raw_pos"] = st[mid]["pos"]
            for k in ("pos", "fb_pos"):
                if st[mid][k] is not None:
                    st[mid][k] -= self.offset[j]
        return st

    def prepare(self, torque_cap: np.ndarray | None) -> None:
        """Motors must be disabled. RAM writes only; restore() puts the original values back."""
        rs = self.rs
        for j, mid in enumerate(MOTOR_IDS):
            o = {"run_mode": rs.read_param_u8(mid, P_RUN_MODE), "limit_torque": rs.read_param_f32(mid, P_LIMIT_TORQUE)}
            if o["run_mode"] is None:
                raise RuntimeError(f"joint{mid}: cannot read run_mode")
            self.orig[mid] = o
            rs.write_param_u8(mid, P_RUN_MODE, RUN_MODE_MIT)
            time.sleep(0.01)
            if rs.read_param_u8(mid, P_RUN_MODE) != RUN_MODE_MIT:
                raise RuntimeError(f"joint{mid}: run_mode did not switch to MIT")
            if torque_cap is not None:
                lt = o["limit_torque"]
                if lt is None or not (0.3 * MOTOR_TORQUE_MAX[j] <= lt <= 1.01 * MOTOR_TORQUE_MAX[j]):
                    raise RuntimeError(f"joint{mid}: limit_torque reads {lt}, not a plausible torque limit; "
                                       "not writing it (use --no-torque-cap to run without the cap)")
                rs.write_param_f32(mid, P_LIMIT_TORQUE, float(torque_cap[j]))
                time.sleep(0.01)
                back = rs.read_param_f32(mid, P_LIMIT_TORQUE)
                if back is None or abs(back - torque_cap[j]) > 1e-3:
                    raise RuntimeError(f"joint{mid}: limit_torque readback {back} != {torque_cap[j]}")

    def enable(self, q_hold: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
        for j, mid in enumerate(MOTOR_IDS):
            self.rs.mit(mid, float(q_hold[j] + self.offset[j]), float(kp[j]), float(kd[j]))   # hold target queued first
        self.rs.drain(0.005)
        for mid in MOTOR_IDS:
            self.rs.enable(mid)
        self.enabled = True
        self.rs.drain(0.005)

    def command(self, q, qd, kp, kd, tau) -> Feedback:
        rs = self.rs
        # Empty the receive queue first, so that the replies read below are this cycle's. Without it
        # every frame left over (a reply after the 4 ms wait, any other frame) stayed queued for good,
        # and the logged feedback fell behind by ~1.2 ms per second of run (27 Nov 2026: 180 ms after
        # 2.5 min), which a fit then explained as delay and rotor inertia.
        while (m := rs.bus.recv(0.0)) is not None:
            rs._handle(m)
        t_send = time.monotonic()
        for j, mid in enumerate(MOTOR_IDS):
            rs.mit(mid, float(q[j] + self.offset[j]), float(kp[j]), float(kd[j]), float(qd[j]), float(tau[j]))
        got: set[int] = set()
        deadline = t_send + 0.004
        while len(got) < N:
            rem = deadline - time.monotonic()
            if rem <= 0:
                break
            r = rs._handle(rs.bus.recv(rem))
            if r and r[0] == T_FEEDBACK and r[1] in MOTOR_IDS:
                got.add(r[1])
        fb = _empty()
        for j, mid in enumerate(MOTOR_IDS):
            f = rs.last_fb.get(mid)
            if f is not None and f.t >= t_send:
                fb.q[j], fb.v[j], fb.tau[j], fb.temp[j], fb.fault[j], fb.t[j] = (f.pos - self.offset[j], f.vel, f.tau,
                                                                                  f.temp_c, f.faults, f.t)
        return fb

    def disable(self) -> None:
        for _ in range(3):
            for mid in MOTOR_IDS:
                try:
                    self.rs.stop(mid)
                except Exception:  # noqa: BLE001  keep trying the other motors
                    pass
            time.sleep(0.005)
        self.enabled = False

    def restore(self) -> list[str]:
        msgs = []
        for mid, o in self.orig.items():
            try:
                if o.get("limit_torque") is not None:
                    self.rs.write_param_f32(mid, P_LIMIT_TORQUE, o["limit_torque"])
                    time.sleep(0.005)
                if o.get("run_mode") is not None:
                    self.rs.write_param_u8(mid, P_RUN_MODE, o["run_mode"])
                    time.sleep(0.005)
            except Exception as e:  # noqa: BLE001
                msgs.append(f"joint{mid}: could not restore params ({e}); they reset at power-off anyway")
        return msgs

    def close(self) -> None:
        self.rs.close()


# ============================================================================================ sim
SIM_TRUTH = SimParams(                       # hidden "real" parameters of the stand-in arm
    armature=np.array([0.030, 0.045, 0.035, 0.008, 0.006, 0.004]),
    damping=np.array([0.80, 1.20, 1.00, 0.30, 0.25, 0.20]),
    frictionloss=np.array([0.40, 0.60, 0.50, 0.15, 0.12, 0.10]),
    bias=np.array([0.010, -0.015, 0.020, -0.010, 0.005, 0.0]),
    delay=0.006,
)


class SimArm:
    """MuJoCo arm with hidden parameters, a virtual clock, sensor quantisation/noise and timing jitter.

    Unmodelled on purpose: link masses x mass_scale (gravity feed-forward is then slightly wrong, as on a
    real arm). inject: "none" | "track" (a 15 Nm push on joint 2 at t = 100 s for 0.8 s -> soft abort and return)
    | "lost" (joint 3 stops answering at t = 100 s -> hard abort).
    """
    is_sim = True
    link = "sim"

    def __init__(self, arm: ArmModel, q_start, kp, kd, cap, truth: SimParams = SIM_TRUTH,
                 mass_scale: float = 1.03, seed: int = 0, inject: str = "none"):
        self.arm, self.truth, self.inject = arm, truth, inject
        self.kp, self.kd = np.array(kp, float), np.array(kd, float)
        self.m = arm.make_dynamics_model(self.kp, self.kd, np.asarray(cap, float), truth)
        for b in range(1, self.m.nbody):
            if self.m.body(b).name not in ("base_link", "table"):
                self.m.body_mass[b] *= mass_scale
                self.m.body_inertia[b] *= mass_scale
        self.d = mujoco.MjData(self.m)
        arm.set_q(self.d, np.asarray(q_start, float))
        mujoco.mj_forward(self.m, self.d)
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.enabled = False
        self.queue: deque = deque()
        self.q_lsb = 2 * P_MAX / 65535

    def now(self) -> float:
        return self.t

    def sleep_until(self, t: float) -> None:
        self.t = max(self.t, t) + abs(self.rng.normal(0.0, 2e-4))     # OS wake-up jitter

    def _advance(self, t: float) -> None:
        m, d = self.m, self.d
        if not self.enabled:
            d.time = t                       # disabled: the arm rests where it is
            return
        while d.time + 0.5 * m.opt.timestep < t:
            while self.queue and self.queue[0][0] <= d.time:
                _, ctrl, kp, kd = self.queue.popleft()
                if not (np.array_equal(kp, self.kp) and np.array_equal(kd, self.kd)):
                    self.kp, self.kd = kp, kd
                    for j, a in enumerate(self.arm.act):
                        set_pd_actuator(m, a, kp[j], kd[j], m.actuator_forcerange[a, 1])
                d.ctrl[self.arm.act] = ctrl
            d.qfrc_applied[:] = 0
            if self.inject == "track" and 100.0 <= d.time < 100.8:
                d.qfrc_applied[self.arm.dadr[1]] = 15.0
            mujoco.mj_step(m, d)

    def read_status(self) -> dict:
        q = self.d.qpos[self.arm.qadr]
        st = {mid: {"pos": float(q[j]), "vel": 0.0, "iq": 0.0, "vbus": 48.0, "run_mode": 1,
                    "limit_torque": float(MOTOR_TORQUE_MAX[j]), "fb_pos": float(q[j]), "temp": 31.0, "faults": 0,
                    "mode": 0} for j, mid in enumerate(MOTOR_IDS)}
        st[GRIPPER_ID] = {"pos": 0.2, "vel": 0.0, "iq": 0.0, "vbus": 48.0, "fb_pos": 0.2, "temp": 30.0,
                          "faults": 0, "mode": 0}
        return st

    def prepare(self, torque_cap) -> None:
        pass

    def enable(self, q_hold, kp, kd) -> None:
        self.d.ctrl[self.arm.act] = mit_ctrl(np.asarray(q_hold), 0.0, 0.0, np.asarray(kp), np.asarray(kd),
                                             self.truth.bias)
        self.enabled = True

    def command(self, q, qd, kp, kd, tau) -> Feedback:
        self._advance(self.t)
        ctrl = mit_ctrl(np.asarray(q), np.asarray(qd), np.asarray(tau), np.asarray(kp), np.asarray(kd), self.truth.bias)
        self.queue.append((self.t + self.truth.delay, ctrl, np.array(kp, float), np.array(kd, float)))
        d, a = self.d, self.arm
        q_true = d.qpos[a.qadr] + self.rng.normal(0.0, 1e-4, N)
        fb = Feedback(np.round(q_true / self.q_lsb) * self.q_lsb,
                      d.qvel[a.dadr] + self.rng.normal(0.0, 0.01, N),
                      d.actuator_force[a.act] + self.rng.normal(0.0, 0.02, N),
                      np.full(N, 31.0), np.zeros(N, int),
                      self.t + 3e-4 + 1.5e-4 * np.arange(N))
        if self.inject == "lost" and self.t >= 100.0:
            fb.q[2] = fb.v[2] = fb.tau[2] = fb.t[2] = np.nan
        self.t += 8e-4                       # time on the bus for 12 frames
        return fb

    def disable(self) -> None:
        self.enabled = False

    def restore(self) -> list[str]:
        return []

    def close(self) -> None:
        pass
