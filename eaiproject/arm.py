"""The real arm: zero corrections and poses (outputs/calibration/arm.json), a read-only joint stream
for the calibration, and the motors' own position mode for the powered scripts.

    q_model = joint angle + offset        (per joint, rad; outside this module every angle is a MODEL angle)

The joint angle is the motor reading with whole turns taken off so that it lies inside the joint's
limits (`unwind`, `LIMITS`). Targets are kept inside the limits too (`ArmConfig.within`), so a move
between two poses never goes the long way round or through a stop.
`offset` corrects a motor whose zero differs from the model's. Only the wrist roll (joint 6) is
measured: with the jaw open and both fingertips resting on the table the jaw is level
(`level_offset`). Joint 6 turns about the finger axis and the fingertip lies on that axis, so the
correction moves no fingertip position: it only turns the jaw.

The calibration tools read the unpowered arm through `ArmLink`, which connects whenever the arm
answers on CAN: they start without it and say so on the pages that need it.

Two poses, stored as MOTOR readings so they stay right when the offset is measured again:
    rest     the arm lies on itself without support: it is released there
    initial  where the arm goes before its work and comes back to after it
Both are recorded by ./rig calibrate (step 4). The file is local (git-ignored), like the ball colour:
the wrist zero is measured again every day, and the last values stay the defaults.

Safety is the motors' own: position mode with a firmware speed limit and current cap, or MIT mode
with a torque cap (RAM-only parameter writes, gone at power-off). `powered()` asks for ENTER before
enabling; Ctrl+C holds the arm where it is, and another ENTER takes it back to rest and releases it.
"""
from __future__ import annotations

import json
import signal
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .robstride import ARM_IDS, GRIPPER_ID, P_MECHPOS, P_RUN_MODE, TARGET_MAX, TARGET_MIN

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "outputs" / "calibration" / "arm.json"
ROLL = 5                                  # index of joint 6, the wrist roll (turns about the finger axis)
FINGER_OPEN_AXIS = 1                      # the fingers open along gripper_end's y axis (model)

P_LOC_REF, P_LIMIT_SPD, P_LIMIT_CUR = 0x7016, 0x7017, 0x7018
RUN_MODE_POSITION = 1
SPEED = 0.8                               # rad/s, firmware speed limit of every joint
CUR_CAP = {1: 6.0, 2: 18.0, 3: 14.0, 4: 5.0, 5: 4.0, 6: 4.0}     # A, firmware current limit (2 and 3 carry the arm)
MOVE_SPEED = 0.3                          # rad/s, peak joint speed of go() (moves between poses)
RATE = 50.0                               # Hz, go() streams its targets at this rate
GRIP_OPEN = 4.5                           # motor rad: gripper opened for the catch (travel 0 closed .. 5 open)
GRIP_SPEED, GRIP_CUR = 2.0, 2.0           # rad/s, A: the gripper's firmware speed limit and current cap

# MIT control (Arm(control="mit")): the motors' MIT mode with the simulation's PD gains and the model's
# gravity as feed-forward, as in BallInterceptEnv and the system-identification run (scripts/sysid), so
# a fitted sim_params.json describes the arm under this controller. MIT mode has no firmware speed
# limit: Arm.command() limits the step to `speed` (rad/s) as a backstop, scaling the whole pose. The
# torque cap is the motors' limit_torque, written for the run and restored on release.
RUN_MODE_MIT = 0
P_LIMIT_TORQUE = 0x700B                   # Nm (checked on this arm's firmware by the sysid run, 27 Nov)
MIT_KP = np.array([80.0, 100.0, 100.0, 35.0, 25.0, 18.0])
MIT_KD = np.array([8.0, 10.0, 10.0, 4.0, 3.0, 2.5])
TORQUE_CAP = np.array([20.0, 20.0, 20.0, 7.0, 7.0, 7.0])     # Nm, as in the sysid run
MOTOR_TORQUE_MAX = np.array([36.0, 36.0, 36.0, 14.0, 14.0, 14.0])
FF_RAMP = 1.0                             # s, gravity feed-forward ramps in after power-on
# While the target stands still the damping gain is lowered: the motor multiplies kd by its own velocity
# reading, which is noisy at rest (0.03 .. 0.09 rad/s -> 0.5 .. 0.85 N m of torque jitter on joints 1-3,
# an audible buzz, 27 Nov 2026). A workaround until the motor's velocity can be filtered: the full
# gain is back with the first target that moves.
HOLD_KD = 0.3                             # fraction of MIT_KD while holding
HOLD_AFTER = 0.3                          # s of unchanged target before it applies


# ------------------------------------------------------------------------------------ arm.json
# Where each joint can be (rad), as angles of the joint itself, i.e. of the motor. Joints 1-5: the arm
# model's ranges (assets/rebot_b601_rs/mjcf/rs_arm.xml, checked against it in tests/test_arm.py); they
# have stops. Joint 6, the wrist roll, has none at the model's +-180 deg: on 27 Nov 2026 it ran from
# 130 to 296 deg during the catches. Its limit is what the motor can count, +-2 turns.
LIMITS = np.array([[-2.8, 2.8], [-0.2, 3.14], [-0.2, 3.14], [-1.57, 1.57], [-1.57, 1.57], [-4 * np.pi, 4 * np.pi]])


def unwind(q) -> np.ndarray:
    """Joint angles with whole turns taken off so that each lies in its joint's range (LIMITS). A motor
    counts turns (joint 1 has read 267 deg while standing at -93 deg), but a joint with stops cannot
    turn all the way round, so its range says which turn is meant. A joint that can (joint 6) keeps
    the motor's count."""
    q, mid, stops = np.array(q, float), LIMITS.mean(axis=1), np.ptp(LIMITS, axis=1) < 2 * np.pi
    return q - 2 * np.pi * np.round((q - mid) / (2 * np.pi)) * stops


@dataclass
class ArmConfig:
    offset: np.ndarray = field(default_factory=lambda: np.zeros(6))     # rad, q_model = q_motor + offset
    rest: np.ndarray | None = None                                      # motor readings, rad
    initial: np.ndarray | None = None
    time: str = ""
    info: dict = field(default_factory=dict)

    def to_model(self, q_motor) -> np.ndarray:
        """Model angles of motor readings: the joint's own angle (unwind) plus the offset."""
        return unwind(q_motor) + self.offset

    def to_motor(self, q_model) -> np.ndarray:
        """The joint's own angle for model angles (the motor may count whole turns on top: Arm.turns)."""
        return np.asarray(q_model, float) - self.offset

    def within(self, q_model, ref=None) -> np.ndarray:
        """The model pose to send for a wanted one: the same pose with every joint inside its limits.
        Whole turns are taken off (a joint with stops has one angle for a direction); the two-finger
        gripper is the same rolled half a turn, so with `ref` (where the arm is, model angles) joint 6
        takes the equivalent angle nearest to it among those inside the limits; what is still outside
        is clipped. A straight joint-space move between two such poses stays inside the limits, so no
        move goes the long way round or through a stop."""
        q = unwind(self.to_motor(q_model))
        if ref is not None:
            lo, hi = LIMITS[ROLL]
            same = q[ROLL] + np.pi * np.arange(np.ceil((lo - q[ROLL]) / np.pi), np.floor((hi - q[ROLL]) / np.pi) + 1)
            q[ROLL] = same[np.abs(same - self.to_motor(ref)[ROLL]).argmin()]
        return np.clip(q, LIMITS[:, 0], LIMITS[:, 1]) + self.offset

    def rest_model(self) -> np.ndarray | None:
        return None if self.rest is None else self.to_model(self.rest)

    def initial_model(self) -> np.ndarray | None:
        return None if self.initial is None else self.to_model(self.initial)

    def to_json(self) -> dict:
        deg = lambda q: None if q is None else np.round(np.degrees(q), 4).tolist()
        return {"version": 1, "time": self.time,
                "convention": "q_model = q_motor + offset; rest / initial are motor readings",
                "offset_deg": deg(self.offset), "rest_motor_deg": deg(self.rest), "initial_motor_deg": deg(self.initial),
                "info": self.info}

    @staticmethod
    def from_json(d: dict) -> "ArmConfig":
        rad = lambda v: None if v is None else np.radians(np.array(v, float))
        off = rad(d.get("offset_deg"))
        return ArmConfig(np.zeros(6) if off is None else off, rad(d.get("rest_motor_deg")),
                         rad(d.get("initial_motor_deg")), d.get("time", ""), d.get("info", {}))

    def save(self, path=None) -> Path:
        path = Path(path or PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.time = time.strftime("%Y-%m-%d %H:%M:%S")
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")
        return path


def load(path=None) -> ArmConfig:
    """The saved arm config; no file = no correction and no poses."""
    path = Path(path or PATH)
    return ArmConfig.from_json(json.loads(path.read_text())) if path.exists() else ArmConfig()


def level_offset(q_model, plane=(0.0, 0.0, 0.0), fk=None) -> tuple[float, float]:
    """Correction of joint 6 (rad, add it to the current offset) from a reading taken with the jaw
    open and both fingertips resting on the table: the jaw is then level, i.e. the axis along which
    the fingers open lies in the table plane. Turning joint 6 by d tilts that axis out of the plane
    as a cos d + b sin d, so the correction is the root of that nearest zero.

    Also returns how well the reading defines it: sin of the angle between the fingers and the
    table normal (0 = fingers straight down, then any roll keeps both fingertips on the table)."""
    from .kinematics import Fingertip
    fk = fk or Fingertip()
    a_, b_, c_ = plane
    n = np.array([-b_, -c_, 1.0]) / np.sqrt(1 + b_ * b_ + c_ * c_)
    q0 = np.asarray(q_model, float)

    def tilt(d):
        q = q0.copy()
        q[ROLL] += d
        fk(q)
        return float(n @ fk.d.xmat[fk.body].reshape(3, 3)[:, FINGER_OPEN_AXIS])

    a, b = tilt(0.0), tilt(np.pi / 2)
    d = float(np.arctan2(-a, b))
    if d > np.pi / 2:
        d -= np.pi
    elif d <= -np.pi / 2:
        d += np.pi
    return d, float(np.hypot(a, b))


# ------------------------------------------------------------------------- read-only stream
@dataclass
class ArmSample:
    t: float                # time.monotonic()
    q: np.ndarray           # (6,) model joint angles, rad
    p: np.ndarray           # (3,) fingertip, base frame


class ArmReader:
    """Joint stream of the UNPOWERED arm for the calibration: a background thread polls the six
    joints (parameter reads only: RobStrideBus(movable=()) refuses any enable / move / write) and
    keeps the last `history_s` seconds of model angles and fingertip positions."""

    def __init__(self, hz: float = 50.0, history_s: float = 20.0, offset=None, rs=None):
        from .kinematics import Fingertip
        from .robstride import RobStrideBus

        self.offset = load().offset if offset is None else np.asarray(offset, float)
        self.rs = rs if rs is not None else RobStrideBus(movable=())
        self._tip, self._tip_main = Fingertip(), Fingertip()
        self.period = 1.0 / hz
        self.history: deque[ArmSample] = deque(maxlen=int(hz * history_s))
        self._lock = threading.Lock()
        self._stop = False
        first = self._read()
        if first is None:
            self.rs.close()
            raise ConnectionError("arm not answering on CAN (powered? cable?)")
        self.history.append(first)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _read(self) -> ArmSample | None:
        q = [self.rs.read_param_f32(m, P_MECHPOS, timeout=0.03) for m in ARM_IDS]
        if any(v is None for v in q):
            return None
        q = np.array(q, float) + self.offset
        return ArmSample(time.monotonic(), q, self._tip(q))

    def _loop(self) -> None:
        nxt = time.monotonic()
        while not self._stop:
            try:
                s = self._read()
            except Exception:                            # bus closed / adapter unplugged: stop quietly
                break
            if s is not None:
                with self._lock:
                    self.history.append(s)
            nxt = max(nxt + self.period, time.monotonic() - self.period)
            time.sleep(max(0.0, nxt - time.monotonic()))

    def latest(self, max_age: float = 0.3) -> ArmSample | None:
        with self._lock:
            s = self.history[-1] if self.history else None
        return s if s is not None and time.monotonic() - s.t <= max_age else None

    def between(self, t0: float, t1: float) -> list[ArmSample]:
        with self._lock:
            return [s for s in self.history if t0 < s.t <= t1]

    def steady(self, window: float = 0.25) -> tuple[ArmSample, float] | None:
        """Mean of the last `window` seconds as one sample, and how far the fingertip moved in it (m)."""
        now = time.monotonic()
        ss = self.between(now - window, now)
        if len(ss) < 3:
            return None
        q = np.mean([s.q for s in ss], axis=0)
        P = np.array([s.p for s in ss])
        return ArmSample(now, q, self._tip_main(q)), float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))

    def close(self) -> None:
        self._stop = True
        self._thread.join(timeout=1.0)
        self.rs.close()


class ArmLink:
    """An ArmReader that connects by itself, so a tool can start without the arm: one try at once,
    then a try every RETRY_S in the background until the arm answers on CAN. Until then it has no
    readings and `status()` says why, for the page to show."""

    RETRY_S = 2.0

    def __init__(self, connect=None):
        self._connect = connect or ArmReader
        self.reader: ArmReader | None = None
        self.error = ""
        self._offset = None
        self._stop = threading.Event()
        if not self._try():
            threading.Thread(target=self._retry, daemon=True).start()

    def _try(self) -> bool:
        try:
            r = self._connect()
        except ConnectionError:
            self.error = "the arm does not answer (arm power on?)"
            return False
        except Exception:  # noqa: BLE001  (python-can raises several types)
            self.error = "no CAN adapter (plugged in?)"
            return False
        if self._offset is not None:
            r.offset = self._offset
        if self._stop.is_set():
            r.close()
        else:
            self.reader = r
        return True

    def _retry(self) -> None:
        while not self._stop.wait(self.RETRY_S) and not self._try():
            pass

    def status(self) -> str:
        """'' when connected; else what to tell the user."""
        return "" if self.reader is not None else f"ARM NOT CONNECTED: {self.error}. It connects by itself"

    @property
    def offset(self) -> np.ndarray:
        if self.reader is not None:
            return self.reader.offset
        return self._offset if self._offset is not None else load().offset

    @offset.setter
    def offset(self, value) -> None:
        self._offset = np.asarray(value, float)
        if self.reader is not None:
            self.reader.offset = self._offset

    def latest(self, max_age: float = 0.3) -> ArmSample | None:
        r = self.reader
        return None if r is None else r.latest(max_age)

    def between(self, t0: float, t1: float) -> list[ArmSample]:
        r = self.reader
        return [] if r is None else r.between(t0, t1)

    def steady(self, window: float = 0.25) -> tuple[ArmSample, float] | None:
        r = self.reader
        return None if r is None else r.steady(window)

    def close(self) -> None:
        self._stop.set()
        if self.reader is not None:
            self.reader.close()


# --------------------------------------------------------------------------------- powered
@contextmanager
def _ctrl_c_after():
    """Ctrl+C cannot cut the block short: it is raised right after it (CAN sequences that must not
    stop halfway: enabling, holding, releasing)."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    hit = []
    old = signal.signal(signal.SIGINT, lambda *_: hit.append(1))
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old)
    if hit:
        raise KeyboardInterrupt


def ease(n: int) -> np.ndarray:
    """n fractions from just above 0 to 1 that start and end at rest (smoothstep)."""
    u = np.linspace(0.0, 1.0, n + 1)[1:]
    return u * u * (3 - 2 * u)


class Gravity:
    """Gravity torques of the arm model (N m per joint) at model angles q, zero velocity."""

    def __init__(self):
        import mujoco
        from .ball_intercept_env import BallInterceptEnv
        env = BallInterceptEnv("forward")
        self.mj, self.m, self.qadr, self.dadr = mujoco, env.model, env.arm_qadr, env.arm_dadr
        self.d = mujoco.MjData(self.m)

    def __call__(self, q) -> np.ndarray:
        self.d.qpos[self.qadr] = q
        self.d.qvel[:] = 0.0
        self.mj.mj_forward(self.m, self.d)
        return self.d.qfrc_bias[self.dadr].copy()


class Arm:
    """The six arm motors, in the motors' position mode (control="position") or in MIT mode with the
    simulation's PD gains and gravity feed-forward (control="mit"). Angles are MODEL angles; the
    offset and the motors' whole turns are applied here, at the motor boundary. Every target is kept
    inside the joint limits (ArmConfig.within)."""

    def __init__(self, cfg: ArmConfig | None = None, rs=None, speed: float = SPEED, caps: dict | None = None,
                 gripper: bool = False, control: str = "position"):
        from .robstride import RobStrideBus
        assert control in ("position", "mit"), control
        self.cfg = cfg or load()
        self.rs = rs if rs is not None else RobStrideBus(movable=ARM_IDS + ((GRIPPER_ID,) if gripper else ()))
        self.speed, self.caps, self.control = speed, dict(caps or CUR_CAP), control
        self.enabled, self.grip_on = False, False
        self.last = None                                  # last target sent (model)
        self.turns = np.zeros(6)                          # whole turns each motor counts on top of its joint's angle (rad)
        self.t_last = None                                # when it was sent (MIT speed limit)
        self.t_move = None                                # when the target last changed (MIT: HOLD_KD)
        self.ff = 1.0                                     # gravity feed-forward scale (MIT)
        self.torque_orig: dict[int, float] = {}           # limit_torque before the run (MIT), restored on release
        self.gravity = Gravity() if control == "mit" else None

    def read(self) -> np.ndarray:
        q = []
        for m in ARM_IDS:
            v = None
            for _ in range(3):
                v = self.rs.read_param_f32(m, P_MECHPOS, timeout=0.02)
                if v is not None:
                    break
            if v is None:
                raise ConnectionError(f"joint {m} is not answering")
            q.append(v)
        self.turns = np.array(q) - unwind(q)
        return self.cfg.to_model(q)

    def _motor(self, q) -> np.ndarray:
        """Motor targets of a model pose, in the turn each motor counts."""
        return self.cfg.to_motor(q) + self.turns

    def enable(self) -> np.ndarray:
        """Position mode, speed limit, current cap and target = where each joint is now, then power
        on (MIT: see enable_mit). Returns the (model) pose it holds. Runs whole: a Ctrl+C meanwhile
        comes right after."""
        if self.control == "mit":
            return self.enable_mit()
        with _ctrl_c_after():
            q = self.read()
            for m, qm in zip(ARM_IDS, self._motor(q)):
                self.rs.stop(m)
                self.rs.write_param_u8(m, P_RUN_MODE, RUN_MODE_POSITION)
                time.sleep(0.01)
                if self.rs.read_param_u8(m, P_RUN_MODE) != RUN_MODE_POSITION:
                    raise RuntimeError(f"joint {m} did not switch to position mode")
                self.rs.write_param_f32(m, P_LIMIT_SPD, self.speed)
                self.rs.write_param_f32(m, P_LIMIT_CUR, self.caps[m])
                self.rs.write_param_f32(m, P_LOC_REF, float(qm))
            time.sleep(0.01)
            self.last = q
            self.enabled = True
            for m in (6, 5, 4, 1, 3, 2):                 # the joints carrying the arm last
                self.rs.enable(m)
        return q

    def enable_mit(self) -> np.ndarray:
        """MIT mode, torque cap, a hold at where each joint is now (gravity feed-forward off), power on,
        then the feed-forward ramps in over FF_RAMP. Runs whole."""
        with _ctrl_c_after():
            q = self.read()
            for i, m in enumerate(ARM_IDS):
                self.rs.stop(m)
                self.rs.write_param_u8(m, P_RUN_MODE, RUN_MODE_MIT)
                time.sleep(0.01)
                if self.rs.read_param_u8(m, P_RUN_MODE) != RUN_MODE_MIT:
                    raise RuntimeError(f"joint {m} did not switch to MIT mode")
                lt = self.rs.read_param_f32(m, P_LIMIT_TORQUE)
                if lt is None or not 0.3 * MOTOR_TORQUE_MAX[i] <= lt <= 1.01 * MOTOR_TORQUE_MAX[i]:
                    raise RuntimeError(f"joint {m}: limit_torque reads {lt}, not a plausible torque limit")
                self.torque_orig[m] = lt
                self.rs.write_param_f32(m, P_LIMIT_TORQUE, float(TORQUE_CAP[i]))
            self.last, self.t_last, self.ff, self.t_move = q, None, 0.0, time.monotonic()
            self._send_mit(q)                            # the hold target goes first
            self.enabled = True
            for m in (6, 5, 4, 1, 3, 2):                 # the joints carrying the arm last
                self.rs.enable(m)
            t0 = time.monotonic()
            while (s := (time.monotonic() - t0) / FF_RAMP) < 1.0:
                self.ff = 0.5 - 0.5 * np.cos(np.pi * s)
                self._send_mit(q)
                time.sleep(1.0 / RATE)
            self.ff = 1.0
            self._send_mit(q)
        return q

    def _send_mit(self, q, kd_scale: float = 1.0) -> None:
        tau = self.ff * self.gravity(q)
        for i, (m, v) in enumerate(zip(ARM_IDS, self._motor(q))):
            self.rs.mit(m, float(v), float(MIT_KP[i]), float(kd_scale * MIT_KD[i]), 0.0, float(tau[i]))
        self._flush()

    def _flush(self) -> None:
        """Take the motors' replies off the bus (MIT: one feedback frame per command)."""
        bus = getattr(self.rs, "bus", None)
        if bus is None:
            return
        while (msg := bus.recv(0.0)) is not None:
            self.rs._handle(msg)

    def measured(self) -> np.ndarray:
        """Model angles from the motors' latest MIT feedback (NaN where none came yet)."""
        fb = getattr(self.rs, "last_fb", {})
        q = np.array([fb[m].pos if m in fb else np.nan for m in ARM_IDS])
        return self.cfg.to_model(q)

    def command(self, q) -> None:
        q = self.cfg.within(q, self.last)
        if self.control == "mit":
            now = time.monotonic()
            if self.last is not None:
                # MIT mode has no firmware speed limit: this is the backstop. The whole step is scaled
                # (never joint by joint), so a limited target stays on the straight joint-space line to
                # the wanted one. Callers keep their own motion below `speed` (the planner does).
                dt = 1.0 / RATE if self.t_last is None else min(now - self.t_last, 2.0 / RATE)
                step = q - self.last
                worst = float(np.abs(step).max())
                if worst > self.speed * dt:
                    step *= self.speed * dt / worst
                q = self.last + step
            if self.last is None or self.t_move is None or float(np.abs(q - self.last).max()) > 1e-6:
                self.t_move = now
            self.last, self.t_last = q, now
            self._send_mit(q, HOLD_KD if now - self.t_move > HOLD_AFTER else 1.0)
            return
        self.last = q
        for m, v in zip(ARM_IDS, self._motor(q)):
            self.rs.write_param_f32(m, P_LOC_REF, float(v))

    def hold(self) -> np.ndarray:
        """Hold where the arm is now."""
        q = self.read()
        self.last, self.t_last = q, None
        self.command(q)
        return q

    def follow(self, path) -> None:
        """Stream the joint targets of `path` (model angles), one every 1 / RATE s."""
        t = time.monotonic()
        for q in path:
            self.command(q)
            t += 1.0 / RATE
            time.sleep(max(0.0, t - time.monotonic()))

    def go(self, q, speed: float | None = None, settle: float = 0.3) -> np.ndarray:
        """Straight joint-space move from where the arm is to `q`: eased, the fastest joint peaking at
        `speed` (default MOVE_SPEED), streamed at RATE. Returns the pose reached after `settle` seconds."""
        q0 = self.read()
        q1 = self.cfg.within(q, q0)
        self.last, self.t_last = q0, None
        n = max(int(np.ceil(np.abs(q1 - q0).max() * 1.5 / (speed or MOVE_SPEED) * RATE)), 1)   # smoothstep: 1.5x the mean
        self.follow(q0 + ease(n)[:, None] * (q1 - q0))
        time.sleep(settle)
        return self.read()

    def open_gripper(self, target: float = GRIP_OPEN) -> str | None:
        """Gripper (motor 7) to `target` motor rad in position mode (speed limit, current cap), held
        until release(). Returns a warning instead when it cannot: then open it by hand."""
        g = None
        for _ in range(3):
            g = self.rs.read_param_f32(GRIPPER_ID, P_MECHPOS, timeout=0.02)
            if g is not None:
                break
        if g is None:
            return "the gripper (motor 7) does not answer: open it by hand"
        if not -0.3 <= g <= TARGET_MAX + 0.3:
            return f"the gripper reads {g:.2f} rad, outside its 0..5 rad travel (zero not set?): open it by hand"
        target = float(np.clip(target, TARGET_MIN, TARGET_MAX))
        with _ctrl_c_after():
            self.rs.stop(GRIPPER_ID)
            self.rs.write_param_u8(GRIPPER_ID, P_RUN_MODE, RUN_MODE_POSITION)
            time.sleep(0.01)
            if self.rs.read_param_u8(GRIPPER_ID, P_RUN_MODE) != RUN_MODE_POSITION:
                return "the gripper did not switch to position mode: open it by hand"
            self.rs.write_param_f32(GRIPPER_ID, P_LIMIT_SPD, GRIP_SPEED)
            self.rs.write_param_f32(GRIPPER_ID, P_LIMIT_CUR, GRIP_CUR)
            self.rs.write_param_f32(GRIPPER_ID, P_LOC_REF, float(g))
            self.grip_on = True
            self.rs.enable(GRIPPER_ID)
            self.rs.write_param_f32(GRIPPER_ID, P_LOC_REF, target)
        time.sleep(abs(target - g) / GRIP_SPEED + 0.3)
        return None

    def release(self) -> None:
        """Stop every joint, and the gripper if it was powered (three rounds). Runs whole: a Ctrl+C
        meanwhile comes right after."""
        with _ctrl_c_after():
            for _ in range(3):
                for m in ARM_IDS + ((GRIPPER_ID,) if self.grip_on else ()):
                    self.rs.stop(m)
                time.sleep(0.01)
            self.enabled, self.grip_on = False, False
            for m, lt in self.torque_orig.items():      # MIT: the motors' own torque limit back
                self.rs.write_param_f32(m, P_LIMIT_TORQUE, float(lt))
            self.torque_orig.clear()

    def close(self) -> None:
        try:
            if self.enabled:
                self.release()
        finally:
            self.rs.close()


@contextmanager
def powered(cfg: ArmConfig | None = None, rs=None, speed: float = SPEED, ask=None, grip: float | None = None,
            control: str = "position"):
    """ENTER -> power on -> initial pose -> the caller's work -> initial pose -> rest pose -> release.
    `control`: "position" (the motors' position mode) or "mit" (see MIT_KP / Arm).
    Without recorded poses it starts and ends where the arm was resting. `grip` (motor rad): at the
    initial pose the gripper opens to it and stays powered until the release (None: not touched).

    Ctrl+C (or an error) holds the arm where it is. Then ENTER takes it back to rest (through the
    initial pose) and releases it there, and the caller goes on after its `with` (saves its log); if
    the arm cannot move (lost bus), ENTER releases it where it is: support it. A second Ctrl+C, at the
    prompt or on the way back, releases it at once and stops the program (KeyboardInterrupt). Ctrl+C
    alone always stops a powered task. A Ctrl+C before the work started is raised again (the work
    never runs)."""
    cfg, ask = cfg or load(), ask or input
    arm = Arm(cfg, rs, speed, gripper=grip is not None, control=control)
    working, rest = False, None
    try:
        ask("Arm clear? ENTER powers it (Ctrl+C holds it, a second Ctrl+C drops it)... ")
        start = arm.enable()
        rest = start if cfg.rest is None else cfg.rest_model()
        if cfg.initial is not None:
            arm.go(cfg.initial_model())
        if grip is not None:
            warn = arm.open_gripper(grip)
            print(f"CHECK: {warn}" if warn else f"gripper open ({grip:.1f} rad)", flush=True)
        working = True
        yield arm
        if cfg.initial is not None:
            arm.go(cfg.initial_model())
        arm.go(rest)
    except (Exception, KeyboardInterrupt) as e:
        forced = False
        if arm.enabled:
            why = "Ctrl+C" if isinstance(e, KeyboardInterrupt) else f"{type(e).__name__}: {e}"
            try:
                with _ctrl_c_after():                    # a second Ctrl+C right away still lets the hold finish
                    try:
                        arm.hold()
                    except Exception:  # noqa: BLE001  (lost the bus: nothing to hold with)
                        pass
                if rest is None:                         # failed while powering on: no pose to go back to
                    ask(f"\n{why} -> holding. Support the arm, then ENTER releases it (Ctrl+C: drop it now)... ")
                else:
                    ask(f"\n{why} -> holding. ENTER takes it back to rest and releases it (Ctrl+C: drop it now)... ")
                    try:
                        if cfg.initial is not None:
                            arm.go(cfg.initial_model())
                        arm.go(rest)
                    except Exception as e2:  # noqa: BLE001  (e.g. lost the bus)
                        with _ctrl_c_after():
                            try:
                                arm.hold()
                            except Exception:  # noqa: BLE001
                                pass
                        ask(f"could not go back to rest ({type(e2).__name__}: {e2}). Support the arm, then ENTER "
                            "releases it (Ctrl+C: drop it now)... ")
            except (KeyboardInterrupt, EOFError):        # the second Ctrl+C (or stdin closed): release now, stop
                forced = True
                print("\nreleased", flush=True)
            arm.release()
        if forced and isinstance(e, KeyboardInterrupt):
            raise KeyboardInterrupt from None
        if not (working and isinstance(e, KeyboardInterrupt)):
            raise
    finally:
        arm.close()
