"""eaiproject/arm.py without hardware: the arm config file, the wrist zero from a level reading, and the
position-mode arm on a fake CAN bus (enable, moves, the powered route, Ctrl+C).

    .venv/bin/python -m unittest tests.test_arm -v
"""
from __future__ import annotations

import io
import os
import signal
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eaiproject import arm  # noqa: E402
from eaiproject.ball_intercept_env import BallInterceptEnv  # noqa: E402
from eaiproject.robstride import ARM_IDS, GRIPPER_ID, P_MECHPOS, P_RUN_MODE  # noqa: E402

REST = np.array([0.0946, -0.0029, -0.0001, -0.0003, -0.0537, 2.8035])     # the real arm's folded rest pose (motor)
INIT = np.array([0.0, 0.9, 0.9, 0.3, 0.0, 0.0])                            # raised, clear of the table (motor)
WORK = np.array([0.1, 1.2, 1.0, -0.4, 0.1, 0.2])                           # model
ROLL15 = np.radians([0, 0, 0, 0, 0, 15.0])                                 # the motor's zero is off by this


def assert_same_pose(got, want, atol=1e-9):
    """The same arm pose: joint 1 may differ by whole turns and joint 6 by half turns (arm.SAME_EVERY),
    since the arm goes to the equivalent angle nearest to where it is."""
    np.testing.assert_allclose(got, arm.nearest(want, got), atol=atol)


def level_pose(pitch: float, yaw: float, x=0.40, y=0.05) -> np.ndarray:
    """Model joint angles with the jaw level (fingers pitched down by `pitch`)."""
    env = BallInterceptEnv("forward" if pitch < 1.2 else "down")
    env.pitch = pitch
    seed = env.q_home if pitch < 1.2 else np.array([0, 1.8, 1.16, -0.93, 0, 0])
    q = env.ik_jaw(x, y, 0.05, yaw=yaw, q0=seed, iters=400)
    assert env.jaw_error(q, x, y, 0.05) < 2e-3
    return q


class FakeBus:
    """Six arm motors (position or MIT mode) and the gripper, reaching their target at once while enabled."""

    def __init__(self, q_motor, run_mode_sticks: bool = True, gripper: float = 0.3):
        self.pos = dict(zip(ARM_IDS, map(float, q_motor))) | {GRIPPER_ID: gripper}
        self.params = {m: {arm.P_LIMIT_TORQUE: 36.0 if m <= 3 else 14.0} for m in ARM_IDS} | {GRIPPER_ID: {}}
        self.gains: dict[int, tuple] = {}             # MIT: the last (kp, kd) per motor
        self.on: set[int] = set()
        self.enable_order: list[int] = []
        self.targets: list[np.ndarray] = []           # the full motor target after every command
        self.sticks, self.closed = run_mode_sticks, False

    def read_param_f32(self, m, idx, timeout=0.05):
        return self.pos[m] if idx == P_MECHPOS else self.params[m].get(idx)

    def read_param_u8(self, m, idx, timeout=0.05):
        return self.params[m].get(idx)

    def write_param_u8(self, m, idx, v):
        if idx != P_RUN_MODE or self.sticks:
            self.params[m][idx] = v

    def write_param_f32(self, m, idx, v):
        self.params[m][idx] = v
        if idx == arm.P_LOC_REF:
            if m in self.on:
                self.pos[m] = v
            if m == ARM_IDS[-1]:
                self.targets.append(np.array([self.params[k][arm.P_LOC_REF] for k in ARM_IDS]))

    def mit(self, m, pos, kp, kd, vel=0.0, tau=0.0):
        self.gains[m] = (kp, kd)
        self.write_param_f32(m, arm.P_LOC_REF, pos)   # a target, as in position mode

    def enable(self, m):
        self.on.add(m)
        self.enable_order.append(m)

    def stop(self, m):
        self.on.discard(m)

    def close(self):
        self.closed = True

    def q(self) -> np.ndarray:
        return np.array([self.pos[m] for m in ARM_IDS])

    def visits(self, *poses) -> bool:
        """The targets pass through `poses` (motor angles) in this order."""
        i = 0
        for t in self.targets:
            if i < len(poses) and np.allclose(t, arm.nearest(poses[i], t), atol=1e-9):     # the same pose (see assert_same_pose)
                i += 1
        return i == len(poses)


class SimTime:
    """Stands in for the `time` module inside eaiproject/arm.py: sleeping advances a simulated clock."""

    def __init__(self):
        self.t = 1000.0
        self.strftime = time.strftime

    def monotonic(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += max(float(dt), 0.0)


def no_sleep():
    return mock.patch.object(arm, "time", SimTime())


class ConfigTest(unittest.TestCase):
    def test_roundtrip_and_conversions(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "arm.json"
            self.assertIsNone(arm.load(p).rest)                                 # no file: no correction, no poses
            np.testing.assert_array_equal(arm.load(p).offset, np.zeros(6))
            cfg = arm.ArmConfig(np.radians([0, 0, 0, 0, 0, 14.0]), REST.copy(), INIT.copy())
            cfg.save(p)
            back = arm.load(p)
            np.testing.assert_allclose(back.offset, cfg.offset, atol=1e-6)
            np.testing.assert_allclose(back.rest, REST, atol=1e-5)
            np.testing.assert_allclose(back.to_motor(back.to_model(INIT)), INIT)
            self.assertAlmostEqual(back.rest_model()[5] - REST[5], np.radians(14.0), places=6)


class LevelTest(unittest.TestCase):
    def test_level_reading_gives_the_wrist_zero(self):
        for pitch, yaw in ((0.5, 0.0), (0.8, 0.4), (1.0, -0.3)):
            q_motor = level_pose(pitch, yaw) - ROLL15                           # what the motor reads
            d, sens = arm.level_offset(q_motor)                                 # no correction yet
            self.assertAlmostEqual(np.degrees(d), 15.0, delta=0.05, msg=f"pitch {pitch}")
            self.assertGreater(sens, 0.5)
            cur = np.radians(5.0)                                               # with an earlier, partial correction
            q_model = q_motor.copy()
            q_model[5] += cur
            self.assertAlmostEqual(np.degrees(cur + arm.level_offset(q_model)[0]), 15.0, delta=0.05)

    def test_on_a_tilted_table(self):
        q = level_pose(0.6, 0.0)                                                # fingers open along y
        along_x = arm.level_offset(q, np.array([0.005, np.tan(np.radians(2.0)), 0.0]))[0]
        along_y = arm.level_offset(q, np.array([0.005, 0.0, np.tan(np.radians(2.0))]))[0]
        self.assertAlmostEqual(np.degrees(along_x), 0.0, delta=0.01)            # a slope along the fingers: still level
        self.assertGreater(abs(np.degrees(along_y)), 1.5)                       # sideways: level to the table, not to gravity

    def test_fingers_straight_down_cannot_tell(self):
        _, sens = arm.level_offset(level_pose(np.pi / 2, 0.0, x=0.32))
        self.assertLess(sens, 0.1)


class ArmLinkTest(unittest.TestCase):
    def test_connects_when_the_arm_answers(self):
        reader = mock.Mock(offset=np.zeros(6))
        reader.latest.return_value = "sample"
        tries = iter([OSError("adapter unplugged"), ConnectionError("arm silent")])

        def connect():
            e = next(tries, None)
            if e is not None:
                raise e
            return reader

        with mock.patch.object(arm.ArmLink, "RETRY_S", 0.02):
            link = arm.ArmLink(connect)
            self.assertIn("no CAN adapter", link.status())                      # first try, at once
            self.assertIsNone(link.latest())
            self.assertIsNone(link.steady())
            self.assertEqual(link.between(0.0, 1.0), [])
            link.offset = np.full(6, 0.1)                                       # set before it connects: handed over
            t_end = time.monotonic() + 2.0
            while link.status() and time.monotonic() < t_end:
                time.sleep(0.01)
        self.assertEqual(link.status(), "")
        self.assertEqual(link.latest(), "sample")
        np.testing.assert_allclose(reader.offset, 0.1)
        link.close()
        reader.close.assert_called_once()


class ArmTest(unittest.TestCase):
    def test_enable_holds_where_the_arm_is(self):
        bus = FakeBus(REST)
        a = arm.Arm(arm.ArmConfig(ROLL15), bus)
        with no_sleep():
            q = a.enable()
        np.testing.assert_allclose(q, REST + ROLL15)                            # model angles
        for i, m in enumerate(ARM_IDS):
            p = bus.params[m]
            self.assertEqual(p[P_RUN_MODE], arm.RUN_MODE_POSITION)
            self.assertEqual(p[arm.P_LIMIT_SPD], arm.SPEED)
            self.assertEqual(p[arm.P_LIMIT_CUR], arm.CUR_CAP[m])
            self.assertAlmostEqual(p[arm.P_LOC_REF], REST[i])                    # motor angle: the offset is taken off
        self.assertEqual(bus.on, set(ARM_IDS))
        self.assertEqual(bus.enable_order[-2:], [3, 2])
        np.testing.assert_allclose(bus.q(), REST)                               # nothing moved

    def test_refuses_when_a_motor_stays_out_of_position_mode(self):
        bus = FakeBus(REST, run_mode_sticks=False)
        with no_sleep(), self.assertRaisesRegex(RuntimeError, "position mode"):
            arm.Arm(arm.ArmConfig(), bus).enable()
        self.assertEqual(bus.on, set())

    def test_go_is_an_eased_straight_line(self):
        bus = FakeBus(REST)
        a = arm.Arm(arm.ArmConfig(ROLL15), bus)
        with no_sleep():
            a.enable()
            n0 = len(bus.targets)
            reached = a.go(WORK)
        work = arm.nearest(WORK, REST + ROLL15)                                 # the equivalent pose nearest the arm
        np.testing.assert_allclose(reached, work)
        T = np.array(bus.targets[n0:])
        np.testing.assert_allclose(T[-1], work - ROLL15)
        d = (work - ROLL15) - REST
        s = (T - REST) @ d / (d @ d)
        np.testing.assert_allclose(T, REST + s[:, None] * d, atol=1e-12)      # on the line
        self.assertTrue(np.all(np.diff(s) > 0))                                 # never back
        step = np.abs(np.diff(np.vstack([REST, T]), axis=0)).max(axis=1)
        self.assertLessEqual(step.max(), 1.5 * arm.MOVE_SPEED / arm.RATE + 1e-9)
        self.assertLess(step[0], 0.2 * step.max())                              # starts and ends slowly
        self.assertLess(step[-1], 0.2 * step.max())


class PoweredTest(unittest.TestCase):
    CFG = arm.ArmConfig(ROLL15, REST.copy(), INIT.copy())

    def test_route_rest_initial_work_initial_rest(self):
        bus, asks = FakeBus(REST), []
        with no_sleep(), arm.powered(self.CFG, bus, ask=asks.append) as a:
            a.go(WORK)
        self.assertEqual(len(asks), 1)                                          # ENTER before powering, nothing else
        self.assertTrue(bus.visits(INIT, WORK - ROLL15, INIT, REST))
        assert_same_pose(bus.q(), REST)
        self.assertEqual(bus.on, set())
        self.assertTrue(bus.closed)

    def test_long_way_round_is_never_taken(self):
        """A target a whole turn away at joint 1 (the rest pose reads 267 deg, the plan says -90) or half a
        turn at joint 6 is the pose the arm is in: it does not move."""
        rest = np.radians([267.2, 0.4, 0.0, -0.1, 1.1, 211.5])
        bus = FakeBus(rest)
        a = arm.Arm(arm.ArmConfig(), bus)
        with no_sleep():
            a.enable()
            a.go(rest - np.radians([360.0, 0, 0, 0, 0, 180.0]))
        T = np.array(bus.targets)
        self.assertLess(np.abs(T - rest).max(), 1e-9)

    def test_without_poses_it_ends_where_it_started(self):
        start = INIT + 0.05
        bus = FakeBus(start)
        with no_sleep(), arm.powered(arm.ArmConfig(), bus, ask=lambda s: "") as a:
            a.go(WORK)
        self.assertTrue(bus.visits(WORK, start))
        assert_same_pose(bus.q(), start)
        self.assertEqual(bus.on, set())

    def test_ctrl_c_holds_then_enter_takes_it_back_to_rest(self):
        bus, asks = FakeBus(REST), []
        powered_at_ask, pose_at_ask = [], []
        ask = lambda s: (asks.append(s), powered_at_ask.append(len(bus.on)), pose_at_ask.append(bus.q()))
        with no_sleep(), arm.powered(self.CFG, bus, ask=ask) as a:
            a.go(WORK)
            raise KeyboardInterrupt
        self.assertEqual(len(asks), 2)
        self.assertIn("back to rest", asks[-1])
        self.assertEqual(powered_at_ask[-1], 6)                                 # still holding while asking
        assert_same_pose(pose_at_ask[-1], WORK - ROLL15)                        # held where it was
        self.assertTrue(bus.visits(INIT, WORK - ROLL15, INIT, REST))            # then home by itself
        assert_same_pose(bus.q(), REST)
        self.assertEqual(bus.on, set())

    def test_when_it_cannot_go_back_enter_releases_it_in_place(self):
        bus, asks = FakeBus(REST), []

        def ask(s):
            asks.append(s)
            if "back to rest" in s:
                bus.read_param_f32 = lambda *a, **k: None                       # the bus is lost before the way back

        with no_sleep(), arm.powered(self.CFG, bus, ask=ask) as a:
            a.go(WORK)
            raise KeyboardInterrupt
        self.assertEqual(len(asks), 3)
        self.assertIn("could not go back to rest", asks[-1])
        self.assertIn("Support the arm", asks[-1])
        assert_same_pose(bus.q(), WORK - ROLL15)                                # not moved
        self.assertEqual(bus.on, set())

    def test_an_error_holds_releases_and_is_raised(self):
        bus, asks = FakeBus(REST), []
        with no_sleep(), self.assertRaisesRegex(ValueError, "planner"):
            with arm.powered(self.CFG, bus, ask=asks.append):
                raise ValueError("planner failed")
        self.assertIn("ValueError: planner failed", asks[-1])
        self.assertEqual(bus.on, set())

    def test_second_ctrl_c_drops_the_arm_and_stops(self):
        """Ctrl+C holds; a second one at the hold prompt releases at once and stops the program."""
        bus, held_at_ask = FakeBus(REST), []

        def ask(s):
            if "holding" in s:
                held_at_ask.append(len(bus.on))
                raise KeyboardInterrupt                                         # Ctrl+C again

        with no_sleep(), redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            with arm.powered(self.CFG, bus, ask=ask) as a:
                a.go(WORK)
                raise KeyboardInterrupt
        self.assertEqual(held_at_ask, [6])                                      # held until the second press
        assert_same_pose(bus.targets[-1], WORK - ROLL15)                        # dropped there, not taken home
        self.assertEqual(bus.on, set())
        self.assertTrue(bus.closed)

    def test_two_quick_ctrl_c_finish_the_hold_then_drop(self):
        """The second press lands while the hold is being sent (a real SIGINT): the hold completes,
        then the arm is released at once, without the prompt."""
        bus, asks, armed = FakeBus(REST), [], []
        read, before = bus.read_param_f32, signal.getsignal(signal.SIGINT)

        def read_with_ctrl_c(m, idx, timeout=0.05):
            if armed:
                armed.clear()
                os.kill(os.getpid(), signal.SIGINT)
            return read(m, idx, timeout)

        bus.read_param_f32 = read_with_ctrl_c
        with no_sleep(), redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            with arm.powered(self.CFG, bus, ask=asks.append) as a:
                a.go(WORK)
                armed.append(1)
                raise KeyboardInterrupt
        self.assertFalse(any("holding" in s for s in asks))
        assert_same_pose(bus.targets[-1], WORK - ROLL15)                        # the hold was sent whole
        self.assertEqual(bus.on, set())
        self.assertIs(signal.getsignal(signal.SIGINT), before)

    def test_ctrl_c_while_enabling_finishes_it_then_holds(self):
        bus, asks = FakeBus(REST), []
        enable = bus.enable

        def enable_with_ctrl_c(m):
            if not bus.enable_order:
                os.kill(os.getpid(), signal.SIGINT)                             # pressed at the first enable
            enable(m)

        bus.enable = enable_with_ctrl_c
        with no_sleep(), self.assertRaises(KeyboardInterrupt):
            with arm.powered(self.CFG, bus, ask=asks.append):
                self.fail("must not run")
        self.assertEqual(sorted(bus.enable_order), list(ARM_IDS))               # not cut halfway
        self.assertIn("Ctrl+C -> holding", asks[-1])
        self.assertEqual(bus.on, set())

    def test_ctrl_c_at_the_first_prompt_powers_nothing(self):
        bus = FakeBus(REST)

        def ask(s):
            raise KeyboardInterrupt

        with no_sleep(), self.assertRaises(KeyboardInterrupt):
            with arm.powered(self.CFG, bus, ask=ask):
                self.fail("must not run")
        self.assertEqual(bus.enable_order, [])
        self.assertTrue(bus.closed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
