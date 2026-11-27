"""The planner's ball prediction and speed limiting, and the arm's backstop limiter (no hardware).

    .venv/bin/python -m unittest tests.test_catch_planner
"""
from __future__ import annotations

import sys
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eaiproject import arm as arm_mod  # noqa: E402
from eaiproject import rig_calib as rc  # noqa: E402
from eaiproject.kinematics import Fingertip  # noqa: E402
from eaiproject.real_intercept import BALL_DECEL, InterceptPlanner, crossing  # noqa: E402

FPS = 60.0
PLANE = np.zeros(3)
FRAME = rc.TableFrame(-90.0)       # as on the real rig: the base turned 90 deg to the right


class CrossingTest(unittest.TestCase):
    def test_point_is_the_straight_line_and_time_counts_the_slowdown(self):
        pos, vel = np.array([0.95, -0.10]), np.array([-0.6, 0.08])
        y, t = crossing(pos, vel, 0.40)
        y0, t0 = crossing(pos, vel, 0.40, decel=0.0)
        self.assertAlmostEqual(y, -0.10 + 0.08 * 0.55 / 0.6, places=9)
        self.assertEqual(y, y0)                                # the slowdown does not move the crossing point
        self.assertAlmostEqual(t0, 0.55 / 0.6, places=9)
        # check the time by stepping the roll: the speed falls by BALL_DECEL per second along the path
        e, p, v, tt, h = vel / np.linalg.norm(vel), pos.copy(), float(np.linalg.norm(vel)), 0.0, 1e-4
        while p[0] > 0.40:
            p, v, tt = p + e * v * h, v - BALL_DECEL * h, tt + h
        self.assertAlmostEqual(t, tt, places=2)
        self.assertGreater(t, t0 + 0.05)                       # ~90 ms later than a constant speed says

    def test_a_ball_too_slow_to_arrive_has_no_time_but_keeps_its_point(self):
        y, t = crossing([0.95, 0.05], [-0.25, 0.0], 0.40)      # 0.25^2 / (2 * 0.2) = 0.16 m of travel
        self.assertIsNone(t)
        self.assertAlmostEqual(y, 0.05)

    def test_a_ball_at_or_past_the_line(self):
        self.assertEqual(crossing([0.40, 0.1], [-0.5, 0.0], 0.40), (0.1, 0.0))
        self.assertEqual(crossing([0.38, 0.1], [-0.5, 0.0], 0.40)[1], 0.0)


class Tracker:
    """What the planner reads from a BallTracker: `ready` and state_at(now) -> pos, vel (ARM frame).
    Here: a ball rolling from p0 with v0 (table axes) from t = 0, slowing down; none before / after."""

    def __init__(self, p0=None, v0=None, t0: float = 0.0):
        self.p0, self.t0 = (None if p0 is None else np.array(p0, float)), t0
        self.v0 = None if v0 is None else np.array(v0, float)
        self.ready = False

    def state_at(self, now: float):
        tau = now - self.t0
        speed = float(np.linalg.norm(self.v0))
        e = self.v0 / speed if speed > 0 else np.zeros(2)
        tau = min(tau, speed / BALL_DECEL) if speed > 0 else tau
        p = self.p0 + self.v0 * tau - 0.5 * BALL_DECEL * e * tau**2
        v = self.v0 - BALL_DECEL * e * tau
        return SimpleNamespace(pos=FRAME.arm_xy([p])[0], vel=FRAME.arm_xy([v])[0])

    def at(self, now: float) -> "Tracker":
        self.ready = self.p0 is not None and now >= self.t0
        return self


class PlannerSpeedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tip = Fingertip()

    def height(self, q) -> float:
        p = self.tip(q)
        return float(p[2] - rc.plane_z(PLANE, p[0], p[1]))

    def run_roll(self, planner, trk: Tracker, T: float):
        out, t = [], trk.t0 - 2 / FPS
        while t < trk.t0 + T:
            out.append((t, planner.plan(trk.at(t), t)))
            t += 1 / FPS
        return out

    def test_fast_roll_keeps_the_joint_speed_and_the_height(self):
        for limit in (1.5, 3.0):
            pl = InterceptPlanner(PLANE, FRAME, max_joint_speed=limit)
            plans = self.run_roll(pl, Tracker([0.95, 0.0], [-1.0, 0.33]), 1.0)     # fast, far to the side: more than the arm can do
            q = np.array([p.q for _, p in plans])
            speed = np.abs(np.diff(q, axis=0)).max() * FPS
            self.assertLessEqual(speed, limit * 1.03, f"limit {limit}")
            self.assertGreater(speed, limit * 0.8, f"limit {limit}: the roll should push the planner to its limit")
            h = np.array([self.height(x) for x in q])
            # every target is a point on the line at the planned height: catch height .. catch height + hover
            self.assertGreater(h.min(), pl.jaw_height - 0.002, f"limit {limit}")
            self.assertLess(h.max(), pl.jaw_height + pl.hover + 0.002, f"limit {limit}")
            self.assertTrue(all(p.ik_err < 0.002 for _, p in plans))

    def test_reachable_roll_is_met_at_the_line(self):
        pl = InterceptPlanner(PLANE, FRAME)
        trk = Tracker([0.95, -0.05], [-0.6, 0.15])
        y_true, t_true = crossing(trk.p0, trk.v0, pl.gate_x)
        plans = self.run_roll(pl, trk, 1.4)
        at_line = min(plans, key=lambda tp: abs(tp[0] - t_true))[1]
        self.assertLess(abs(at_line.jaw_y - y_true), 0.005)
        self.assertIn("arrived", [p.phase for _, p in plans])
        early = next(p for t, p in plans if t >= 0.1)                    # the time it announces is right from the start
        self.assertLess(abs((0.1 + early.t_cross) - t_true), 0.03)
        self.assertLess(abs(self.height(at_line.q) - pl.jaw_height), 0.002)   # lowered to the catch height

    def test_no_ball_or_a_receding_ball_does_not_move_the_jaw(self):
        pl = InterceptPlanner(PLANE, FRAME)
        for trk in (Tracker(), Tracker([0.5, 0.2], [0.7, 0.0])):
            trk.t0 = 0.0
            plans = self.run_roll(pl, trk, 0.6)
            self.assertTrue(all(p.phase == "home" for _, p in plans))
            self.assertLess(max(abs(p.jaw_y) for _, p in plans), 1e-9)

    def test_reach_is_30_cm(self):
        pl = InterceptPlanner(PLANE, FRAME)
        plans = self.run_roll(pl, Tracker([0.95, 0.45], [-0.5, 0.0]), 1.4)      # crosses beyond the reach
        self.assertAlmostEqual(max(p.jaw_y for _, p in plans), 0.30, places=3)
        self.assertTrue(all(p.ik_err < 0.002 for _, p in plans))

    def test_recentre_goes_back_at_the_planners_speed(self):
        pl = InterceptPlanner(PLANE, FRAME, max_joint_speed=2.0)
        plans = self.run_roll(pl, Tracker([0.95, 0.2], [-0.7, 0.0]), 1.0)
        self.assertGreater(plans[-1][1].jaw_y, 0.15)
        pl.recentre()
        trk, t, q = Tracker(), plans[-1][0], [plans[-1][1].q]
        for _ in range(int(1.5 * FPS)):
            t += 1 / FPS
            q.append(pl.plan(trk, t).q)
        self.assertLess(abs(pl.y_cmd), 1e-3)
        self.assertLessEqual(np.abs(np.diff(np.array(q), axis=0)).max() * FPS, 2.0 * 1.03)


class Bus:
    """Six MIT motors that take every target at once."""

    def __init__(self, q):
        self.pos = dict(zip(arm_mod.ARM_IDS, map(float, q)))
        self.sent = []

    def read_param_f32(self, m, idx, timeout=0.05):
        return self.pos[m]

    def mit(self, m, pos, kp, kd, vel=0.0, tau=0.0):
        self.pos[m] = pos
        self.kd = getattr(self, "kd", {}) | {m: kd}
        if m == arm_mod.ARM_IDS[-1]:
            self.sent.append(np.array([self.pos[k] for k in arm_mod.ARM_IDS]))


class BackstopTest(unittest.TestCase):
    def test_a_jump_is_cut_along_the_straight_line_to_the_target(self):
        q0 = np.radians([-90.0, 87.0, 16.0, 42.0, 0.0, 0.0])
        a = arm_mod.Arm(arm_mod.ArmConfig(), Bus(q0), speed=2.0, control="mit")
        a.last = q0.copy()
        target = q0 + np.radians([40.0, 10.0, -20.0, 5.0, 0.0, 0.0])
        a.command(target)
        step = a.last - q0
        self.assertAlmostEqual(float(np.abs(step).max()), 2.0 / arm_mod.RATE, places=6)     # the limit, on the fastest joint
        np.testing.assert_allclose(step / np.abs(step).max(), (target - q0) / np.abs(target - q0).max(), atol=1e-9)

    def test_a_step_below_the_limit_is_sent_as_it_is(self):
        q0 = np.radians([-90.0, 87.0, 16.0, 42.0, 0.0, 0.0])
        a = arm_mod.Arm(arm_mod.ArmConfig(), Bus(q0), speed=2.0, control="mit")
        a.last = q0.copy()
        target = q0 + np.radians([1.0, -0.5, 0.2, 0.0, 0.0, 0.0])
        a.command(target)
        np.testing.assert_allclose(a.last, target, atol=1e-12)


class HoldTest(unittest.TestCase):
    def test_damping_is_lower_while_the_target_stands_and_full_when_it_moves(self):
        q0 = np.radians([-90.0, 87.0, 16.0, 42.0, 0.0, 0.0])
        bus, clock = Bus(q0), SimpleNamespace(t=100.0)
        a = arm_mod.Arm(arm_mod.ArmConfig(), bus, speed=2.0, control="mit")
        a.last, a.t_move = q0.copy(), clock.t
        kd = lambda: np.array([bus.kd[m] for m in arm_mod.ARM_IDS])
        with unittest.mock.patch.object(arm_mod.time, "monotonic", lambda: clock.t):
            for _ in range(10):                                  # 0.2 s of the same target: still the full gain
                clock.t += 0.02
                a.command(q0)
            np.testing.assert_allclose(kd(), arm_mod.MIT_KD)
            for _ in range(10):                                  # 0.4 s: holding
                clock.t += 0.02
                a.command(q0)
            np.testing.assert_allclose(kd(), arm_mod.HOLD_KD * arm_mod.MIT_KD)
            clock.t += 0.02
            a.command(q0 + np.radians([0.5, 0, 0, 0, 0, 0]))     # it moves: the full gain at once
            np.testing.assert_allclose(kd(), arm_mod.MIT_KD)


if __name__ == "__main__":
    unittest.main()
