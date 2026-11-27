"""The system identification's link to the rig (no hardware): joint offsets from the rig's calibration,
one set of controller gains, the same gravity feed-forward, the excitation's speed option.

    .venv/bin/python -m unittest tests.test_sysid
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eaiproject import arm  # noqa: E402
from eaiproject import rig_calib as rc  # noqa: E402
from eaiproject.real_intercept import InterceptPlanner  # noqa: E402
from eaiproject.sysid import excitation, model  # noqa: E402
from eaiproject.sysid.arm_io import rig_offset  # noqa: E402

REST = np.radians([267.2, 0.36, -0.04, -0.10, 1.10, 211.47])       # motor readings of the rig, 27 Nov 2026
CFG = arm.ArmConfig(np.radians([0, 0, 0, 0, 0, -32.8]))            # that day's wrist zero


class OffsetTest(unittest.TestCase):
    def test_offsets_are_the_ones_typed_by_hand_that_day(self):
        np.testing.assert_allclose(np.degrees(rig_offset(REST, CFG, -90.0)), [270.0, 0, 0, 0, 0, 212.8], atol=1e-6)

    def test_the_model_pose_is_near_its_zero_with_the_table_in_front(self):
        q = REST - rig_offset(REST, CFG, -90.0)
        np.testing.assert_allclose(np.degrees(q), [-2.8, 0.36, -0.04, -0.10, 1.10, -1.33], atol=1e-6)
        lo, hi = model.ArmModel().range.T
        self.assertTrue(np.all((q >= lo) & (q <= hi)))

    def test_a_base_that_faces_down_the_table_and_a_motor_in_its_own_turn(self):
        raw = np.radians([12.0, 5.0, 3.0, 0.0, 0.0, 40.0])
        np.testing.assert_allclose(np.degrees(rig_offset(raw, arm.ArmConfig(), 0.0)), 0.0, atol=1e-9)

    def test_the_base_turn_is_joint_1s_as_the_planner_has_it(self):
        for yaw in (-90.0, 90.0):
            self.assertAlmostEqual(InterceptPlanner(np.zeros(3), rc.TableFrame(yaw)).j1_turn, np.radians(yaw), places=9)


class SharedControllerTest(unittest.TestCase):
    def test_gains_and_caps_are_the_arms(self):
        self.assertIs(model.KP_DEFAULT, arm.MIT_KP)
        self.assertIs(model.KD_DEFAULT, arm.MIT_KD)
        self.assertIs(model.TORQUE_CAP_DEFAULT, arm.TORQUE_CAP)

    def test_gravity_feed_forward_is_the_same_in_both(self):
        g_arm, g_sysid = arm.Gravity(), model.ArmModel()
        for q in (excitation.POSE_A, excitation.POSE_B, np.radians([-90.0, 87, 16, 42, 0, 0])):
            np.testing.assert_allclose(g_arm(q), g_sysid.gravity(q), atol=0.02)      # N m


class ExcitationTest(unittest.TestCase):
    def test_speed_raises_the_amplitude_where_speed_limits_it(self):
        f = 1.0                                                     # Hz: here every joint is limited by speed, not by A_MAX
        a1, a2 = excitation.amplitude(f, 1.0), excitation.amplitude(f, 1.0, speed=2.0)
        np.testing.assert_allclose(a1, excitation.V_MAX / (2 * np.pi * f))
        np.testing.assert_allclose(a2, np.minimum(excitation.A_MAX, 2 * a1))
        np.testing.assert_allclose(excitation.amplitude(0.05, 1.0, speed=2.0), excitation.A_MAX)   # slow: the amplitude cap

    def test_steps_are_catch_like(self):
        seg = excitation.steps("steps_A", "train", excitation.POSE_A)
        t = np.arange(0.0, seg.duration, 0.001)
        q = np.array([seg.q(x) for x in t])
        np.testing.assert_allclose(q[0], excitation.POSE_A)
        np.testing.assert_allclose(q[-1], excitation.POSE_A)                 # it ends where it started
        self.assertTrue(np.all(np.abs(q - excitation.POSE_A) <= excitation.A_MAX + 1e-9))
        change = np.flatnonzero(np.abs(np.diff(q, axis=0)).max(axis=1) > 0)
        self.assertGreaterEqual(np.diff(change).min(), 16)                    # targets change 60 times a second, not more
        speed = np.abs(np.diff(q, axis=0)).max(axis=1)[change] * excitation.STEP_RATE
        self.assertLessEqual(speed.max(), max(excitation.STEP_SPEEDS) * 1.02)
        self.assertGreater(speed.max(), 2.0)                                  # and it does reach the fast moves
        self.assertGreater((np.abs(np.diff(q, axis=0)).max(axis=1) == 0).mean(), 0.9)   # stops between the moves

    def test_the_plan_is_unchanged_without_the_option(self):
        q0 = np.zeros(6)
        a, b = excitation.build_plan(q0), excitation.build_plan(q0, speed=1.0)
        for t in (5.0, 17.3, 29.0):
            np.testing.assert_allclose(a[3].q(t), b[3].q(t))


if __name__ == "__main__":
    unittest.main()
