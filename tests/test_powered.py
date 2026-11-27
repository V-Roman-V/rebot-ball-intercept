"""./rig move and ./rig catch on a fake CAN bus (motors that reach their targets at once) and the
simulated camera: the route rest -> initial -> work -> initial -> rest, the heights of the passes,
the jaw meeting a rolled ball, and the release at the end.

    .venv/bin/python -m unittest tests.test_powered -v
"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_rig  # noqa: E402
from test_arm import INIT, REST, ROLL15, FakeBus, SimTime, assert_same_pose  # noqa: E402
from test_calibrate import BIAS, SimRigCase  # noqa: E402
from eaiproject import arm, markers  # noqa: E402
from eaiproject import rig_calib as rc  # noqa: E402
from eaiproject.kinematics import Fingertip  # noqa: E402


class PoweredScriptsTest(SimRigCase):
    def setUp(self):
        super().setUp()
        a, b, c = self.world.plane                     # the map in the arm's kinematic frame (physical + BIAS)
        self.plane = np.array([a - b * BIAS[0] - c * BIAS[1] + BIAS[2], b, c])
        rc.RigMap({k: P[:4] + BIAS for k, P in self.world.truth.items()}, self.plane, dict(markers.SIZES)).save(rc.MAP_PATH)
        arm.ArmConfig(ROLL15, REST.copy(), INIT.copy()).save()
        self.cfg = arm.load()                          # as saved (4 decimals of a degree)
        self.bus = FakeBus(REST)
        self.asks = []
        self.hw = [mock.patch("eaiproject.robstride.RobStrideBus", lambda **k: self.bus),
                   mock.patch("builtins.input", lambda s="": self.asks.append(s) or ""),
                   mock.patch.object(arm, "time", SimTime())]

    def heights_mm(self, targets) -> np.ndarray:
        """Fingertip height above the table (mm) at motor targets."""
        tip = Fingertip()
        P = np.array([tip(q + ROLL15) for q in targets])
        return (P[:, 2] - rc.plane_z(self.plane, P[:, 0], P[:, 1])) * 1000

    def test_move(self):
        code, out = self.run_script("move", ["--heights", "40", "20", "--y", "0.12", "--speed", "0.2"], patches=self.hw)
        self.assertEqual(code, 0, out)
        self.assertEqual(len(self.asks), 1, self.asks)                          # ENTER before powering, nothing else
        got = [float(line.split()[-5]) for line in out.splitlines() if "mm above the table" in line and ("start" in line or "end" in line)]
        np.testing.assert_allclose(got, [40, 40, 20, 20], atol=1.0, err_msg=out)
        T = np.array(self.bus.targets)
        tip = Fingertip()
        xs = np.array([tip(q + ROLL15)[0] for q in T])
        h = self.heights_mm(T)
        on_line = np.abs(xs - rc.GATE_X) < 0.002
        self.assertGreater(on_line.sum(), 100)
        self.assertGreater(h[on_line].min(), 19.0)
        self.assertGreater(h.min(), 15.0, "a move between poses comes close to the table")
        self.assertTrue(self.bus.visits(self.cfg.initial, self.cfg.initial, self.cfg.rest))
        assert_same_pose(self.bus.q(), self.cfg.rest)
        self.assertEqual(self.bus.on, set())

    def test_catch(self):
        """A ball rolled towards the arm: the jaw is where it crosses the intercept line (MIT control,
        the default), no target dips toward the table, the gripper opens, everything is off at the end."""
        cam = sim_rig.FakeCamera(self.world, 640, 480)
        cam.set_pose((1.55, 0.30, 0.95), (0.55, 0.0, 0.0))
        start, vel, clock = np.array([0.95, 0.12]), np.array([-0.35, -0.04]), {}

        def roll():
            now = time.monotonic()
            time.sleep(max(0.0, 0.025 - (now - clock.get("last", 0.0))))           # ~40 frames per second
            clock["last"] = now = time.monotonic()
            t = now - clock.setdefault("t0", now)
            p = start + vel * t
            self.world.ball = tuple(p) if p[0] > 0.15 else None

        code, out = self.run_script("catch", [], [(None, roll)] * 110 + [("q", None)], cam=cam, patches=self.hw)
        self.assertEqual(code, 0, out)
        t = (start[0] - (rc.GATE_X - BIAS[0])) / -vel[0]                         # crossing, physical x = line - bias
        y_cross = start[1] + vel[1] * t + BIAS[1]                                # in the arm's frame
        self.assertIn("ball at the line", out)
        log = np.load(next((self.tmp / "hardware").glob("catch_*.npz")))
        arrived = np.flatnonzero(log["phase"] == 2)
        self.assertGreater(len(arrived), 0)
        self.assertLess(abs(log["jaw"][arrived[0], 1] - y_cross) * 1000, 10.0, out)
        self.assertIn("control: mit", out)
        self.assertEqual(set(self.bus.gains), set(arm.ARM_IDS))                  # driven with MIT frames
        self.assertIn("gripper open", out)
        self.assertAlmostEqual(self.bus.pos[arm.GRIPPER_ID], arm.GRIP_OPEN)
        working = np.array(log["q_sent"]) - ROLL15                              # what the loop sent, as motor targets
        self.assertGreater(self.heights_mm(working).min(), 19.0, "a target below the catch height")
        step = np.abs(np.diff(log["q_sent"], axis=0)).max(axis=1) / np.clip(np.diff(log["t"]), 1e-3, None)
        self.assertLess(step.max(), 3.0 * 1.3, "faster than the planner's joint speed")
        self.assertTrue(self.bus.visits(self.cfg.initial, self.cfg.initial, self.cfg.rest))
        assert_same_pose(self.bus.q(), self.cfg.rest)
        self.assertEqual(self.bus.on, set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
