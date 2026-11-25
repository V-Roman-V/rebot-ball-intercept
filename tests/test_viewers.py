"""./rig view, check and track on the simulated rig (headless), with a table map made from the
simulated truth: the camera located from a new pose at the tracking size, fingertip and ball
errors recorded, and a camera bump noticed while tracking.

    .venv/bin/python -m unittest tests.test_viewers -v
"""
from __future__ import annotations

import csv
import json
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_rig  # noqa: E402
from test_calibrate import BIAS, SimRigCase  # noqa: E402
from eaiproject import markers  # noqa: E402
from eaiproject import rig_calib as rc  # noqa: E402


class ViewersTest(SimRigCase):
    def setUp(self):
        super().setUp()
        a, b, c = self.world.plane                     # the map in the arm's kinematic frame (physical + BIAS)
        plane = np.array([a - b * BIAS[0] - c * BIAS[1] + BIAS[2], b, c])
        rc.RigMap({k: P[:4] + BIAS for k, P in self.world.truth.items()}, plane, dict(markers.SIZES)).save(rc.MAP_PATH)
        self.cam2 = sim_rig.FakeCamera(self.world, 640, 480)
        self.cam2.set_pose((1.55, 0.30, 0.95), (0.55, 0.0, 0.0))

    def test_view(self):
        code, out = self.run_script("view", [], [(None, None)] * 3)
        self.assertEqual(code, 0, out)

    def test_check(self):
        """Fingertip on a corner, then the ball on a marker centre and on a touched point; the camera is
        moved and located again (c)."""
        world, arm, cam2, idle = self.world, self.arm, self.cam2, [(None, None)] * 2
        centre = world.truth["467"][4]
        steps = idle + [(None, lambda: arm.touch(world.truth["107"][1])), (None, None), (" ", None),
                        (None, lambda: arm.touch(world.on_table([(0.45, 0.05)])[0])), ("t", None),
                        ("b", lambda: setattr(world, "ball", tuple(centre[:2]))), (None, None), (" ", None),
                        (None, lambda: setattr(world, "ball", (0.45, 0.05))), (None, None), (" ", None), ("s", None),
                        ("c", lambda: cam2.set_pose((1.50, -0.30, 0.90))), (None, None), (None, None), (" ", None)]
        code, out = self.run_script("check", ["--label", "sim1"], steps + idle, cam=cam2)
        self.assertEqual(code, 0, out)
        pose = json.loads((self.out_dir / "camera_pose.json").read_text())
        self.assertEqual((pose["width"], pose["height"]), (640, 480))
        self.assertLess(np.linalg.norm(np.array(pose["camera_position"]) - (np.array([1.50, -0.30, 0.90]) + BIAS)) * 1000, 8.0)
        with (self.out_dir / "validation.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual([r["kind"] for r in rows], ["touch", "ball", "ball", "ball"], out)   # the last one after the camera moved
        self.assertEqual(rows[2]["point"], "point 1")
        for r in rows:
            self.assertLess(float(r["err_mm"]), 4.0, r)
        self.assertEqual(len(list(self.out_dir.glob("check_sim1_*.png"))), 1)

    def test_check_refuses_with_too_few_markers(self):
        cam3 = sim_rig.FakeCamera(self.world, 640, 480)
        cam3.set_pose((0.9, 0.0, 0.5), (0.30, 0.0, 0.0))
        code, out = self.run_script("check", [], cam=cam3)
        self.assertEqual(code, 1, out)
        self.assertIn("CANNOT LOCATE THE CAMERA", out)

    def test_track_notices_a_moved_camera(self):
        """The pose monitor warns after the camera is bumped; c locates it again."""
        cam2 = self.cam2
        self.world.ball = (0.80, 0.05)
        monitor = rc.PoseMonitor
        fast = lambda *a, **k: monitor(*a, **{**k, "period": 0.05})
        wait = [(None, lambda: time.sleep(0.03))] * 8
        steps = wait + [(None, lambda: cam2.set_pose((1.55, 0.32, 0.95), (0.56, 0.0, 0.0)))] + wait * 2 + [("c", None)] + wait * 2
        code, out = self.run_script("track", [], steps, cam=cam2, patches=[mock.patch.object(rc, "PoseMonitor", fast)])
        self.assertEqual(code, 0, out)
        self.assertIn("CAMERA MOVED", out)
        self.assertEqual(out.count("camera located:"), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
