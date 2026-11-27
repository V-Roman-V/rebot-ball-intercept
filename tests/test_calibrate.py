"""./rig calibrate end to end on the simulated rig (tests/sim_rig.py), headless: the table map (views,
touch, sweep, solve), the offline re-solve, and the next day's arm and ball steps.

The OpenCV window functions are replaced by a scripted key sequence, the camera server and the CAN
arm by fakes, and every output path points into a temporary folder. Set RIG_SIM_SHOTS=<dir> to keep
the last frame each window showed (to look at the UI).

    .venv/bin/python -m unittest tests.test_calibrate -v
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import signal
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import sim_rig  # noqa: E402
from eaiproject import arm, cli, markers, real_intercept  # noqa: E402
from eaiproject import rig_calib as rc  # noqa: E402
from eaiproject.ball_intercept_env import BallInterceptEnv  # noqa: E402

BIAS = np.array([0.002, -0.001, 0.009])
VIEW_EYES = [(1.45, 0.25, 0.75), (1.40, -0.35, 0.60), (1.30, 0.05, 1.00), (1.50, 0.55, 0.85), (1.25, -0.55, 0.90),
             (1.60, -0.05, 0.65)]
REST = np.array([0.0946, -0.0029, -0.0001, -0.0003, -0.0537, 2.8035])     # the real arm's folded rest pose
INIT = np.array([0.0, 0.9, 0.9, 0.3, 0.0, 0.0])                            # raised, clear of the table
ROLL_ERR = np.radians(15.0) * np.eye(6)[5]                                 # the wrist motor's zero is off by this


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "real" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Keys:
    """Scripted keyboard: each step is (key or None, callable run just before it or None)."""

    def __init__(self, steps, name):
        self.steps, self.name, self.last = list(steps), name, None

    def imshow(self, win, img):
        self.last = img

    def waitkey(self, delay=0):
        if not self.steps:
            return ord("q")
        key, fn = self.steps.pop(0)
        if fn is not None:
            fn()
        if not self.steps and os.environ.get("RIG_SIM_SHOTS") and self.last is not None:
            cv2.imwrite(str(Path(os.environ["RIG_SIM_SHOTS"]) / f"{self.name}.png"), self.last)
        return -1 if key is None else ord(key)


def level_pose(pitch: float, yaw: float, y: float = 0.05) -> np.ndarray:
    """Model joint angles with the jaw level (fingers pitched down by `pitch`)."""
    env = BallInterceptEnv("forward")
    env.pitch = pitch
    return env.ik_jaw(0.40, y, 0.05, yaw=yaw, q0=env.q_home, iters=400)


class SimRigCase(unittest.TestCase):
    """Outputs in a temporary folder; a simulated camera and unpowered arm."""
    bars: dict = {}                                 # the sliders of the ball step

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.out_dir = self.tmp / "calibration"
        self.enterContext(mock.patch.multiple(rc, SESSIONS_DIR=self.tmp / "rig_calib", OUT_DIR=self.out_dir,
                                              MAP_PATH=self.tmp / "config" / "rig_map.json"))
        self.enterContext(mock.patch.object(real_intercept, "CAL_DIR", self.out_dir))
        self.enterContext(mock.patch.object(arm, "PATH", self.tmp / "config" / "arm.json"))
        self.addCleanup(signal.signal, signal.SIGINT, signal.getsignal(signal.SIGINT))   # the scripts' Ctrl+C = q
        self.addCleanup(setattr, cli, "_pressed", False)
        self.world = sim_rig.World()
        self.cam, self.arm = sim_rig.FakeCamera(self.world), sim_rig.FakeArm(self.world, BIAS)

    def run_script(self, name, argv, steps=(), cam=None, patches=()):
        keys = Keys(steps, f"{name}_{argv[0] if argv else 'main'}")
        out = io.StringIO()
        with mock.patch.object(sys, "argv", [name, *argv]), \
                mock.patch("eaiproject.realsense_client.RealSenseClient", lambda *a, **k: cam or self.cam), \
                mock.patch.object(arm, "ArmReader", lambda *a, **k: self.arm), \
                mock.patch.multiple(cv2, imshow=keys.imshow, waitKey=keys.waitkey, pollKey=lambda: -1, namedWindow=lambda *a: None,
                                    resizeWindow=lambda *a: None, destroyAllWindows=lambda: None,
                                    destroyWindow=lambda *a: None, getWindowProperty=lambda *a: 1.0,
                                    createTrackbar=lambda name, win, v, mx, cb: self.bars.__setitem__(name, v),
                                    getTrackbarPos=lambda n, w: self.bars[n],
                                    setTrackbarPos=lambda n, w, v: self.bars.__setitem__(n, v)), \
                redirect_stdout(out):
            for p in patches:
                self.enterContext(p)
            code = load_script(name).main()
        return code, out.getvalue()

    def calibrate_map(self):
        """./rig calibrate on a fresh session: six views, all corners of 107 and 880, a sweep, the solve."""
        world, idle, enter = self.world, [(None, None)] * 2, [("\r", None)]
        steps = []
        for eye in VIEW_EYES:                                               # 1. six camera positions
            steps += [(None, lambda e=eye: self.cam.set_pose(e)), (None, None), (" ", None)]
        steps += enter + idle
        # 2. all corners of the two markers within reach: 107, then n x3 past the far ones to 880
        for key, skip in (("107", 0), ("880", 3)):
            steps += [("n", None)] * skip
            for c in range(4):
                steps += [(None, lambda k=key, c=c: self.arm.touch(world.truth[k][c])), (" ", None)]
        steps += [("c", lambda: self.arm.touch(world.truth["880"][4]))] + enter
        # 3. hold SPACE while the fingertip rasters over the table
        t0 = time.monotonic()

        def start_sweep():
            self.arm.path = lambda t: (0.28 + 0.26 * (((t - t0) * 0.9) % 1.0), 0.27 * np.sin((t - t0) * 40.0))

        steps += [(None, start_sweep)] + idle + [(" ", lambda: time.sleep(0.02))] * 80
        steps += [(None, lambda: time.sleep(0.4))] * 4 + enter
        steps += idle                                                       # the result screen; then q
        code, out = self.run_script("calibrate", ["--session", "sim"], steps)
        self.arm.path = None
        return code, out


class CalibrateTest(SimRigCase):
    def test_table_map_then_the_daily_steps(self):
        code, out = self.calibrate_map()
        self.assertEqual(code, 0, out)
        sess = self.tmp / "rig_calib/sim"
        self.assertEqual(len(json.loads((sess / "views.json").read_text())["views"]), 6)
        touches = json.loads((sess / "touches.json").read_text())["touches"]
        self.assertEqual([(t["key"], t["corner"]) for t in touches],
                         [("107", c) for c in range(4)] + [("880", c) for c in range(5)])
        self.assertNotIn("CHECK", out)
        self.assertGreater(len(np.load(sess / "sweep.npz")["p"]), 500)
        self.assertIn("no warnings", out)
        rig_map = rc.RigMap.load(rc.MAP_PATH)
        kin = {k: P[:4] + BIAS for k, P in self.world.truth.items()}        # what the map should contain
        err = np.concatenate([np.linalg.norm(rig_map.markers[k] - kin[k], axis=1) for k in kin]) * 1000
        self.assertLess(err.max(), 2.0, f"map corner errors (mm): {err.round(2)}")
        self.assertAlmostEqual(rig_map.plane[0], self.world.plane[0] + BIAS[2], delta=0.001)
        sw = json.loads((sess / "report/report.json").read_text())["sweep"]
        self.assertLess(sw["gross"] + sw["lifted"] + sw["below"], 0.02 * sw["samples"], sw)   # nothing was lifted
        for f in ("report.txt", "layout.png", "sweep_heatmap.png", "reproj_view_00.png"):
            self.assertTrue((sess / "report" / f).exists(), f)

        # ---- the offline re-solve gives the same map
        code, out = self.run_script("calibrate", ["--solve"])
        self.assertEqual(code, 0, out)
        again = rc.RigMap.load(rc.MAP_PATH)
        for k in kin:
            np.testing.assert_allclose(again.markers[k], rig_map.markers[k], atol=1e-5)

        # ---- the next day it opens at step 4: wrist level at two spots (the motor reads 15 deg off),
        # rest, initial; then step 5: the ball alone on the table, auto-tune; ENTER leaves (saved)
        idle = [(None, None)] * 2
        motor = lambda q: (lambda: setattr(self.arm, "q", np.asarray(q, float)))
        steps = idle + [(None, motor(level_pose(0.6, 0.0) - ROLL_ERR)), (" ", None),
                        (None, motor(level_pose(0.9, 0.3, -0.1) - ROLL_ERR)), (" ", None),
                        ("n", motor(REST)), (" ", None), ("n", motor(INIT - ROLL_ERR)), (" ", None),
                        ("\r", lambda: setattr(self.world, "ball", (0.50, 0.10)))]
        steps += idle * 2 + [("a", None)] + idle * 2 + [("\r", None)]
        code, out = self.run_script("calibrate", [], steps)
        self.assertEqual(code, 0, out)
        cfg = arm.load()
        self.assertAlmostEqual(np.degrees(cfg.offset[5]), 15.0, delta=1.0)
        np.testing.assert_allclose(cfg.rest, REST, atol=2e-3)                    # motor readings
        np.testing.assert_allclose(cfg.initial_model(), INIT, atol=np.radians(1.0))
        self.assertIn("ball colour saved", out)
        self.assertTrue((self.out_dir / "ball_hsv.json").exists())
        self.assertIn("arm: wrist zero +1", out)                                # the summary at the end

    def test_a_view_with_two_markers_is_recorded_with_a_note(self):
        self.cam.set_pose((0.9, 0.0, 0.5), (0.30, 0.0, 0.0))                  # only two markers in view
        code, out = self.run_script("calibrate", ["--session", "few"], [(None, None), (" ", None), (None, None)])
        views = json.loads((self.tmp / "rig_calib/few/views.json").read_text())["views"]
        self.assertEqual(len(views), 1, out)
        self.assertEqual(len(views[0]["corners"]), 2)
        self.assertIn("CHECK view_00: only 2 marker(s) in this view", out)

    def test_starts_without_the_arm_and_picks_it_up(self):
        tries = iter([ConnectionError("no answer")] * 2)

        def connect(*a, **k):                              # the arm answers from the third try on
            e = next(tries, None)
            if e is not None:
                raise e
            return self.arm

        steps = [(None, None), ("4", None), (None, None), (None, None),     # step 4 without the arm: it stays open
                 (None, lambda: time.sleep(0.3)),                           # the arm is powered: it connects by itself
                 ("n", lambda: setattr(self.arm, "q", REST.copy())), (" ", None), ("q", None)]
        code, out = self.run_script("calibrate", ["--session", "x"], steps,
                                    patches=[mock.patch.object(arm, "ArmReader", connect),
                                             mock.patch.object(arm.ArmLink, "RETRY_S", 0.05)])
        self.assertEqual(code, 0, out)
        self.assertIn("ARM NOT CONNECTED: the arm does not answer", out)
        self.assertIn("rest pose saved", out)
        np.testing.assert_allclose(arm.load().rest, REST, atol=2e-3)


class SessionChoiceTest(unittest.TestCase):
    def test_continues_the_newest_session(self):
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(mock.patch.object(rc, "SESSIONS_DIR", tmp))
        cal = load_script("calibrate")
        a = lambda **k: argparse.Namespace(**{"session": None, "new": False, **k})
        on = lambda day: mock.patch.object(cal.time, "strftime", lambda f: day)
        with on("20261125"), redirect_stdout(io.StringIO()):
            self.assertEqual(cal.pick_session(a()).name, "20261125")              # nothing yet: today
        (tmp / "20261125").mkdir()
        (tmp / "20261125/views.json").write_text('{"views": [{}, {}, {}, {}, {}, {}]}')
        with on("20261126"), redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cal.pick_session(a()).name, "20261125")              # another day: continue it
        self.assertIn("6 views, 0 touches", out.getvalue())
        with on("20261126"), redirect_stdout(io.StringIO()):
            self.assertEqual(cal.pick_session(a(new=True)).name, "20261126")
            self.assertEqual(cal.pick_session(a(session="x")).name, "x")
        with on("20261125"), redirect_stdout(io.StringIO()):
            self.assertEqual(cal.pick_session(a(new=True)).name, "20261125_2")


class GuidedCornersTest(unittest.TestCase):
    """Markers the ArUco detector cannot read are found where a pose from the others predicts them."""

    def setUp(self):
        self.world = sim_rig.World()
        self.map = rc.RigMap({k: P[:4] for k, P in self.world.truth.items()}, self.world.plane, dict(markers.SIZES))
        self.cam = sim_rig.FakeCamera(self.world, 640, 480)
        self.cam.set_pose((1.55, 0.30, 0.95))             # 107 is too small here for the detector

    def truth_px(self, k):
        return rc.project(self.cam.K, self.cam.dist, self.cam.rvec, self.cam.tvec, self.world.truth[k][:4])

    def test_recovers_an_unread_marker(self):
        img = cv2.GaussianBlur(self.cam.read().img, (0, 0), 1.2)     # blurred until 107 cannot be read
        plain = markers.detect(img)
        self.assertNotIn("107", plain)
        found, guided = rc.detect_with_map(img, self.map, self.cam.K, self.cam.dist)       # pose from the other three
        self.assertIn("107", guided)
        for k in guided:
            self.assertLess(np.abs(found[k] - self.truth_px(k)).max(), 0.6, k)
        res = rc.relocalize(self.map, self.cam.K, self.cam.dist, plain, guided={k: found[k] for k in guided})
        self.assertEqual(res.guided, guided)
        self.assertLess(res.reproj_rms_px, 0.5)
        self.assertLess(np.linalg.norm(res.camera_position - [1.55, 0.30, 0.95]) * 1000, 5.0)

    def test_guided_markers_do_not_count_for_the_warnings(self):
        img = self.cam.read().img
        plain = markers.detect(img)
        found, guided = rc.detect_with_map(img, self.map, self.cam.K, self.cam.dist)
        two = {k: plain[k] for k in sorted(plain)[:2]}
        res = rc.relocalize(self.map, self.cam.K, self.cam.dist, two, guided={k: found[k] for k in found if k not in two})
        self.assertTrue(any("only 2 map marker" in w for w in res.warnings), res.warnings)

    def test_covered_marker_is_not_invented(self):
        img = self.cam.read().img
        c = self.truth_px("107").astype(np.int32)
        cv2.fillConvexPoly(img, cv2.convexHull(c + np.array([[-6, -6], [6, -6], [6, 6], [-6, 6]])), (120, 140, 170))   # a hand
        self.assertEqual(rc.detect_with_map(img, self.map, self.cam.K, self.cam.dist)[1], [])

    def test_wrong_pose_finds_nothing(self):
        img = self.cam.read().img
        rv, tv = self.cam.rvec + [0.0, 0.1, 0.0], self.cam.tvec       # every outline predicted > 3 px away
        self.assertEqual(rc.guided_corners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), self.map, self.cam.K, self.cam.dist, rv, tv), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
