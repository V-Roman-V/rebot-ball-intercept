"""Synthetic end-to-end test of eaiproject/rig_calib.py (no hardware).

A random marker layout on a slightly tilted table, several camera poses, corners with pixel noise,
fingertip touches with ~1.5 mm aiming noise and a sweep of raw samples along strokes, some of them
lifted off the table for a moment. The arm's model bias is a constant offset added to everything
measured through FK (touches and sweep), so the "kinematic frame" the solver must recover is the
physical layout shifted by that offset.

    .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eaiproject import rig_calib as rc  # noqa: E402
from eaiproject.ball_tracker import CameraCalib  # noqa: E402
from eaiproject.real_intercept import make_tracker, on_table  # noqa: E402

BIAS = np.array([0.003, -0.002, 0.009])          # constant FK model bias (m)
K_HD = np.array([[910.0, 0, 640.0], [0, 910.0, 360.0], [0, 0, 1]])
K_VGA = np.array([[607.0, 0, 320.0], [0, 607.0, 240.0], [0, 0, 1]])
DIST = np.array([0.02, -0.05, 0.0005, -0.0003, 0.0])
SIZE = 0.070


def look_at(eye, target):
    """OpenCV world -> camera pose of a camera at `eye` looking at `target`, image up = world z."""
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z])
    return cv2.Rodrigues(R)[0].reshape(3), -R @ np.asarray(eye, float)


class Rig:
    """Ground truth, in the kinematic frame (physical layout + BIAS)."""

    def __init__(self, seed: int, n_near: int = 3, n_far: int = 2):
        self.rng = np.random.default_rng(seed)
        r = self.rng
        self.plane = np.array([0.006 + BIAS[2], r.uniform(-0.004, 0.004), r.uniform(-0.004, 0.004)])
        near = [(0.30, -0.22), (0.32, 0.20), (0.46, -0.02)][:n_near]
        far = [(0.78, -0.20), (0.90, 0.18), (0.70, 0.05)][:n_far]
        self.sizes = {str(100 + i): SIZE for i in range(n_near + n_far)}
        self.keys = list(self.sizes)
        self.near = self.keys[:n_near]
        self.truth = {}
        for k, (x, y) in zip(self.keys, near + far):
            pose = (x + r.uniform(-0.02, 0.02), y + r.uniform(-0.02, 0.02), r.uniform(-np.pi, np.pi))
            self.truth[k] = rc.marker_points(self.plane, pose, SIZE)          # (5, 3) corners + centre

    def camera(self, i: int | None = None):
        r = self.rng
        az = r.uniform(-1.0, 1.0) if i is None else np.linspace(-0.9, 0.9, 6)[i % 6]
        d, h = r.uniform(0.9, 1.2), r.uniform(0.55, 0.9)
        eye = np.array([0.55 + d * np.cos(az), d * np.sin(az), h])
        return look_at(eye, [0.55 + r.uniform(-0.05, 0.05), r.uniform(-0.05, 0.05), 0.0])

    def detect(self, rvec, tvec, K, wh, noise_px):
        out = {}
        for k, P in self.truth.items():
            uv = rc.project(K, DIST, rvec, tvec, P[:4]) + self.rng.normal(0, noise_px, (4, 2))
            if np.all((uv > 5) & (uv < np.array(wh) - 5)):
                out[k] = uv
        return out

    def observations(self, touched: list[str], n_views: int = 6, noise_px: float = 0.3, touch_noise: float = 0.0015,
                     corners=(0, 1, 2, 3), lifts: int = 0, strokes: int = 24) -> rc.Observations:
        """`touch_noise`: in-plane aiming error per axis (the real session: 1.74 mm RMS in the plane); the
        height of a pressed tip scatters less (1 mm). `lifts`: strokes that leave the table for a moment."""
        r = self.rng
        views = []
        for i in range(n_views):
            rv, tv = self.camera(i)
            views.append(rc.View(f"view_{i:02d}", K_HD, DIST, 1280, 720, self.detect(rv, tv, K_HD, (1280, 720), noise_px)))
        touches = []
        for k in touched:
            for c in corners:
                noise = r.normal(0, 1, 3) * [touch_noise, touch_noise, min(touch_noise, 0.001)]
                touches.append(rc.Touch(k, c, self.truth[k][c] + noise))
        P, T, t0 = [], [], 0.0
        for i in range(strokes):                           # straight strokes at 0.1 m/s, 50 Hz
            a = np.array([r.uniform(0.26, 0.56), r.uniform(-0.26, 0.26)])
            b = np.array([r.uniform(0.26, 0.56), r.uniform(-0.26, 0.26)])
            n = max(int(np.linalg.norm(b - a) / 0.1 * 50), 10)
            xy = a + np.linspace(0, 1, n)[:, None] * (b - a)
            contact = np.c_[xy, rc.plane_z(self.plane, xy[:, 0], xy[:, 1])]
            if i < lifts:                                  # the tip leaves the table mid-stroke
                m = n // 2 + np.arange(-r.integers(5, 15), r.integers(5, 15))
                contact[m, 2] += r.uniform(0.003, 0.012) * np.sin(np.linspace(0, np.pi, len(m)))
            P.append(contact + r.normal(0, 0.0007, (n, 3)))
            T.append(t0 + np.arange(n) / 50.0)
            t0 += n / 50.0 + 1.0
        return rc.Observations(views, touches, rc.Sweep(np.concatenate(P), np.concatenate(T)), dict(self.sizes))

    def oracle_rms_mm(self, obs: rc.Observations) -> float:
        """Corner RMS error that the touch noise alone causes: the best rigid in-plane fit of the true
        touched corners to the touched points, applied to every corner."""
        src = np.array([self.truth[t.key][t.corner][:2] for t in obs.touches])
        dst = np.array([t.p[:2] for t in obs.touches])
        sc, dc = src.mean(axis=0), dst.mean(axis=0)
        U, _, Vt = np.linalg.svd((src - sc).T @ (dst - dc))
        R = Vt.T @ U.T
        allc = np.concatenate([self.truth[k][:4, :2] for k in self.keys])
        return float(np.sqrt(np.mean(np.sum(((allc - sc) @ R.T + dc - allc) ** 2, axis=1)))) * 1000

    def corner_errors_mm(self, rig_map: rc.RigMap, keys=None) -> np.ndarray:
        return np.concatenate([np.linalg.norm(rig_map.markers[k] - self.truth[k][:4], axis=1)
                               for k in (keys or self.keys)]) * 1000

    def ball_errors_mm(self, rig_map: rc.RigMap, points_xy, tracker: bool = False) -> np.ndarray:
        """Relocalise from a NEW camera pose at the tracking resolution, then locate balls (with
        `tracker`: through make_tracker() + BallTracker, as the live scripts do)."""
        rv, tv = self.camera()
        det = self.detect(rv, tv, K_VGA, (640, 480), 0.2)
        res = rc.relocalize(rig_map, K_VGA, DIST, det)
        errs = []
        for x, y in points_xy:
            centre = rc.from_plane(self.plane, rc.to_plane(self.plane, [[x, y, 0]])[0], rc.BALL_R)[0]   # true ball centre
            uv = rc.project(K_VGA, DIST, rv, tv, centre)
            if tracker:
                trk = make_tracker(CameraCalib(K_VGA, DIST, res.rvec, res.tvec, 640, 480), rig_map,
                                   hsv=((0, 0, 0), (1, 1, 1)))
                xy = trk.pixel_to_world(uv[0])
            else:
                xy = rc.pixel_to_ball_xy(K_VGA, DIST, res.rvec, res.tvec, rig_map.plane, uv)[0]
            errs.append(np.linalg.norm(xy - centre[:2]) * 1000)
        return np.array(errs)


GATE = [(0.40, y) for y in (-0.24, -0.12, 0.0, 0.12, 0.24)]
LANE = GATE + [(x, y) for x in (0.55, 0.70) for y in (-0.2, 0.0, 0.2)]


class SolverTest(unittest.TestCase):
    def test_recovers_corners_within_2mm(self):
        worst, rmss = [], []
        for seed in range(5):
            rig = Rig(seed)
            obs = rig.observations(rig.near)
            sol = rc.solve_rig(obs)
            e = rig.corner_errors_mm(sol.rig_map)
            worst.append(e.max())
            rmss.append(np.sqrt(np.mean(e ** 2)))
            # 12 touches place the map to ~0.5 mm and ~2-3 mm per metre of yaw, so single far corners
            # can be 3-4 mm off in an unlucky draw: the solver must add little to what the touch noise
            # alone causes, and the 2 mm requirement holds for the typical (median) calibration.
            self.assertLess(rmss[-1], rig.oracle_rms_mm(obs) + 1.0, f"seed {seed}: corner RMS error")
            self.assertLess(sol.report["reproj_rms_px"], 1.0)
            self.assertLess(abs(sol.rig_map.plane[0] - rig.plane[0]) * 1000, 0.5)       # height bias absorbed
            self.assertEqual(sol.warnings, [])
        self.assertLess(np.median(rmss), 2.0)
        print(f"\n  corner RMS over 5 seeds: {np.round(rmss, 2)} mm, max single corner {max(worst):.2f} mm")

    def test_exact_without_noise(self):
        rig = Rig(0)
        sol = rc.solve_rig(rig.observations(rig.near, noise_px=0.0, touch_noise=0.0))
        self.assertLess(rig.corner_errors_mm(sol.rig_map).max(), 0.5)       # what is left: sweep noise
        self.assertLess(sol.report["reproj_rms_px"], 0.01)

    def test_report_contents(self):
        rig = Rig(1)
        sol = rc.solve_rig(rig.observations(rig.near))
        r = sol.report
        self.assertEqual(set(r["per_marker_reproj_px"]), set(rig.keys))
        self.assertEqual(len(r["per_view_reproj_px"]), 6)
        self.assertEqual(len(r["touches"]), 12)
        self.assertLess(abs(r["touch_scale_percent"]), rc.MAX_SCALE * 100)
        self.assertEqual(len(sol.sweep_residuals), r["n_sweep_cells"])
        self.assertEqual(len(sol.sweep_points), r["n_sweep_cells"])
        self.assertEqual(set(r["far_markers"]), set(rig.keys[3:]))

    def test_report_files(self):
        import tempfile
        from eaiproject.rig_report import save_report
        rig = Rig(1)
        obs = rig.observations(rig.near, lifts=3)
        sol = rc.solve_rig(obs)
        with tempfile.TemporaryDirectory() as d:
            out, text = save_report(sol, obs, Path(d))
            self.assertIn("RESULT: OK", text)
            for f in ("report.txt", "report.json", "layout.png", "sweep_heatmap.png"):
                self.assertTrue((out / f).exists(), f)

    def test_map_roundtrip(self):
        rig = Rig(2)
        m = rc.solve_rig(rig.observations(rig.near)).rig_map
        m2 = rc.RigMap.from_json(json.loads(json.dumps(m.to_json())))
        for k in m.markers:
            np.testing.assert_allclose(m2.markers[k], m.markers[k], atol=1e-5)
        self.assertEqual(m2.touched, m.touched)
        np.testing.assert_allclose(m2.plane, m.plane)

    def test_relocalise_new_pose_ball_error(self):
        worst = 0.0
        for seed in range(5):
            rig = Rig(seed)
            m = rc.solve_rig(rig.observations(rig.near)).rig_map
            e = rig.ball_errors_mm(m, LANE)
            worst = max(worst, e.max())
            self.assertLess(e.max(), 3.0, f"seed {seed}: ball-plane error {e.max():.2f} mm")
        print(f"\n  max ball-plane error over 5 seeds (new camera pose, 640x480): {worst:.2f} mm")

    def test_ball_through_the_real_tracker_path(self):
        """The end-to-end sign convention (table + r on the camera's side) on the tilted table plane."""
        for seed in range(5):
            rig = Rig(seed)
            m = rc.solve_rig(rig.observations(rig.near)).rig_map
            e = rig.ball_errors_mm(m, LANE, tracker=True)
            self.assertLess(e.max(), 3.0, f"tracker-path error {e.max():.2f} mm")

    def test_two_touched_markers(self):
        for seed in range(5):
            rig = Rig(seed)
            sol = rc.solve_rig(rig.observations(rig.near[:2]))
            e = rig.corner_errors_mm(sol.rig_map)
            self.assertLess(np.sqrt(np.mean(e ** 2)), 4.0)       # 8 touches: weaker, far markers feel the yaw lever arm
            self.assertLess(rig.ball_errors_mm(sol.rig_map, GATE).max(), 4.0)

    def test_untouched_marker_only_in_a_later_view(self):
        """The initial guess has to chain: a far marker that the first views do not show."""
        rig = Rig(4)
        obs = rig.observations(rig.near)
        for v in obs.views[:3]:
            v.corners.pop(rig.keys[-1], None)
        self.assertLess(rig.corner_errors_mm(rc.solve_rig(obs).rig_map).max(), 2.0)


class WarningsAndFailuresTest(unittest.TestCase):
    def test_warns_on_one_touched_marker(self):
        rig = Rig(0)
        sol = rc.solve_rig(rig.observations(rig.near[:1]))
        self.assertTrue(any("only one marker touched" in w for w in sol.warnings), sol.warnings)

    def test_warns_on_a_wrong_marker_size(self):
        """Markers printed 1 % larger than markers.py says: the camera's layout is 1 % too small."""
        rig = Rig(0)
        obs = rig.observations(rig.near)
        c = np.mean([t.p for t in obs.touches], axis=0)
        for t in obs.touches:
            t.p = c + (t.p - c) * [1.01, 1.01, 1.0]
        sol = rc.solve_rig(obs)
        self.assertGreater(sol.report["touch_scale_percent"], 0.6)
        self.assertTrue(any("measure a printed marker" in w for w in sol.warnings), sol.warnings)

    def test_fails_only_when_nothing_can_be_placed(self):
        rig = Rig(0)
        with self.assertRaisesRegex(rc.CalibrationError, "touched points"):
            rc.solve_rig(rig.observations([]))
        obs = rig.observations(rig.near)
        for v in obs.views:
            v.corners = {}
        with self.assertRaisesRegex(rc.CalibrationError, "no table marker"):
            rc.solve_rig(obs)

    def test_two_markers_solve_with_a_warning(self):
        rig = Rig(0)
        obs = rig.observations(rig.near)
        for v in obs.views:
            v.corners = {k: c for k, c in v.corners.items() if k in rig.keys[:2]}
        sol = rc.solve_rig(obs)
        self.assertEqual(sorted(sol.rig_map.markers), rig.keys[:2])
        self.assertTrue(any("only 2 table marker" in w for w in sol.warnings), sol.warnings)

    def test_leaves_out_what_is_not_linked(self):
        """A view that sees only markers no other view sees, none of them touched: left out, with them."""
        rig = Rig(0)
        obs = rig.observations(rig.near)
        far = [k for k in rig.keys if k not in rig.near]
        island = obs.views[0]
        island.corners = {k: c for k, c in island.corners.items() if k in far}
        self.assertTrue(island.corners)
        for v in obs.views[1:]:
            v.corners = {k: c for k, c in v.corners.items() if k not in far}
        sol = rc.solve_rig(obs)
        self.assertEqual(sorted(sol.rig_map.markers), sorted(rig.near))
        self.assertNotIn(island.name, sol.cams)
        self.assertTrue(any("left out" in w and island.name in w for w in sol.warnings), sol.warnings)

    def test_relocalise_warns_on_few_or_near_markers(self):
        rig = Rig(0)
        m = rc.solve_rig(rig.observations(rig.near)).rig_map
        rv, tv = rig.camera()
        det = rig.detect(rv, tv, K_VGA, (640, 480), 0.2)
        true_c = rc.camera_centre(rv, tv)
        self.assertEqual(rc.relocalize(m, K_VGA, DIST, det).warnings, [])
        res = rc.relocalize(m, K_VGA, DIST, {k: det[k] for k in rig.keys[3:]})
        self.assertTrue(any("only 2 map marker" in w for w in res.warnings), res.warnings)
        res = rc.relocalize(m, K_VGA, DIST, {k: det[k] for k in rig.near})
        self.assertTrue(any("far from the arm" in w for w in res.warnings), res.warnings)
        res = rc.relocalize(m, K_VGA, DIST, {k: det[k] for k in rig.near[:1]})         # one marker places it
        self.assertEqual(len(res.warnings), 2)
        self.assertLess(np.linalg.norm(res.camera_position - true_c), 0.10)
        with self.assertRaisesRegex(rc.CalibrationError, "no map marker"):
            rc.relocalize(m, K_VGA, DIST, {})

    def test_relocalise_drops_a_moved_marker(self):
        rig = Rig(0)
        m = rc.solve_rig(rig.observations(rig.near)).rig_map
        rv, tv = rig.camera()
        det = rig.detect(rv, tv, K_VGA, (640, 480), 0.2)
        det[rig.near[0]] = det[rig.near[0]] + 12.0                # this marker was nudged on the table
        res = rc.relocalize(m, K_VGA, DIST, det)
        self.assertEqual(res.rejected, [rig.near[0]])
        self.assertLess(res.reproj_rms_px, 1.0)

    def test_wrong_sign_of_r_fails_loudly(self):
        rig = Rig(0)
        m = rc.solve_rig(rig.observations(rig.near)).rig_map
        rv, tv = rig.camera()
        res = rc.relocalize(m, K_VGA, DIST, rig.detect(rv, tv, K_VGA, (640, 480), 0.2))
        with self.assertRaisesRegex(rc.CalibrationError, "radius must be positive"):
            rc.pixel_to_ball_xy(K_VGA, DIST, res.rvec, res.tvec, m.plane, [[320, 240]], r=-rc.BALL_R)
        with self.assertRaisesRegex(rc.CalibrationError, "not above"):          # the tracker refuses a camera under the table
            on_table(CameraCalib(K_VGA, DIST, res.rvec, res.tvec, 640, 480), [2.0, 0, 0])
        flipped = -np.asarray(m.plane)                         # a plane whose "up" is away from the camera
        with self.assertRaisesRegex(rc.CalibrationError, "from above"):
            rc.pixel_to_ball_xy(K_VGA, DIST, res.rvec, res.tvec, [flipped[0] + 2.0, 0, 0], [[320, 240]])

    def test_pose_monitor_warns_after_a_bump(self):
        rig = Rig(0)
        m = rc.solve_rig(rig.observations(rig.near)).rig_map
        rv, tv = rig.camera()
        det = rig.detect(rv, tv, K_VGA, (640, 480), 0.2)
        res = rc.relocalize(m, K_VGA, DIST, det)
        mon = rc.PoseMonitor(m, K_VGA, DIST, res.rvec, res.tvec)
        self.assertIsNone(mon.update(det, 0.0))
        bumped = rig.detect(rv + [0.0, 0.03, 0.0], tv, K_VGA, (640, 480), 0.2)      # a few px
        self.assertIsNone(mon.update(bumped, 3.0))              # one bad check is not enough
        self.assertIn("CAMERA MOVED", mon.update(bumped, 6.0))
        self.assertIsNone(mon.update(det, 9.0))


class SweepTest(unittest.TestCase):
    def test_lifted_strokes_are_rejected(self):
        for seed in range(3):
            rig = Rig(seed)
            obs = rig.observations(rig.near, lifts=8)
            sol = rc.solve_rig(obs)
            sw = sol.report["sweep"]
            self.assertGreaterEqual(sw["lifted"], 8 * 8, sw)                    # at least the high part of every lift
            self.assertLess(sw["lifted"] + sw["below"], 0.1 * sw["samples"], sw)
            self.assertLess(abs(sol.rig_map.plane[0] - rig.plane[0]) * 1000, 0.5)
            for y in (-0.24, 0.0, 0.24):                                       # table height on the intercept line
                self.assertAlmostEqual(rc.plane_z(sol.rig_map.plane, rc.GATE_X, y), rc.plane_z(rig.plane, rc.GATE_X, y),
                                       delta=0.0007)
            h = obs.sweep.p[:, 2] - rc.plane_z(rig.plane, obs.sweep.p[:, 0], obs.sweep.p[:, 1])
            self.assertGreaterEqual(len(sol.sweep_rejected), int((h > 0.004).sum()))   # the high samples are all gone

    def test_lift_test_is_one_sided_and_padded_in_time(self):
        rng = np.random.default_rng(0)
        n = 3000
        t = np.arange(n) / 50.0
        xy = np.c_[rng.uniform(0.3, 0.5, n), rng.uniform(-0.2, 0.2, n)]
        h = rng.normal(0, 0.0008, n) + 0.002 * (xy[:, 1] / 0.2) ** 2       # smooth model error: not lifting
        h[1000:1010] += 0.006                                                # lifted for 0.2 s
        h[2000] -= 0.002                                                     # a low sample: kept (> -3 mm)
        h[2500] += 0.005                                                     # a lone spike: dropped, not padded
        lifted, below, sig = rc.lifted_samples(h, xy, t, np.ones(n, bool))
        self.assertTrue(lifted[2500])
        self.assertFalse(lifted[2499] or lifted[2501])
        self.assertTrue(lifted[1000:1010].all())
        self.assertTrue(lifted[995:1015].all())                              # +-0.12 s around it
        self.assertLess(lifted.sum(), 40)
        self.assertFalse(below[2000])
        self.assertLess(sig, 0.001)
        np.testing.assert_array_equal(rc.expand_in_time([0, 1, 2, 3], [False, True, False, False], 1.0),
                                      [True, True, True, False])
        np.testing.assert_array_equal(rc.count_in_time([3, 0, 1, 2], [True, True, False, True], 1.0), [2, 1, 2, 2])

    def test_lifted_touch_height_is_ignored(self):
        rig = Rig(0)
        obs = rig.observations(rig.near)
        obs.touches[5].p = obs.touches[5].p + [0, 0, 0.008]
        sol = rc.solve_rig(obs)
        self.assertEqual([i for i, t in enumerate(sol.report["touches"]) if t["lifted"]], [5])
        self.assertTrue(any("tip lifted" in w for w in sol.warnings))
        self.assertLess(abs(sol.rig_map.plane[0] - rig.plane[0]) * 1000, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
