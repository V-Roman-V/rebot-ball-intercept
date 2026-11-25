"""The camera server's in-place size switch, with a fake pyrealsense2 (no camera, no root).

    .venv/bin/python -m unittest tests.test_camera_server -v
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eaiproject.realsense_client import RealSenseClient  # noqa: E402

PORT = 8779


def fake_realsense() -> types.ModuleType:
    rs = types.ModuleType("pyrealsense2")
    rs.stream = types.SimpleNamespace(depth="depth", color="color")
    rs.format = types.SimpleNamespace(z16="z16", bgr8="bgr8")
    rs.camera_info = types.SimpleNamespace(name="name", serial_number="serial")
    rs.option = types.SimpleNamespace(enable_auto_exposure="ae", exposure="exp")

    class config:
        def __init__(self):
            self.color = None

        def enable_stream(self, s, w, h, fmt, fps):
            if s == "color":
                self.color = (w, h, fps)

    class Sensor:
        def get_info(self, what):
            return "RGB Camera"

        def set_option(self, *a):
            pass

    class Device:
        def query_sensors(self):
            return [Sensor()]

        def get_info(self, what):
            return "FAKE123"

    class Profile:
        def __init__(self, w, h):
            self.w, self.h = w, h

        def get_device(self):
            return Device()

        def get_stream(self, s):
            intr = types.SimpleNamespace(width=self.w, height=self.h, fx=600.0, fy=600.0, ppx=self.w / 2, ppy=self.h / 2,
                                         coeffs=[0.0] * 5)
            return types.SimpleNamespace(as_video_stream_profile=lambda: types.SimpleNamespace(get_intrinsics=lambda: intr))

    class pipeline:
        def start(self, c):
            self.w, self.h, self.fps = c.color
            self.n = 0
            return Profile(self.w, self.h)

        def wait_for_frames(self, timeout=5000):
            time.sleep(1.0 / self.fps)
            self.n += 1
            img = np.full((self.h, self.w, 3), self.n % 255, np.uint8)
            frame = types.SimpleNamespace(get_data=lambda: img, get_frame_number=lambda: self.n,
                                          get_timestamp=lambda: self.n * 1000.0 / self.fps)
            return types.SimpleNamespace(get_color_frame=lambda: frame)

        def stop(self):
            pass

    rs.config, rs.pipeline = config, pipeline
    return rs


class CameraServerTest(unittest.TestCase):
    def setUp(self):
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        shutil.copy(ROOT / "scripts/real/rs_server.py", tmp / "rs_server.py")      # its pid file lands in tmp
        self.enterContext(mock.patch.dict(sys.modules, {"pyrealsense2": fake_realsense()}))
        spec = importlib.util.spec_from_file_location("rs_server_under_test", tmp / "rs_server.py")
        srv = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(srv)
        argv = ["rs_server", "--port", str(PORT), "--width", "640", "--height", "480", "--fps", "60"]
        with mock.patch.object(sys, "argv", argv):
            self.thread = threading.Thread(target=srv.main, daemon=True)
            self.thread.start()
            time.sleep(0.3)                                # argv is read at start
        for _ in range(50):
            try:
                RealSenseClient(port=PORT, timeout=1.0).close()
                break
            except OSError:
                time.sleep(0.1)

    def tearDown(self):
        try:
            RealSenseClient(port=PORT, timeout=1.0).stop_server()
        except OSError:
            pass
        self.thread.join(timeout=5.0)

    def test_switches_resolution_and_drops_clients(self):
        c = RealSenseClient(port=PORT)
        self.assertEqual((c.width, c.height, c.fps), (640, 480, 60))
        self.assertEqual(c.read(timeout=2.0).img.shape, (480, 640, 3))
        c.request_mode(1280, 720, 30)
        last, t_end = None, time.monotonic() + 5.0
        with self.assertRaises(ConnectionError):          # dropped: the header it read is out of date
            while time.monotonic() < t_end:
                last = c.read(timeout=0.5, newer_than=last).frame_number
        c2 = RealSenseClient(port=PORT)
        self.assertEqual((c2.width, c2.height, c2.fps), (1280, 720, 30))
        np.testing.assert_allclose(c2.K[0, 2], 640.0)
        self.assertEqual(c2.read(timeout=2.0).img.shape, (720, 1280, 3))
        c2.close()

    def test_switch_mode_returns_a_client_at_the_new_size(self):
        from eaiproject.realsense_client import switch_mode
        c = RealSenseClient(port=PORT)
        c2 = switch_mode(c, 1280, 720, 30)
        self.assertEqual((c2.width, c2.height, c2.fps), (1280, 720, 30))
        self.assertIs(switch_mode(c2, 1280, 720, 30), c2)                     # already there: the same client
        c2.close()

    def test_launcher_switches_and_stops_without_a_restart(self):
        spec = importlib.util.spec_from_file_location("rig_launcher", ROOT / "scripts/real/rig.py")
        rig = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rig)
        rig.PORT = PORT
        refuse = mock.Mock(side_effect=AssertionError("must not start a server (password)"))
        with mock.patch.object(rig, "start_camera", refuse), mock.patch("builtins.print"):
            self.assertEqual(rig.camera_mode(), (640, 480, 60))
            self.assertTrue(rig.switch_camera(rig.HD))
            self.assertEqual(rig.camera_mode(), rig.HD)
            self.assertTrue(rig.ensure_camera())                       # running: nothing to start
            self.assertTrue(rig.switch_camera(rig.VGA))
            self.assertEqual(rig.camera_mode(), rig.VGA)
            self.assertTrue(rig.stop_camera())
            self.assertIsNone(rig.camera_mode())


if __name__ == "__main__":
    unittest.main(verbosity=2)
