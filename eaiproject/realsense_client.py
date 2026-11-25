"""Client for scripts/real/rs_server.py: newest RealSense colour frame + hardware timestamp.

    cam = RealSenseClient()            # connects to 127.0.0.1:8770
    frame = cam.read()                 # Frame(img BGR, frame_number, hw_ts_ms, t_host); blocks for the next one
    cam.K, cam.dist, cam.width, cam.height
`t_host` is time.monotonic() when the frame reached the server (same clock in every process on
macOS), so `time.monotonic() - frame.t_host` is the age of the frame on the host side.
"""
from __future__ import annotations

import json
import socket
import struct
import threading
import time
from dataclasses import dataclass

import numpy as np

HDR = struct.Struct("<QddII")


@dataclass
class Frame:
    img: np.ndarray
    frame_number: int
    hw_ts_ms: float
    t_host: float


class RealSenseClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8770, timeout: float = 3.0):
        self.host, self.port = host, port
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._buf = b""
        info = json.loads(self._readline())
        self.width, self.height, self.fps = info["width"], info["height"], info["fps"]
        self.K = np.array(info["K"], float)
        self.dist = np.array(info["dist"], float)
        self.serial = info["serial"]
        self._latest: Frame | None = None
        self._cond = threading.Condition()
        self._closed = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _readline(self) -> bytes:
        while b"\n" not in self._buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("camera server closed the connection")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line

    def _recv_exact(self, n: int) -> bytes:
        out = bytearray(self._buf[:n])
        self._buf = self._buf[n:]
        while len(out) < n:
            chunk = self.sock.recv(min(1 << 20, n - len(out)))
            if not chunk:
                raise ConnectionError("camera server closed the connection")
            out += chunk
        return bytes(out)

    def _loop(self):
        self.sock.settimeout(None)
        try:
            while not self._closed:
                fn, ts, th, w, h = HDR.unpack(self._recv_exact(HDR.size))
                img = np.frombuffer(self._recv_exact(w * h * 3), np.uint8).reshape(h, w, 3)
                with self._cond:
                    self._latest = Frame(img, fn, ts, th)
                    self._cond.notify_all()
        except (ConnectionError, OSError):
            with self._cond:
                self._closed = True
                self._cond.notify_all()

    def read(self, timeout: float = 1.0, newer_than: int | None = None) -> Frame:
        """Newest frame; waits for one newer than `newer_than` (frame number) if given."""
        with self._cond:
            ok = self._cond.wait_for(lambda: self._closed or (self._latest is not None and (
                newer_than is None or self._latest.frame_number != newer_than)), timeout)
            if self._closed:
                raise ConnectionError("camera server closed the connection")
            if not ok:
                raise TimeoutError("no frame from the camera server")
            return self._latest

    def request_mode(self, width: int, height: int, fps: int) -> None:
        """Ask the server to restream at this size; it then drops every client (this one too)."""
        try:
            self.sock.sendall(f"mode {int(width)} {int(height)} {int(fps)}\n".encode())
        except OSError:
            pass

    def stop_server(self) -> None:
        try:
            self.sock.sendall(b"quit\n")
        except OSError:
            pass

    def close(self) -> None:
        self._closed = True
        try:
            self.sock.close()
        except OSError:
            pass


def switch_mode(cam, width: int, height: int, fps: int, timeout: float = 12.0):
    """A client of the server streaming at this size: `cam` itself if it already is (or cannot switch,
    e.g. a simulated camera); else the server restreams (no password) and a new client is returned.
    TimeoutError if the server does not switch (one started before the switch existed)."""
    want = (int(width), int(height), int(fps))
    if not hasattr(cam, "request_mode") or (cam.width, cam.height, getattr(cam, "fps", fps)) == want:
        return cam
    host, port = cam.host, cam.port
    cam.request_mode(*want)
    cam.close()
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        time.sleep(0.5)
        try:
            c = RealSenseClient(host, port, timeout=2.0)
        except (OSError, ConnectionError, ValueError):
            continue
        if (c.width, c.height, c.fps) == want:
            return c
        c.close()
    raise TimeoutError(f"the camera server did not switch to {want[0]}x{want[1]}@{want[2]}: ./rig camera stop, then try again")
