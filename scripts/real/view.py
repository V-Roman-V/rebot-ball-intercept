"""Live camera view with the table markers, for aiming the camera (reads from the camera server).

    ./rig view                          # q quit, s save a snapshot (outputs/snapshots/)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eaiproject import markers  # noqa: E402
from eaiproject.rig_draw import GREEN, RED, put, window_closed  # noqa: E402


def main() -> int:
    from eaiproject.realsense_client import RealSenseClient
    try:
        cam = RealSenseClient()
    except OSError:
        print("Camera server not running: ./rig camera start")
        return 2
    win = "camera (q quit, s snapshot)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    k = 2 if cam.width < 900 else 1
    cv2.resizeWindow(win, cam.width * k, cam.height * k)
    last_fn, t_prev, fps = None, time.monotonic(), 0.0
    while True:
        f = cam.read(timeout=2.0, newer_than=last_fn)
        last_fn = f.frame_number
        now = time.monotonic()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
        t_prev = now
        img = f.img.copy()
        found = markers.detect(f.img)
        markers.draw(img, found)
        put(img, f"{cam.width}x{cam.height}  {fps:4.1f} fps  frame age {(now - f.t_host) * 1000:3.0f} ms", (10, 24))
        put(img, f"table markers: {len(found)}  {sorted(found)}", (10, 48), GREEN if len(found) >= 3 else RED)
        cv2.imshow(win, img)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or window_closed(win):
            break
        if key == ord("s"):
            out = ROOT / "outputs" / "snapshots"
            out.mkdir(parents=True, exist_ok=True)
            p = out / f"snap_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(p), f.img)
            print("saved", p)
    cam.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
