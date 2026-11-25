"""One entry point for the real rig:  ./rig <task> [options of the task's script]

    ./rig status              camera server, CAN adapter, table map, arm poses, ball colour
    ./rig camera [start|stop|hd|vga]
                              the RealSense server: start it (asks for your password), stop it, or
                              switch the running one to 1280x720@30 / 640x480@60 (no password)
    ./rig view                the camera with the table markers, for aiming it
    ./rig arm                 joint states, read only: is the arm answering on CAN?
    ./rig calibrate           the table map once (--new when the table or a marker moved);
                              daily: wrist zero, arm poses, ball colour (opens there when the map exists)
    ./rig check               live check of the calibration: markers, fingertip, ball
    ./rig track               ball tracking and the planned arm motion; the arm is NOT powered
    ./rig move                POWERED: the jaw passes along the intercept line at 60 / 35 / 20 mm
    ./rig catch               POWERED: the jaw follows the ball
    ./rig test                unit tests, no hardware

Options after the task go to its script, e.g. ./rig calibrate --solve, ./rig move --heights 40,
./rig catch --help. The camera server is started first when a task needs it; each script switches
it to the size it works at.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "scripts" / "real"
sys.path.insert(0, str(ROOT))

HD, VGA = (1280, 720, 30), (640, 480, 60)
PORT = 8770                       # the camera server's port (scripts/real/rs_server.py)
TASKS = {                         # task -> (script, needs the camera)
    "view": ("view.py", True),
    "calibrate": ("calibrate.py", True),
    "check": ("check.py", True),
    "track": ("track.py", True),
    "move": ("move.py", False),
    "catch": ("catch.py", True),
}


def fmt(mode) -> str:
    return f"{mode[0]}x{mode[1]}@{mode[2]}"


def client():
    from eaiproject.realsense_client import RealSenseClient
    try:
        return RealSenseClient(port=PORT, timeout=2.0)
    except (OSError, ConnectionError, ValueError):
        return None


def camera_mode() -> tuple | None:
    """(width, height, fps) of the running camera server, or None."""
    c = client()
    if c is None:
        return None
    mode = (c.width, c.height, c.fps)
    c.close()
    return mode


def start_camera(mode=VGA) -> bool:
    w, h, f = mode
    return subprocess.run(["bash", str(HERE / "start_camera.sh"), "--width", str(w), "--height", str(h),
                           "--fps", str(f)]).returncode == 0 and camera_mode() is not None


def stop_camera() -> bool:
    c = client()
    if c is not None:
        c.stop_server()
        c.close()
        t_end = time.monotonic() + 5.0
        while camera_mode() is not None and time.monotonic() < t_end:
            time.sleep(0.2)
    print("camera server stopped" if camera_mode() is None else "the camera server did not stop")
    return camera_mode() is None


def switch_camera(mode) -> bool:
    """Switch the running server to `mode` (no password)."""
    from eaiproject.realsense_client import switch_mode
    c = client()
    if c is None:
        print("camera server not running: ./rig camera start")
        return False
    try:
        switch_mode(c, *mode).close()
    except TimeoutError as e:
        print(e)
        return False
    print(f"camera server: {fmt(camera_mode())}")
    return True


def ensure_camera() -> bool:
    """The camera server running (started at 640x480 @ 60 if it is not)."""
    cur = camera_mode()
    if cur is not None:
        print(f"camera server: {fmt(cur)}")
        return True
    return start_camera(VGA)


def can_adapter() -> str:
    usb = subprocess.run(["ioreg", "-p", "IOUSB", "-l", "-w0"], capture_output=True, text=True).stdout
    lib = (Path.home() / "lib" / "libPCBUSB.dylib").exists()
    if not ("PCAN" in usb or "PEAK" in usb or '"idVendor" = 3186' in usb):        # PEAK-System, 0x0c72
        return "PCAN-USB adapter not found on USB (plugged in?)" + ("" if lib else "; libPCBUSB missing in ~/lib")
    return "PCAN-USB adapter on USB" + ("" if lib else ", but libPCBUSB is missing in ~/lib (see docs/real_rig.md)")


def arm_states() -> int:
    """Joint states, the model angles and the fingertip (read only: nothing can be enabled)."""
    from eaiproject import arm as arm_mod
    from eaiproject import rig_calib as rc
    from eaiproject.kinematics import Fingertip
    from eaiproject.robstride import ARM_IDS, RobStrideBus, print_states, read_joint_states
    try:
        rs = RobStrideBus(movable=())
    except Exception as e:  # noqa: BLE001  (python-can raises several types)
        print(f"CAN adapter not available: {e}")
        return 1
    try:
        states = read_joint_states(rs)
    finally:
        rs.close()
    print_states(states)
    q = [states[m]["pos"] for m in ARM_IDS]
    if None in q:
        print("not every joint answered: arm power on? CAN cable?")
        return 1
    qm = arm_mod.load().to_model(q)
    p = Fingertip()(qm)
    print("model angles (deg, with config/arm.json's zero corrections): " + " ".join(f"{np.degrees(v):+.1f}" for v in qm))
    above = ""
    if rc.MAP_PATH.exists():
        above = f", {(p[2] - rc.plane_z(rc.RigMap.load(rc.MAP_PATH).plane, p[0], p[1])) * 1000:.0f} mm above the table"
    print(f"fingertip x {p[0]:.3f}  y {p[1]:+.3f}  z {p[2]:.3f} m{above}")
    return 0


def status() -> int:
    from eaiproject import arm as arm_mod
    from eaiproject import rig_calib as rc
    from eaiproject.real_intercept import hsv_saved_time
    cur = camera_mode()
    print(f"camera   {'server running at ' + fmt(cur) if cur else 'server not running (./rig camera start)'}")
    print(f"arm      {can_adapter()}  (./rig arm reads the joints)")
    if rc.MAP_PATH.exists():
        m = rc.RigMap.load(rc.MAP_PATH)
        line = [rc.plane_z(m.plane, rc.GATE_X, y) * 1000 for y in (-0.24, 0.0, 0.24)]
        print(f"map      {rc.MAP_PATH.relative_to(ROOT)} of {m.time}: markers {sorted(m.markers)} "
              f"(far: {[k for k in m.markers if m.is_far(k)]}), reprojection {m.stats.get('reproj_rms_px')} px, "
              f"touches {m.stats.get('touch_xy_rms_mm')} mm in the plane, {len(m.warnings)} warning(s)")
        print(f"         table on the intercept line {line[0]:.1f} / {line[1]:.1f} / {line[2]:.1f} mm (right / centre / left)")
    else:
        print("map      none: ./rig calibrate")
    cfg = arm_mod.load()
    if cfg.time:
        rec = [n for n, q in (("rest", cfg.rest), ("initial", cfg.initial)) if q is not None]
        print(f"poses    config/arm.json: wrist zero {np.degrees(cfg.offset[arm_mod.ROLL]):+.1f} deg "
              f"({cfg.info.get('level_time', 'not measured')}), poses: {', '.join(rec) or 'none'}")
    else:
        print("poses    none (./rig calibrate, step 4): the powered tasks start and end where the arm rests")
    when = hsv_saved_time()
    print(f"ball     colour {'saved ' + when if when else 'default (./rig calibrate, step 5)'}")
    sess = rc.latest_session()
    if sess is not None:
        n_v = len(json.loads((sess / "views.json").read_text())["views"]) if (sess / "views.json").exists() else 0
        n_t = len(json.loads((sess / "touches.json").read_text())["touches"]) if (sess / "touches.json").exists() else 0
        n_s = len(np.load(sess / "sweep.npz")["t"]) if (sess / "sweep.npz").exists() else 0
        print(f"session  {sess.relative_to(ROOT)}: {n_v} views, {n_t} touches, {n_s} sweep samples")
    return 0


def main(argv: list[str]) -> int:
    os.chdir(ROOT)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    task, args = argv[0], argv[1:]
    if task == "status":
        return status()
    if task == "arm":
        return arm_states()
    if task == "camera":
        sub = args[0] if args else ""
        if sub == "start":
            return 0 if ensure_camera() else 1
        if sub == "stop":
            return 0 if stop_camera() else 1
        if sub in ("hd", "vga"):
            return 0 if switch_camera(HD if sub == "hd" else VGA) else 1
        cur = camera_mode()
        print(f"camera server {'running at ' + fmt(cur) if cur else 'not running'}   (./rig camera start|stop|hd|vga)")
        return 0
    if task == "test":
        os.execv(sys.executable, [sys.executable, "-m", "unittest", "discover", "-s", "tests", *args])
    if task not in TASKS:
        print(f"unknown task '{task}'\n{__doc__}")
        return 2
    script, needs_camera = TASKS[task]
    offline = any(a in ("-h", "--help", "--solve") for a in args)           # calibrate --solve: no camera
    if needs_camera and not offline and not ensure_camera():
        print("no camera: see the messages above (docs/real_rig.md, troubleshooting)")
        return 1
    os.execv(sys.executable, [sys.executable, str(HERE / script), *args])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
