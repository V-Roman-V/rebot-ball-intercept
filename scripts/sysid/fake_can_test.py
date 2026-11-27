"""Test the REAL hardware code path of collect.py without the arm: emulated RobStride motors on a
python-can virtual bus.

What this exercises (everything --sim skips): RobStrideBus framing and scaling, the preflight reads,
run_mode / limit_torque writes with readback and restore, enable/stop, the 200 Hz MIT loop with
feedback collection and the real clock. The emulated motors are crude (each joint follows its target
with a first-order lag, no gravity), so the recorded data is NOT for fitting.

Usage
    .venv/bin/python scripts/sysid/fake_can_test.py                          # short plan (chirp_A_small), ~60 s
    .venv/bin/python scripts/sysid/fake_can_test.py -- --only chirp_B         # anything after -- goes to collect.py
    .venv/bin/python scripts/sysid/fake_can_test.py --start 0 0.1 0.2 0 0 0   # emulated rest pose, rad

Expected result: "motors disabled", run_mode/limit_torque restored (1, 36.0 / 14.0), loop ~200 Hz.
The quality line "at least one train and one val segment" FAILs with the default short plan; that is
expected (run the full plan with `-- --only chirp_A_large chirp_A_small multisine_A chirp_B`).
"""
from __future__ import annotations

import argparse
import builtins
import importlib.util
import math
import struct
import sys
import threading
import time
from pathlib import Path

import can

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import eaiproject.robstride as robstride  # noqa: E402

CHANNEL = "rebot_fake_can"
P_MAX = 4 * math.pi
RANGES = {m: ((50.0, 36.0) if m <= 3 else (33.0, 14.0)) for m in range(1, 8)}   # v_max, t_max (RS06 / RS00)
P_RUN_MODE, P_LIMIT_TORQUE, P_MECHPOS, P_MECHVEL, P_IQF, P_VBUS = 0x7005, 0x700B, 0x7019, 0x701B, 0x701A, 0x701C


def f2u(x: float, lo: float, hi: float) -> int:
    return int(round((min(max(x, lo), hi) - lo) / (hi - lo) * 65535))


def u2f(u: int, lo: float, hi: float) -> float:
    return u / 65535 * (hi - lo) + lo


class Motor:
    def __init__(self, mid: int, q: float):
        self.mid, self.q, self.v = mid, q, 0.0
        self.enabled = False
        self.params = {P_RUN_MODE: 1, P_LIMIT_TORQUE: RANGES[mid][1], P_MECHPOS: q, P_MECHVEL: 0.0, P_IQF: 0.0,
                       P_VBUS: 48.2}


class Emulator(threading.Thread):
    """Answers like RobStride firmware: type 17 read, 18 write (refused while enabled for run_mode),
    3 enable / 4 stop / 1 MIT -> one type-2 feedback frame each."""

    def __init__(self, start_pose: list[float], lag: float = 0.08):
        super().__init__(daemon=True)
        self.stop_evt = threading.Event()
        self.lag = lag
        self.motors = {m: Motor(m, start_pose[m - 1] if m <= 6 else 0.2) for m in range(1, 8)}
        self.n_mit = 0
        self.writes: list[tuple[int, str, float]] = []

    def _feedback(self, bus: can.BusABC, m: Motor) -> None:
        v_max, t_max = RANGES[m.mid]
        arb = (2 << 24) | ((2 if m.enabled else 0) << 22) | (m.mid << 8) | robstride.HOST_ID
        data = struct.pack(">HHHH", f2u(m.q, -P_MAX, P_MAX), f2u(m.v, -v_max, v_max), f2u(0.5, -t_max, t_max), 315)
        bus.send(can.Message(arbitration_id=arb, is_extended_id=True, data=data))

    def run(self) -> None:
        bus = can.Bus(interface="virtual", channel=CHANNEL)
        try:
            while not self.stop_evt.is_set():
                msg = bus.recv(0.05)
                if msg is None:
                    continue
                comm = (msg.arbitration_id >> 24) & 0x1F
                m = self.motors.get(msg.arbitration_id & 0xFF)
                if m is None:
                    continue
                if comm == 1:                                  # MIT: pos, vel, kp, kd (big-endian u16)
                    p, _, _, _ = struct.unpack(">HHHH", bytes(msg.data))
                    if m.enabled:
                        dq = self.lag * (u2f(p, -P_MAX, P_MAX) - m.q)
                        m.q += dq
                        m.v = dq * 200.0
                    m.params[P_MECHPOS] = m.q
                    self.n_mit += 1
                    self._feedback(bus, m)
                elif comm in (3, 4):
                    m.enabled = comm == 3
                    if not m.enabled:
                        m.v = 0.0
                    self._feedback(bus, m)
                elif comm == 17:
                    idx = struct.unpack("<H", bytes(msg.data[:2]))[0]
                    val = m.params.get(idx, 0.0)
                    data = (struct.pack("<HHB3x", idx, 0, int(val)) if idx == P_RUN_MODE
                            else struct.pack("<HHf", idx, 0, float(val)))
                    bus.send(can.Message(arbitration_id=(17 << 24) | (m.mid << 8) | robstride.HOST_ID,
                                         is_extended_id=True, data=data))
                elif comm == 18:
                    idx = struct.unpack("<H", bytes(msg.data[:2]))[0]
                    val = msg.data[4] if idx == P_RUN_MODE else struct.unpack("<f", bytes(msg.data[4:8]))[0]
                    if m.enabled and idx == P_RUN_MODE:
                        continue
                    m.params[idx] = val
                    self.writes.append((m.mid, hex(idx), round(float(val), 3)))
        finally:
            bus.shutdown()

    def summary(self) -> str:
        final = {m: (int(self.motors[m].params[P_RUN_MODE]), round(float(self.motors[m].params[P_LIMIT_TORQUE]), 1))
                 for m in range(1, 7)}
        restored = all(v == (1, RANGES[m][1]) for m, v in final.items())
        return (f"[emulator] MIT frames {self.n_mit}, parameter writes {len(self.writes)}\n"
                f"[emulator] final (run_mode, limit_torque) per joint: {final}\n"
                f"[emulator] motor parameters restored: {'YES' if restored else 'NO'}")


def main() -> int:
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=float, nargs=6, default=[0.0] * 6, help="emulated rest pose of joints 1-6, rad")
    ap.add_argument("--out", default=None, help="output directory (default outputs/sysid/<time>_fakecan)")
    args = ap.parse_args(own)

    # route RealArm's CAN bus to the virtual channel and answer the "press Enter" prompts automatically
    robstride.resolve_can = lambda interface=None, channel=None: ("virtual", CHANNEL)
    real_input = builtins.input
    builtins.input = lambda prompt="": (print(prompt + "[auto: Enter]"), "")[1]

    emu = Emulator(args.start)
    emu.start()
    time.sleep(0.1)
    out = args.out or str(ROOT / "outputs/sysid" / (time.strftime("%Y%m%d_%H%M%S") + "_fakecan"))
    if "--only" not in extra:
        extra = ["--only", "chirp_A_small"] + extra
    sys.argv = ["collect.py", "--yes", "--out", out] + extra
    spec = importlib.util.spec_from_file_location("sysid_collect_cli", ROOT / "scripts/sysid/collect.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    try:
        rc = cli.main()
    finally:
        builtins.input = real_input
        emu.stop_evt.set()
        emu.join(2)
    print(emu.summary())
    print(f"collect.py exit code {rc} (3 = data-quality FAIL, expected with the short default plan)")
    return 0 if rc in (0, 3) else rc


if __name__ == "__main__":
    sys.exit(main())
