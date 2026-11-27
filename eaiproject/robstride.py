"""RobStride private CAN protocol over a PEAK PCAN-USB adapter (python-can + PCBUSB on macOS).

Shared by scripts/real/gripper_test.py and scripts/real/joint_check.py.

Safety rule enforced in `RobStrideBus._send`: enable (type 3), MIT (type 1) and parameter writes
(type 18) may only go to motor IDs in `RobStrideBus.movable` (default: only the gripper). Reads
(type 17) and stop/disable (type 4) are allowed to any motor.

Protocol (29-bit extended IDs; comm type in bits 24-28):
    type 1  MIT command    id = 1<<24 | torque_u16<<8 | motor     data = pos,vel,kp,kd  (u16 big-endian)
    type 2  feedback       id = 2<<24 | mode/faults<<16 | motor<<8 | host   data = pos,vel,tau,temp*10
    type 3  enable / type 4 stop (data[0]=0: keep faults)          id = t<<24 | host<<8 | motor
    type 17 read param / type 18 write param (index u16 LE, value at bytes 4..7 LE)
"""
from __future__ import annotations

import math
import struct
import sys
import time
from dataclasses import dataclass

import can

HOST_ID = 0xFD
ARM_IDS = (1, 2, 3, 4, 5, 6)          # read-only, never enabled
GRIPPER_ID = 7
JOINT_NAMES = {1: "joint1", 2: "joint2", 3: "joint3", 4: "joint4", 5: "joint5", 6: "joint6", 7: "gripper"}

# Gripper travel from Seeed's rebotarm_hardware.yaml (motor rad): close 0.0, open 5.0
GRIP_CLOSE, GRIP_OPEN = 0.0, 5.0
TARGET_MIN, TARGET_MAX = 0.1, 5.0     # never drive into the closed hard stop

# RS-00 MIT scaling (RobStride manual). Low gains are used, so even a 10x range error stays gentle.
# MIT / feedback scaling per motor model (RobStride manuals). The torque ranges match the motors'
# own limit_torque parameter read on this arm (RS06: 36 Nm, RS00: 14 Nm).
P_MAX = 4 * math.pi
MODEL_RANGES = {            # v_max rad/s, t_max Nm, kp_max, kd_max
    "rs-06": (50.0, 36.0, 5000.0, 100.0),
    "rs-00": (33.0, 14.0, 500.0, 5.0),
}
MOTOR_MODEL = {1: "rs-06", 2: "rs-06", 3: "rs-06", 4: "rs-00", 5: "rs-00", 6: "rs-00", 7: "rs-00"}


def ranges(mid: int) -> tuple[float, float, float, float]:
    return MODEL_RANGES[MOTOR_MODEL[mid]]

P_MECHPOS, P_MECHVEL, P_IQF, P_VBUS, P_RUN_MODE = 0x7019, 0x701B, 0x701A, 0x701C, 0x7005
P_LIMIT_TORQUE = 0x700B      # float, Nm; caps the output torque in every mode (RAM write only)
T_MIT, T_FEEDBACK, T_ENABLE, T_STOP, T_READ, T_WRITE = 1, 2, 3, 4, 17, 18


def f2u(x: float, lo: float, hi: float) -> int:
    x = min(max(x, lo), hi)
    return int(round((x - lo) / (hi - lo) * 65535))


def u2f(u: int, lo: float, hi: float) -> float:
    return u / 65535 * (hi - lo) + lo


@dataclass
class Feedback:
    pos: float
    vel: float
    tau: float
    temp_c: float
    faults: int        # bits 16-21 of the id
    mode: int          # bits 22-23: 0 reset, 1 calibration, 2 running
    t: float


def resolve_can(interface: str | None = None, channel: str | None = None) -> tuple[str, str]:
    """Pick the python-can interface and channel for this OS.

    macOS: "pcan" (PEAK adapter through the PCBUSB library), channel PCAN_USBBUS1.
    Linux: "socketcan" (kernel peak_usb driver), channel can0. Bring the link up once per boot:
           sudo ip link set can0 up type can bitrate 1000000
    A channel name starting with "PCAN_" always means the pcan interface (PCAN-Basic on Linux too).
    """
    if interface in (None, "", "auto"):
        if channel and channel.upper().startswith("PCAN_"):
            interface = "pcan"
        else:
            interface = "socketcan" if sys.platform.startswith("linux") else "pcan"
    if not channel:
        channel = "can0" if interface == "socketcan" else "PCAN_USBBUS1"
    return interface, channel


class RobStrideBus:
    def __init__(self, channel: str | None = None, bitrate: int = 1_000_000,
                 movable: tuple[int, ...] = (GRIPPER_ID,), interface: str | None = None):
        self.interface, self.channel = resolve_can(interface, channel)
        if self.interface == "socketcan":     # bitrate is set on the link (ip link), not here
            self.bus = can.Bus(interface="socketcan", channel=self.channel)
        else:
            self.bus = can.Bus(interface=self.interface, channel=self.channel, bitrate=bitrate)
        self.movable = tuple(movable)
        self.last_fb: dict[int, Feedback] = {}

    def close(self) -> None:
        self.bus.shutdown()

    def _send(self, comm: int, mid: int, data: bytes, mid_field: int | None = None) -> None:
        if comm in (T_MIT, T_ENABLE, T_WRITE) and mid not in self.movable:
            raise RuntimeError(f"refusing to send command type {comm} to motor {mid}: not in movable={self.movable}")
        field = (HOST_ID if mid_field is None else mid_field) & 0xFFFF
        msg = can.Message(arbitration_id=(comm << 24) | (field << 8) | mid, is_extended_id=True, data=data)
        try:
            self.bus.send(msg, timeout=0.01)
        except can.CanOperationError:          # SocketCAN: tx queue momentarily full (ENOBUFS); retry once
            time.sleep(0.001)
            self.bus.send(msg, timeout=0.01)

    def _handle(self, m: can.Message) -> tuple[int, int, bytes] | None:
        if m is None or not m.is_extended_id or m.is_error_frame:
            return None
        comm = (m.arbitration_id >> 24) & 0x1F
        if comm == T_FEEDBACK:
            mid = (m.arbitration_id >> 8) & 0xFF
            p, v, tq, temp = struct.unpack(">HHHH", bytes(m.data))
            if mid not in MOTOR_MODEL:
                return comm, mid, bytes(m.data)
            v_max, t_max, _, _ = ranges(mid)
            self.last_fb[mid] = Feedback(u2f(p, -P_MAX, P_MAX), u2f(v, -v_max, v_max), u2f(tq, -t_max, t_max),
                                         temp / 10.0, (m.arbitration_id >> 16) & 0x3F,
                                         (m.arbitration_id >> 22) & 0x3, time.monotonic())
        return comm, (m.arbitration_id >> 8) & 0xFF, bytes(m.data)

    def drain(self, timeout: float = 0.0) -> None:
        end = time.monotonic() + timeout
        while True:
            self._handle(self.bus.recv(max(0.0, end - time.monotonic())))
            if time.monotonic() >= end:
                break

    def read_param_f32(self, mid: int, idx: int, timeout: float = 0.05) -> float | None:
        self._send(T_READ, mid, struct.pack("<HHf", idx, 0, 0.0))
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            r = self._handle(self.bus.recv(end - time.monotonic()))
            if r and r[0] == T_READ and r[1] == mid:
                i, _, v = struct.unpack("<HHf", r[2])
                if i == idx:
                    return v
        return None

    def read_param_u8(self, mid: int, idx: int, timeout: float = 0.05) -> int | None:
        self._send(T_READ, mid, struct.pack("<HHf", idx, 0, 0.0))
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            r = self._handle(self.bus.recv(end - time.monotonic()))
            if r and r[0] == T_READ and r[1] == mid and struct.unpack("<H", r[2][:2])[0] == idx:
                return r[2][4]
        return None

    def write_param_u8(self, mid: int, idx: int, value: int) -> None:
        self._send(T_WRITE, mid, struct.pack("<HHB3x", idx, 0, value))

    def write_param_f32(self, mid: int, idx: int, value: float) -> None:
        """RAM write (type 18): lost at power-off; nothing is saved to flash."""
        self._send(T_WRITE, mid, struct.pack("<HHf", idx, 0, float(value)))

    def enable(self, mid: int) -> None:
        self._send(T_ENABLE, mid, bytes(8))

    def stop(self, mid: int) -> None:
        self._send(T_STOP, mid, bytes(8))     # data[0] = 0: disable without clearing faults

    def mit(self, mid: int, pos: float, kp: float, kd: float, vel: float = 0.0, tau: float = 0.0) -> None:
        v_max, t_max, kp_max, kd_max = ranges(mid)
        data = struct.pack(">HHHH", f2u(pos, -P_MAX, P_MAX), f2u(vel, -v_max, v_max),
                           f2u(kp, 0, kp_max), f2u(kd, 0, kd_max))
        self._send(T_MIT, mid, data, mid_field=f2u(tau, -t_max, t_max))


def read_joint_states(rs: RobStrideBus) -> dict[int, dict]:
    out = {}
    for mid in (*ARM_IDS, GRIPPER_ID):
        out[mid] = {"pos": rs.read_param_f32(mid, P_MECHPOS), "vel": rs.read_param_f32(mid, P_MECHVEL),
                    "iq": rs.read_param_f32(mid, P_IQF), "vbus": rs.read_param_f32(mid, P_VBUS)}
    return out


def print_states(states: dict[int, dict]) -> None:
    fmt = lambda v, f: "   ----" if v is None else format(v, f)
    print(f"{'joint':8s} {'id':>3s} {'pos rad':>9s} {'pos deg':>9s} {'vel rad/s':>10s} {'iq A':>7s} {'vbus V':>7s}")
    for mid, s in states.items():
        deg = None if s["pos"] is None else math.degrees(s["pos"])
        print(f"{JOINT_NAMES[mid]:8s} {mid:3d} {fmt(s['pos'], '9.4f')} {fmt(deg, '9.2f')} "
              f"{fmt(s['vel'], '10.4f')} {fmt(s['iq'], '7.3f')} {fmt(s['vbus'], '7.2f')}")
