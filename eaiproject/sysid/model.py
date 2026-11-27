"""MuJoCo arm model for identification: MIT PD actuators, gravity feed-forward, identified parameters.

The real motors run in MIT mode:   tau = sat( kp (q_cmd - q) + kd (qd_cmd - qd) + tau_ff )
In MuJoCo every arm actuator is turned into an affine actuator with the same law:
    force = kp * ctrl - kp * q - kd * qd,   ctrl = q_cmd + bias + (tau_ff + kd * qd_cmd) / kp
clamped to +-torque_cap (forcerange), which is the motors' limit_torque during the run.
`bias` is the PACE joint bias q_b: sat(kp (q_cmd - q + q_b) - kd qd ...).

Identified per joint: armature (dof_armature), viscous damping (dof_damping), Coulomb friction
(dof_frictionloss), bias; plus one global command delay. Everything else (masses, PD gains, torque
limits) is known and fixed, as in the paper.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SYSID_SCENE = ROOT / "assets/rebot_b601_rs/mjcf/sysid_scene.xml"

N = 6
ARM_JOINTS = [f"joint{i}" for i in range(1, N + 1)]
MOTOR_IDS = tuple(range(1, N + 1))
KP_DEFAULT = np.array([80.0, 100.0, 100.0, 35.0, 25.0, 18.0])   # Seeed rebotarm_mujoco_rs gains, used by our sim
KD_DEFAULT = np.array([8.0, 10.0, 10.0, 4.0, 3.0, 2.5])
MOTOR_TORQUE_MAX = np.array([36.0, 36.0, 36.0, 14.0, 14.0, 14.0])  # RS06 x3, RS00 x3 (firmware limit_torque)
TORQUE_CAP_DEFAULT = np.array([20.0, 20.0, 20.0, 7.0, 7.0, 7.0])    # written to limit_torque (RAM) for the run

# Search box per parameter (RS06 joints 1-3, RS00 joints 4-6). Normalised to [-1, 1] for CMA-ES.
BOUNDS = {
    "armature":     (np.zeros(N), np.array([0.20, 0.20, 0.20, 0.05, 0.05, 0.05])),     # kg m^2
    "damping":      (np.zeros(N), np.array([5.0, 5.0, 5.0, 3.0, 3.0, 3.0])),           # N m s / rad
    "frictionloss": (np.zeros(N), np.array([3.0, 3.0, 3.0, 1.0, 1.0, 1.0])),           # N m
    "bias":         (np.full(N, -0.05), np.full(N, 0.05)),                              # rad
}
DELAY_BOUNDS = (0.0, 0.05)                                                               # s
PER_JOINT = ("armature", "damping", "frictionloss", "bias")


@dataclass
class SimParams:
    armature: np.ndarray
    damping: np.ndarray
    frictionloss: np.ndarray
    bias: np.ndarray
    delay: float = 0.0
    info: dict = field(default_factory=dict)

    # ---- vector <-> params (fixed order: 4 x 6 per-joint values, then delay) ----
    def to_vector(self) -> np.ndarray:
        return np.concatenate([getattr(self, k) for k in PER_JOINT] + [[self.delay]])

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "SimParams":
        v = np.asarray(v, float)
        return cls(*(v[i * N:(i + 1) * N].copy() for i in range(len(PER_JOINT))), delay=float(v[-1]))

    @staticmethod
    def bounds() -> tuple[np.ndarray, np.ndarray]:
        lo = np.concatenate([BOUNDS[k][0] for k in PER_JOINT] + [[DELAY_BOUNDS[0]]])
        hi = np.concatenate([BOUNDS[k][1] for k in PER_JOINT] + [[DELAY_BOUNDS[1]]])
        return lo, hi

    @classmethod
    def from_normalized(cls, x: np.ndarray) -> "SimParams":
        lo, hi = cls.bounds()
        return cls.from_vector(lo + (np.clip(x, -1, 1) + 1) / 2 * (hi - lo))

    @classmethod
    def nominal(cls, m: mujoco.MjModel) -> "SimParams":
        """What the MJCF says now (bias 0, no delay)."""
        dadr = [m.joint(j).dofadr[0] for j in ARM_JOINTS]
        return cls(m.dof_armature[dadr].copy(), m.dof_damping[dadr].copy(), m.dof_frictionloss[dadr].copy(),
                   np.zeros(N), 0.0)

    # ---- artifact I/O ----
    def to_json(self) -> dict:
        return {
            "units": {"armature": "kg m^2", "damping": "N m s/rad", "frictionloss": "N m", "bias": "rad",
                      "delay": "s"},
            "joints": {j: {k: float(getattr(self, k)[i]) for k in PER_JOINT} for i, j in enumerate(ARM_JOINTS)},
            "delay": float(self.delay),
            **({"info": self.info} if self.info else {}),
        }

    @classmethod
    def from_json(cls, d: dict) -> "SimParams":
        g = lambda k: np.array([d["joints"][j][k] for j in ARM_JOINTS], float)
        return cls(g("armature"), g("damping"), g("frictionloss"), g("bias"), float(d["delay"]), d.get("info", {}))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "SimParams":
        return cls.from_json(json.loads(Path(path).read_text()))

    def apply_to(self, m: mujoco.MjModel) -> None:
        """Write armature / damping / frictionloss into any model that has joints joint1..joint6
        (e.g. BallInterceptEnv.m). Bias and delay belong to the controller, not the model."""
        dadr = [m.joint(j).dofadr[0] for j in ARM_JOINTS]
        m.dof_armature[dadr] = self.armature
        m.dof_damping[dadr] = self.damping
        m.dof_frictionloss[dadr] = self.frictionloss


def dynamics_model(xml: str | Path = SYSID_SCENE) -> mujoco.MjModel:
    """The arm for rollouts: gripper joints removed (the real jaw is closed and unpowered, so it is a
    rigid part of link 6) and all mesh geoms dropped. Masses and inertias are explicit <inertial>
    elements, so the dynamics are unchanged, but the model is ~300 MB smaller per copy (one copy per
    CMA-ES candidate) and steps ~2x faster."""
    spec = mujoco.MjSpec.from_file(str(xml))
    for e in list(spec.equalities):
        spec.delete(e)
    for a in list(spec.actuators):
        if a.name == "joint7_motor":
            spec.delete(a)
    for n in ("joint7", "joint_left", "joint_right"):
        spec.delete(spec.joint(n))
    for g in list(spec.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            spec.delete(g)
    for me in list(spec.meshes):
        spec.delete(me)
    return spec.compile()


class ArmModel:
    """Light dynamics model (rollouts, gravity) + full model (collision check of the plan)."""

    def __init__(self, xml: str | Path = SYSID_SCENE):
        self.m = dynamics_model(xml)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.jid = np.array([m.joint(j).id for j in ARM_JOINTS])
        self.qadr = np.array([m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.dadr = np.array([m.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.act = np.array([m.actuator(f"{j}_motor").id for j in ARM_JOINTS])
        self.range = m.jnt_range[self.jid].copy()
        self.nominal = SimParams.nominal(m)
        self.dt = m.opt.timestep
        self.xml = Path(xml)
        self._mc = None                                     # full model, loaded on first collision check

    def _collision_model(self):
        if self._mc is None:
            mc = mujoco.MjModel.from_xml_path(str(self.xml))
            self._mc = (mc, mujoco.MjData(mc), np.array([mc.joint(j).qposadr[0] for j in ARM_JOINTS]))
        return self._mc

    # ---- kinematics / gravity (nominal model) ----
    def set_q(self, d: mujoco.MjData, q: np.ndarray) -> None:
        d.qpos[:] = self.m.qpos0
        d.qpos[self.qadr] = q
        d.qvel[:] = 0

    def gravity(self, q: np.ndarray) -> np.ndarray:
        """Gravity torque on joints 1-6 at pose q (gripper closed). This is the MIT tau_ff."""
        self.set_q(self.d, q)
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        res = np.zeros(self.m.nv)
        mujoco.mj_rne(self.m, self.d, 0, res)
        return res[self.dadr].copy()

    # ---- dynamics model for rollouts ----
    def make_dynamics_model(self, kp, kd, torque_cap, params: SimParams | None = None,
                            contacts: bool = False) -> mujoco.MjModel:
        m = copy.copy(self.m)
        if not contacts:
            m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        for i, a in enumerate(self.act):
            set_pd_actuator(m, a, kp[i], kd[i], torque_cap[i])
        (params or self.nominal).apply_to(m)
        return m

    def initial_state(self, m: mujoco.MjModel, q0: np.ndarray, qd0: np.ndarray | None = None) -> np.ndarray:
        d = mujoco.MjData(m)
        self.set_q(d, q0)
        if qd0 is not None:
            d.qvel[self.dadr] = qd0
        mujoco.mj_forward(m, d)
        spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        s = np.empty(mujoco.mj_stateSize(m, spec))
        mujoco.mj_getState(m, d, s, spec)
        return s

    def contacts(self, q: np.ndarray, depth: float = 0.002) -> set[tuple[str, str]]:
        """Penetrating contact pairs (body names) at pose q in the full model (jaw closed)."""
        mc, d, qadr = self._collision_model()
        d.qpos[:] = mc.qpos0
        d.qpos[qadr] = q
        mujoco.mj_forward(mc, d)
        out = set()
        for c in d.contact[: d.ncon]:
            if c.dist < -depth:
                a, b = (mc.body(mc.geom_bodyid[g]).name for g in (c.geom1, c.geom2))
                out.add(tuple(sorted((a, b))))
        return out


def set_pd_actuator(m: mujoco.MjModel, a: int, kp: float, kd: float, cap: float) -> None:
    m.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
    m.actuator_gainprm[a, :] = 0
    m.actuator_gainprm[a, 0] = kp
    m.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
    m.actuator_biasprm[a, :] = 0
    m.actuator_biasprm[a, 1] = -kp
    m.actuator_biasprm[a, 2] = -kd
    m.actuator_ctrllimited[a] = 0
    m.actuator_forcelimited[a] = 1
    m.actuator_forcerange[a] = (-cap, cap)


def mit_ctrl(q_cmd, qd_cmd, tau_ff, kp, kd, bias) -> np.ndarray:
    """MIT command -> ctrl of the affine actuator (see module docstring)."""
    return q_cmd + bias + (tau_ff + kd * qd_cmd) / kp
