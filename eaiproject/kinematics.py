"""Forward kinematics of the real arm from joint readings, using the MuJoCo model."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parents[1] / "assets/rebot_b601_rs/mjcf/ball_intercept_scene.xml"


class Fingertip:
    """Fingertip point (gripper_end origin, as in BallInterceptEnv) for arm joint angles q (rad, joints 1-6).

    Takes MODEL joint angles: eaiproject.arm adds the zero corrections of config/arm.json to the
    motor readings. After a call, `self.d.xmat[self.body]` holds the gripper orientation.
    """

    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.d = mujoco.MjData(self.m)
        self.adr = [self.m.joint(f"joint{i}").qposadr[0] for i in range(1, 7)]
        self.body = self.m.body("gripper_end").id

    def __call__(self, q) -> np.ndarray:
        self.d.qpos[:] = self.m.qpos0
        for a, v in zip(self.adr, q):
            self.d.qpos[a] = v
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.xpos[self.body].copy()
