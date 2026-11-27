"""Actuator system identification for the reBot B601-RS (PACE-style, Bjelonic et al., arXiv:2509.06342).

Module 1 (collect): drive all six joints with chirp position targets through the motors' own MIT PD
loop, log commands and feedback. Module 2 (fit): replay the logged commands in MuJoCo and fit per-joint
armature, viscous damping, Coulomb friction and position bias plus one command delay with CMA-ES.

Entry points: scripts/sysid/collect.py and scripts/sysid/fit.py. Guide: docs/sysid.md.
"""
