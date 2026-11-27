# Sim-to-real system identification: how to use it

These scripts make the MuJoCo arm respond like the real reBot B601-RS. They follow the first part of
PACE ([arXiv:2509.06342](https://arxiv.org/abs/2509.06342)).

The workflow has two steps:

1. **`collect.py`** runs a fixed excitation sequence on the arm and records the response.
2. **`fit.py`** finds simulator parameters that reproduce that response. The output is
   **`sim_params.json`**.

Background, the safety model and troubleshooting are in [docs/sysid.md](../../docs/sysid.md).
A one-minute walkthrough on simulated data: [docs/media/sysid_demo.mp4](../../docs/media/sysid_demo.mp4).

| script | needs the arm | what it does | time |
|---|---|---|---|
| `collect.py --sim` | no | whole run on a MuJoCo stand-in arm with hidden parameters | ~40 s |
| `fake_can_test.py` | no | the **real** CAN code path against emulated motors | ~60 s |
| `collect.py --dry-run` | yes, unpowered | preflight reads + plan check, nothing enabled | ~10 s |
| `collect.py` | yes, powered | the real data collection | ~4 min |
| `fit.py <run> [<run> ...]` | no | CMA-ES fit, writes `sim_params.json` | 15-40 min |

On the rig the same two are `./rig sysid [options]` and `./rig sysid-fit <run> [<run> ...]`.

## 0. Install

```bash
.venv/bin/pip install -r requirements.txt     # adds cmaes and matplotlib
```

## 1. Rehearse at home (no hardware)

```bash
# the full sequence on a simulated arm (it has its own hidden "true" parameters)
.venv/bin/python scripts/sysid/collect.py --sim

# rehearse the safety paths
.venv/bin/python scripts/sysid/collect.py --sim --inject track   # bump on joint 2 -> slow return, then disable
.venv/bin/python scripts/sysid/collect.py --sim --inject lost    # joint 3 goes silent -> immediate disable

# the hardware code path (CAN frames, parameter writes/restore, 200 Hz loop) against fake motors
.venv/bin/python scripts/sysid/fake_can_test.py
```

`fake_can_test.py` should end with:

- `motors disabled`
- `motor parameters restored: YES`
- a loop rate of about 200 Hz

With the default short plan, one quality line FAILs (`at least one train and one val segment`).
That is expected. To run the whole plan against the fake motors:
`fake_can_test.py -- --only chirp_A_large chirp_A_small multisine_A chirp_B`.

The fake motors only follow their targets with a lag and have no gravity. Use them to test the
plumbing, not to collect data.

## 2. In the lab

Before you start:

- The arm sits in its **rest pose**, jaw closed, with about 0.7 m clear around it.
- The 48 V supply is on and the PCAN-USB adapter is plugged in.
- One person stays at the power switch.

```bash
.venv/bin/python scripts/sysid/collect.py --dry-run    # must say: preflight OK ... nothing was enabled
.venv/bin/python scripts/sysid/collect.py              # type 'yes'; at the end press Enter to disable
```

- **Joint offsets come from the rig's calibration.** Do today's arm step first
  (`./rig calibrate`, step 4: the wrist zero changes when the gripper is remounted). The run then
  takes its offsets from `outputs/calibration/arm.json` and the table map, and prints them:
  the wrist zero for joint 6, the base's turn on the table for joint 1 (the model has its table in
  front of joint 1's zero, ours is turned 90° to the right), and the turn each motor counts (joint 1
  reads 267° at rest). On 27 Nov 2026 that gave 270° and 212.8°, the values typed by hand that day.
  `--joint-offset J=DEG` still sets one by hand. The offsets are applied to every read and command
  and saved in `meta.json`.
- **Switch the arm's 48 V off and on before the run** if a position-mode task (`./rig move`) ran
  since power-on: its speed and current limits stay in the motors' RAM.
- **Plug the PCAN-USB straight into the laptop** if you can. Behind a USB-C hub it kept
  disconnecting on one laptop; on our Mac the hub worked (27 Nov 2026).
- **CAN backend.** It is chosen automatically: `pcan` on macOS, `socketcan` on Linux. On Linux,
  bring the link up once per boot: `sudo ip link set can0 up type can bitrate 1000000`.
- **Stopping.** `Ctrl+C` stops safely: the arm moves slowly back to rest, then the motors are
  disabled. A second `Ctrl+C`, or a fault or a lost motor, disables everything at once. **Joints 2-3
  then fall, so support the arm.**
- **Result.** The run ends with a data-quality verdict. **`DATA OK`** means you can leave. Copy
  `outputs/sysid/<run>/` off the laptop (`outputs/` is git-ignored).
- **If the plan check or the run fails,** the message says why. The usual fix is a smaller
  amplitude, `--scale 0.6`. See the troubleshooting table in [docs/sysid.md](../../docs/sysid.md).

Useful options: `--scale` (amplitude), `--speed` (joint speed and acceleration the excitation
reaches, up to 2x), `--fmax` (highest chirp frequency, default 4 Hz), `--rate` (default 200 Hz),
`--only <segments>`, `--torque-cap` / `--no-torque-cap`.

**More data for a better fit.** One run excites the joints up to 1.0 / 0.8 / 0.8 / 1.5 / 1.5 /
2.0 rad/s, while the catch moves them at up to 3 rad/s, and the first real fit described large fast
motion better than small slow motion. Record several runs and fit them together:

```bash
./rig sysid                          # the standard run
./rig sysid --scale 0.6              # smaller motion
./rig sysid --speed 2                # twice the joint speed; if the plan check asks for more torque:
./rig sysid --speed 2 --torque-cap 28 28 28 10 10 10
./rig sysid-fit outputs/sysid/<run1> outputs/sysid/<run2> outputs/sysid/<run3>
```

Runs under different torque caps can be fitted together (each is replayed under its own cap); runs
under different PD gains cannot.

## 3. Fit the parameters (any machine)

```bash
.venv/bin/python scripts/sysid/fit.py outputs/sysid/<run>              # full fit, all CPU cores
.venv/bin/python scripts/sysid/fit.py outputs/sysid/<run> --gens 120   # good first result in ~15 min
.venv/bin/python scripts/sysid/fit.py outputs/sysid/<run1> outputs/sysid/<run2>   # several runs in one fit
.venv/bin/python scripts/sysid/fit.py outputs/sysid/<run> --eval <other>/sim_params.json   # score given params
```

The fit writes these files to `outputs/sysid/<run>/fit_<time>/`:

- `sim_params.json`: **the artifact**. Per joint it holds armature, damping, frictionloss and bias,
  plus one `delay`.
- `fit_report.txt`: RMS joint error, nominal vs fitted, on the train windows and on the held-out
  `multisine_A`. It also has the parameter table. For a `--sim` run it adds the hidden true values.
- `fit_<segment>.png`: real vs simulated joints. `fit_loss.png`: the loss curve.

Judge the fit by the **val** row of `fit_report.txt`, not the train rows. The fit is good when the
val error is far below the nominal one.

## 4. Use the result in the simulation

```python
from eaiproject.sysid.model import SimParams

p = SimParams.load("outputs/sysid/<run>/fit_<time>/sim_params.json")
p.apply_to(env.m)          # any MjModel with joint1..joint6, e.g. BallInterceptEnv.m
# p.delay (s) and p.bias (rad) belong to the controller: delay the joint targets by p.delay
# and add p.bias to them to reproduce the real arm's response.
```

The parameters describe the arm under the PD gains used during collection (`kp`/`kd` in
`sim_params.json` → `info`). These are the gains of the real catch (`./rig catch`, MIT control):
one definition, `MIT_KP` / `MIT_KD` in `eaiproject/arm.py`. If you change them, collect again.

## Files

```
scripts/sysid/collect.py        module 1: data collection (real arm, --dry-run, --sim)
scripts/sysid/fit.py            module 2: CMA-ES fit -> sim_params.json
scripts/sysid/fake_can_test.py  emulated RobStride motors on a virtual CAN bus
eaiproject/sysid/               excitation plan, arm I/O, safety runner, MuJoCo replay, fit, reports
assets/rebot_b601_rs/mjcf/sysid_scene.xml   arm + table scene used by the fit
docs/sysid.md                   method, safety, troubleshooting
```
