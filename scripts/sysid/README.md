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
| `fit.py <run>` | no | CMA-ES fit, writes `sim_params.json` | 15-40 min |

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

- **Our arm: joint 6 zero offset.** In the zero pose the joint-6 motor reads +333.64 deg (checked
  25 Nov 2026), so add `--joint-offset 6=333.64` to both commands. The offset is applied to every
  read and command and saved in `meta.json`.
- **Plug the PCAN-USB straight into the laptop.** Behind a USB-C hub it kept disconnecting.
- **CAN backend.** It is chosen automatically: `pcan` on macOS, `socketcan` on Linux. On Linux,
  bring the link up once per boot: `sudo ip link set can0 up type can bitrate 1000000`.
- **Stopping.** `Ctrl+C` stops safely: the arm moves slowly back to rest, then the motors are
  disabled. A second `Ctrl+C`, or a fault or a lost motor, disables everything at once. **Joints 2-3
  then fall, so support the arm.**
- **Result.** The run ends with a data-quality verdict. **`DATA OK`** means you can leave. Copy
  `outputs/sysid/<run>/` off the laptop (`outputs/` is git-ignored).
- **If the plan check or the run fails,** the message says why. The usual fix is a smaller
  amplitude, `--scale 0.6`. See the troubleshooting table in [docs/sysid.md](../../docs/sysid.md).

Useful options: `--scale` (amplitude), `--fmax` (highest chirp frequency, default 4 Hz), `--rate`
(default 200 Hz), `--only <segments>`, `--torque-cap` / `--no-torque-cap`.

## 3. Fit the parameters (any machine)

```bash
.venv/bin/python scripts/sysid/fit.py outputs/sysid/<run>              # full fit, all CPU cores
.venv/bin/python scripts/sysid/fit.py outputs/sysid/<run> --gens 120   # good first result in ~15 min
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
`sim_params.json` → `info`). If you change the controller gains, collect again.

## Files

```
scripts/sysid/collect.py        module 1: data collection (real arm, --dry-run, --sim)
scripts/sysid/fit.py            module 2: CMA-ES fit -> sim_params.json
scripts/sysid/fake_can_test.py  emulated RobStride motors on a virtual CAN bus
eaiproject/sysid/               excitation plan, arm I/O, safety runner, MuJoCo replay, fit, reports
assets/rebot_b601_rs/mjcf/sysid_scene.xml   arm + table scene used by the fit
docs/sysid.md                   method, safety, troubleshooting
```
