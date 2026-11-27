# Actuator system identification (sim-to-real)

Make the MuJoCo arm respond like the real one. We follow the first part of PACE
(Bjelonic et al., *Towards bridging the gap: Systematic sim-to-real transfer for diverse legged
robots*, [arXiv:2509.06342](https://arxiv.org/abs/2509.06342)):

1. **Collect** (`scripts/sysid/collect.py`). All six arm joints follow chirp position targets at the
   same time, tracked by the motors' own MIT PD loop (our sim gains, gravity feed-forward from the
   model). Commands and feedback are logged at 200 Hz.
2. **Fit** (`scripts/sysid/fit.py`). The logged commands are replayed in MuJoCo, and CMA-ES fits
   25 parameters: armature, viscous damping, Coulomb friction and position bias for each joint, plus
   one command delay. The objective is the mean squared joint-position error. A held-out multisine
   is used only for validation.

The artifact is `sim_params.json`. Apply it to any model with
`SimParams.load(path).apply_to(model)` (`eaiproject/sysid/model.py`).

## At home: rehearse with `--sim` (no hardware)

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/sysid/collect.py --sim                  # full run on a MuJoCo stand-in, ~40 s
.venv/bin/python scripts/sysid/collect.py --sim --inject track   # bump on joint 2 -> slow return, then disable
.venv/bin/python scripts/sysid/collect.py --sim --inject lost    # joint 3 stops answering -> hard abort
.venv/bin/python scripts/sysid/fit.py outputs/sysid/<run>_sim    # should recover the hidden parameters
```

The stand-in arm has hidden "true" parameters, 3 % heavier links than the model, sensor quantisation
and noise, and timing jitter. The fit report prints the true values next to the fitted ones.

## In the lab: the short version

| step | what | time |
|---|---|---|
| 0 | Arm in its **rest pose**, jaw closed, 0.7 m clear around it. 48 V on, PCAN-USB plugged in. One person at the power switch. | 5 min |
| 1 | `./rig sysid --dry-run` (= `.venv/bin/python scripts/sysid/collect.py --dry-run`): reads everything and checks the plan in the model. Nothing is powered. | 10 s |
| 2 | `./rig sysid`, then type `yes`. | ~4 min |
| 3 | The script ends with `DATA OK`. Copy `outputs/sysid/<run>/` somewhere safe. | |
| 4 | `./rig sysid-fit outputs/sysid/<run>` (any laptop, no hardware) | 15-40 min |

The joint offsets (wrist zero, the base's turn on the table, the turn each motor counts) come from
the rig's calibration: do `./rig calibrate` step 4 first on the day. See
[scripts/sysid/README.md](../scripts/sysid/README.md).

The CAN backend is chosen automatically: `pcan` on macOS (see [real_rig.md](real_rig.md)) and
`socketcan` on Linux. On Linux, bring the link up once per boot with
`sudo ip link set can0 up type can bitrate 1000000`. Use `--can` / `--channel` to override.

### What the run does

1. **Preflight, unpowered.** All seven motors answer, there are no faults, the bus is at about 48 V,
   the arm is still and not already enabled, each feedback frame matches its mechPos reading, and
   every position is inside the model range.
2. **Plan check in MuJoCo.** Every target stays inside the joint limits. The model predicts no new
   contacts (contacts already present in the rest pose are ignored). Predicted torque stays below
   70 % of the cap. The script prints a per-joint table of the predicted motion.
3. **Motor setup, still disabled.** MIT run mode, and `limit_torque` set to 20 Nm on joints 1-3 and
   7 Nm on joints 4-6. These are RAM writes, restored at the end.
4. **Enable.** The motors are enabled holding the rest pose, and the gravity feed-forward ramps in
   over 2 s. A steady-state check follows: an error over 4.6° means the feed-forward is wrong, so
   the arm returns and stops.
5. **Pose A.** A full-amplitude chirp, a 35 % chirp, a held-out multisine (val), and `steps_A`:
   25 s of catch-like motion (fast straight moves between random poses near pose A at up to
   2.2 rad/s, short stops, targets changing 60 times a second as in `./rig catch`). After each
   segment the script prints rate, missing feedback, RMS error and peak torque/cap, and saves the log.
6. **Pose B.** One chirp at 70 % amplitude.
7. **Back to rest.** The feed-forward ramps down, then the script waits for **Enter** while still
   holding the arm, and disables the motors.

Chirps sweep 0.1 → 4 Hz (`--fmax`) over 30 s. The amplitude is limited by joint speed and
acceleration, so it shrinks at high frequency (about 1° at 4 Hz for joints 2-3).

### Safety

- **Soft abort:** tracking or deviation error, torque at the cap, temperature, joint range, speed,
  or `Ctrl+C`. The motors stay powered, the arm moves slowly back to the rest pose, the feed-forward
  ramps down, and then the motors are disabled.
- **Hard abort:** motor fault, a joint silent for 50 ms, `Ctrl+C` during the return, or an error
  during the return. All motors are disabled at once. **Joints 2-3 then fall: support the arm.**
- During excitation, the tracking check compares the measured motion with the nominal model's
  predicted motion, not with the target, because the PD lag at high frequency is expected. The
  allowed deviation is printed before the run.
- Segments that finished before an abort stay in the log. `fit.py --allow-partial` also uses the
  segment cut short.

### If something goes wrong

| message | cause / fix |
|---|---|
| `No arm motor answered` | 48 V off or CAN cable not in the arm |
| `Cannot open the CAN adapter` | adapter unplugged; macOS: libPCBUSB missing; Linux: `can0` not up |
| `joint N reads ... outside the model range` | the arm is not in its rest pose, or the wrist zero is not today's: `./rig calibrate --step arm`; `./rig arm` shows the joints |
| `joint N draws ... A` | another script still has the motor enabled; close it |
| `limit_torque reads ..., not a plausible torque limit` | unexpected firmware parameter map; rerun with `--no-torque-cap` only if you are sure |
| `PLAN CHECK FAILED: predicted torque` | lower `--scale` (e.g. 0.7) or `--fmax` |
| soft abort `steady error ... gravity feed-forward wrong` | model mass or sign mismatch; check `joint_check.py`, run with `--scale 0.5` |
| soft abort `deviation from the predicted motion` | real arm much slower or faster than the model; rerun with `--scale 0.6` |
| quality `loop rate` FAIL | the laptop is too slow or busy; close other programs, or use `--rate 150` |

## Fit details

- Model: `assets/rebot_b601_rs/mjcf/sysid_scene.xml`. For rollouts the jaw is rigid (the gripper
  joints are removed) and the meshes are dropped. Masses and inertias are unchanged.
- MIT law in MuJoCo: an affine actuator, `force = kp (ctrl - q) - kd qd`, with
  `ctrl = q_cmd + bias + tau_ff / kp`, clamped to the run's torque cap and delayed by `delay`.
- Search box (`model.BOUNDS`): armature 0-0.6 / 0-0.15 kg m², damping 0-5 / 0-3 N m s/rad,
  friction 0-3 / 0-2 N m (RS06 / RS00), bias ±0.05 rad, delay 0-100 ms. The report flags any
  parameter that ends at a box edge; widen the box if one does (done once, after the first real
  fit of 27 Nov 2026 ended at the edge for every armature, the delay and joint 6's friction).
- Several runs can be fitted together (`fit.py <run1> <run2> ...`): windows are named
  `<segment>_<run>`, each run is replayed under its own torque cap, and the output goes to the
  first run's folder.
- Output in `<run>/fit_<time>/`:
  - `sim_params.json`, the artifact
  - `fit_report.txt`, RMS error per joint (nominal vs fitted, train and val) and a parameter table
  - `fit_<segment>.png`, real vs simulated joint positions
  - `fit_loss.png`, the loss curve
- Runtime: about 1 s per generation per training minute on 12 threads. `--gens 60` gives a quick
  first look.
