# reBot ball interception

A 6-DoF robot arm that blocks a ball rolled toward it down a table. A fixed camera tracks the ball,
the arm predicts where it will cross an intercept line, and moves its open gripper there so the ball
rolls into the jaw.

The project runs both in **MuJoCo simulation** and on the **real hardware**: a Seeed Studio
[reBot Arm B601-RS](https://github.com/Seeed-Projects/reBot-DevArm) (RobStride motors over CAN) and an
Intel RealSense D435i.

<p align="center">
  <img src="docs/media/sim_demo_forward.gif" alt="Simulated arm intercepting a rolling ball" width="720">
  <br><sub>Simulation: overview, close-up at the intercept line, top view (<a href="docs/media/sim_demo_forward.mp4">mp4</a>)</sub>
</p>

## Status

| part | state |
|---|---|
| Simulation: scripted interception from ground truth | ✅ 100/100 balls caught |
| Simulation: vision in the loop (camera model + classic CV tracker) | ✅ 100/100, 2-3 mm error, robust to 30 fps / 100 ms latency |
| Real arm: CAN link, joint readout, model match | ✅ |
| Real camera: 60 fps stream, located from table markers mapped in the arm's frame | ✅ 0.6 px, touches 1.7 mm |
| Real tracking in shadow mode (arm unpowered) | 🔧 ready, being tested |
| Powered: jaw along the intercept line, interception (all joints, motor position mode) | 🔧 ready for the first runs |

## Repository layout

```
eaiproject/                     Python package
  ball_intercept_env.py         MuJoCo env, scripted InterceptPolicy, IK
  ball_tracker.py               camera calibration model + HSV/Kalman ball tracker
  tracked_intercept.py          sim camera model + tracker-driven policy
  real_intercept.py             real rig: camera pose from the table map, tracker setup, real-time planner
  rig_calib.py                  table map: markers and table plane in the arm's frame; camera relocalisation
  rig_report.py, rig_draw.py    calibration report and drawing helpers
  arm.py                        real arm: wrist zero, rest / initial poses, position-mode control
  robstride.py                  RobStride CAN protocol over PEAK PCAN-USB (with a motor allow-list)
  realsense_client.py           client for the RealSense frame server
  kinematics.py, markers.py     FK from joint readings, ArUco table markers
scripts/
  sim/                          render_demo, eval_scripted, eval_tracked, live_tracking
  real/                         the ./rig tasks (calibrate, check, track, move, catch, ...), camera server
config/rig_map.json             the table map (written by ./rig calibrate)
rig                             launcher for the real rig: ./rig help
assets/rebot_b601_rs/           arm model (MJCF/URDF/meshes) from Seeed + our scene XML
docs/
  simulation.md                 sim details, env API, results
  real_rig.md                   hardware setup, calibration, powered runs, troubleshooting
```

## Quick start (simulation)

Python 3.12. The simulation needs no hardware.

```bash
git clone https://github.com/AbdullahSohail15/rebot-ball-intercept.git
cd rebot-ball-intercept
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/sim/eval_scripted.py --n 20               # scripted policy, random launches
.venv/bin/python scripts/sim/render_demo.py --mode forward         # -> outputs/demo_ball_intercept_forward.mp4
.venv/bin/python scripts/sim/live_tracking.py --window             # camera + tracker in the loop, live
```

On a headless Linux machine, prefix rendering commands with `MUJOCO_GL=egl`. More in
[docs/simulation.md](docs/simulation.md).

## Real rig

Everything runs through `./rig`. The full guide, including the macOS-specific camera and CAN setup,
is in [docs/real_rig.md](docs/real_rig.md).

```bash
./rig status             # camera server, CAN adapter, table map, arm poses, ball colour
./rig arm                # joint states, read only
./rig view               # the camera with the table markers, for aiming it
./rig calibrate          # the table map once (markers + fingertip touches + sweep); daily: wrist zero, ball colour
./rig check              # live check of the calibration
./rig track              # ball tracking and the planned motion, arm NOT powered
./rig move               # POWERED: the jaw along the intercept line at 60 / 35 / 20 mm
./rig catch              # POWERED: the jaw follows the ball
```

The table map is committed (`config/rig_map.json`). Daily values, logs and camera frames are
written to `outputs/`, which is git-ignored.

### A new day with a new marker layout

About 40 minutes from bare table to catching. Before you start: arm power on (the motors stay
off), the PCAN-USB adapter plugged in, the camera on a USB 3 port of the Mac, the gripper mounted.

1. **Markers.** Tape the 4 ArUco markers flat on the table: at least 2 within the arm's reach (they
   get touched), at least 1 more than 0.5 m from the arm base, not in a line, spread out. Keep a
   white margin of 1 cm or more and no tape on the black border.
2. **Camera.** `./rig camera start` (macOS asks for your password), then `./rig view`: every marker
   should be outlined. Close the window with `q`.
3. **Table map**, `./rig calibrate --new` (about 20 min, one window, everything is saved as you go):
   - *1 Views:* arm and hands off the markers, camera still, SPACE. Repeat from 6-8 camera positions:
     left, right, front, low and high, near and far. Then put the camera where it will stay. ENTER.
   - *2 Touch:* support the unpowered arm, rest the **closed** fingertips on the corner circled in
     yellow, SPACE. All 4 corners of every marker you can reach (`n` next marker). ENTER.
   - *3 Sweep:* hold SPACE and slide the closed fingertips slowly over the table until the yellow
     box is 70 % covered. ENTER solves.
   - *Result:* accept reprojection < 1.5 px, touches < 5 mm, sweep < 1.5 mm and no warnings; each
     warning says what to redo (`1`-`3` jump back). ENTER.
4. **Arm and ball** (same window, steps 4 and 5, 2 min):
   - *4 Arm:* open the jaw about 3 cm by hand, both fingertips on the table, fingers tilted forward,
     SPACE, then again at another spot (this is the wrist zero). `n` → **rest**: fold the arm onto
     itself, let go, SPACE. `n` → **initial**: hold the arm above the table, clear of the markers,
     SPACE. ENTER.
   - *5 Ball:* the ball alone on the table, hands away, `a`. Roll it: the ring must follow it and the
     mask must show only the ball. ENTER saves and closes.
5. **Commit the map:** `git add config/rig_map.json && git commit -m "Table map: new marker layout"`.
6. **Simple tests**, arm unpowered:
   - `./rig check`: hands off the markers for the first second. Rest the closed fingertips on a
     marker corner (error < 5 mm), then `b` and the ball on a marker centre (< 5 mm). `q`.
   - `./rig track`: roll the ball a few times. The predicted crossing should land where the ball
     crosses the line; `q` prints the prediction error over the rolls (the open jaw catches within
     ±28 mm).
7. **Powered tests:** `./rig move --heights 60`, ENTER powers the arm, and the jaw passes along the
   intercept line 60 mm above the table: check the printed height with a ruler. Then `./rig move`
   for 60 / 35 / 20 mm.
8. **Catch:** `./rig catch`. Wait for `camera located: OK`, open the jaw by hand, press ENTER in the
   terminal: the arm goes from rest to the initial pose and waits at the intercept line. Roll the
   ball. `q` in the window brings the arm back to rest and releases it.

**Next days, same markers:** `./rig calibrate` opens at step 4. Redo the wrist level (the gripper
is remounted daily), press ENTER through the rest, then `./rig catch`. If the camera was moved, there
is nothing to redo: every task locates it from the markers when it starts.

Ctrl+C in the terminal stops any task, like `q`. Details and troubleshooting:
[docs/real_rig.md](docs/real_rig.md).

### Safety

The calibration and check tools open the CAN bus read-only: the wrapper refuses enable, command and
parameter writes to any motor not allowed. The powered tasks use the motors' own position mode with
a firmware speed limit and current cap, and ask for ENTER before powering. Ctrl+C holds the arm
where it is: support it and press ENTER to release it, or press Ctrl+C again to release it at once.
Joints 2 and 3 carry the arm's weight and drop when released. See
[docs/real_rig.md#powered-runs](docs/real_rig.md#powered-runs).

## How it works

1. **Tracking.** The ball is found by an HSV colour threshold inside the calibrated table area. Its
   centroid is projected onto the plane of the ball centre and filtered by a constant-velocity Kalman
   filter with real frame timestamps. No learned detector; about 1-2 ms per frame.
2. **Prediction.** A straight-line extrapolation gives where and when the ball crosses the intercept
   line (x = 0.40 m from the arm base).
3. **Planning.** The open jaw (fingertips 20 mm above the table, tilted 30° down, turned to face the
   ball) is moved along the line with rate limits. Damped least-squares IK in the MuJoCo model gives
   the joint targets.
4. **Calibration (real rig).** A one-time table map holds the ArUco markers and the table plane in
   the arm's frame, fitted to camera views, fingertip touches on marker corners and a fingertip sweep
   of the table. Heights measured with the arm's own readings cancel its model's height bias. The
   camera is then located from the markers wherever it stands.

## Contributing

- Branch from `main`, keep pull requests focused, and describe how you tested the change (sim run,
  hardware run, or both).
- Don't commit anything from `outputs/`: camera frames can show people in the lab.
- Hardware changes: check with `./rig arm` (read only) first, and never bypass the motor allow-list
  in `RobStrideBus`.

## Credits and licenses

- Code in this repository: [MIT](LICENSE).
- Arm model (MJCF, URDF, meshes): [Seeed-Projects/reBot-DevArm](https://github.com/Seeed-Projects/reBot-DevArm)
  and the RS digital-twin repositories listed in `assets/rebot_b601_rs/source.json`, under
  CERN-OHL-W-2.0 (`assets/rebot_b601_rs/LICENSE_reBot-DevArm`). Wrist-camera assets keep their own
  licences in `assets/rebot_b601_rs/mjcf/licenses/`.
- Motor protocol details follow Seeed's `reBotArm_control_py` and RobStride's documentation.
- PCAN on macOS: [PCBUSB library](https://www.mac-can.com) (separate download, non-commercial use).
