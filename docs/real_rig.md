# Real rig guide

Everything needed to run the ball interception on the physical reBot Arm B601-RS with an Intel
RealSense D435i: setup, the calibration, and the powered runs. Every task starts with `./rig` from
the repository root.

- [Hardware](#hardware)
- [Software setup](#software-setup)
- [The launcher](#the-launcher)
- [Calibration](#calibration)
- [Checks and shadow tracking](#checks-and-shadow-tracking)
- [Powered runs](#powered-runs)
- [Validation experiment](#validation-experiment)
- [Troubleshooting](#troubleshooting)
- [Protocol notes](#protocol-notes)

## Hardware

| item | details |
|---|---|
| Arm | Seeed Studio reBot Arm B601-RS: RobStride RS06 motors on joints 1-3, RS00 on joints 4-6 and the gripper (CAN IDs 1-7, host ID 0xFD), 48 V supply |
| CAN adapter | PEAK PCAN-USB (shows up as "XCAN-USB", USB 0c72:000c), 1 Mbit/s |
| Camera | Intel RealSense D435i, USB 3, on a stand looking at the table; it may be moved between runs |
| Markers | 4 or more ArUco markers taped flat on the table (`DICT_ARUCO_ORIGINAL`, IDs 107, 880, 467, 718, 71.5 mm black square). Defined in `eaiproject/markers.py` |
| Ball | orange ping-pong ball, 44 mm |

Marker layout: at least 2 within the arm's reach (they get touched), at least 1 further than 0.50 m
from the base (without one in view the camera pose is loose far down the table), not in a line, as
far apart as the arm reaches. Each needs a white margin of 1 cm or more and no tape over the black border.
Measure one printed marker with a ruler: it must match `SIZES` in `eaiproject/markers.py` within
0.3 mm, because the camera measures every distance in units of the marker size.

Tested on an Apple Silicon Mac. On Linux the camera needs no root, and `RobStrideBus` would need
the SocketCAN interface instead of PCAN.

## Software setup

```bash
python3.12 -m venv .venv                     # or: uv venv --python 3.12 .venv
.venv/bin/pip install -r requirements.txt
./rig test                                   # unit tests, no hardware
```

**PCAN driver on macOS (no sudo).** python-can needs the free PCBUSB library from
[mac-can.com](https://www.mac-can.com) (non-commercial use). Copy the dylib into `~/lib`, which macOS
searches by default:

```bash
curl -LO https://github.com/mac-can/PCBUSB-Library/releases/download/v0.13/macOS_Library_for_PCANUSB_v0.13.tar.gz
tar xzf macOS_Library_for_PCANUSB_v0.13.tar.gz
mkdir -p ~/lib && cp PCBUSB/libPCBUSB.0.13.dylib ~/lib/ && ln -sf ~/lib/libPCBUSB.0.13.dylib ~/lib/libPCBUSB.dylib
```

**RealSense on macOS.**

- librealsense must run as **root** on macOS, and root processes can't read `~/Documents`. So the
  camera runs as a small frame server (`scripts/real/rs_server.py`) from `~/realsense_tools`, with
  its own venv. `scripts/real/start_camera.sh` sets that up on first use and starts the server with
  a macOS password dialog. `./rig` calls it when a task needs the camera.
- The scripts switch the running server between 1280x720 @ 30 (calibration views) and 640x480 @ 60
  (tracking) without a password.
- Use `pyrealsense2-macosx==2.54.2`; 2.56.5 segfaults when it lists devices.
- The colour stream only delivers frames while the depth stream is also enabled. The server does this.

## The launcher

```bash
./rig status             # camera server, CAN adapter, table map, arm poses, ball colour
./rig camera start       # also: stop, hd (1280x720), vga (640x480)
./rig view               # the camera with the table markers, for aiming it
./rig arm                # joint states, read only: does the arm answer on CAN?
./rig calibrate          # the table map once; daily: wrist zero, arm poses, ball colour
./rig check              # live check of the calibration
./rig track              # ball tracking and the planned motion; the arm is NOT powered
./rig move               # POWERED: the jaw along the intercept line at 60 / 35 / 20 mm
./rig catch              # POWERED: the jaw follows the ball
./rig test               # unit tests
```

Options after the task go to its script: `./rig calibrate --solve`, `./rig move --heights 40`,
`./rig catch --help`.

## Calibration

The calibration measures where the table markers and the table surface are **in the arm's own
frame**: positions as the arm's model reports them from the joint readings (`Fingertip`, MuJoCo FK).
That model reads heights 8-15 mm off, but since the planner commands the arm through the same
model, the error cancels. With the map, the camera can stand anywhere: every tool finds the camera
pose from the markers when it starts.

```bash
./rig calibrate          # continues the latest session; with a map it opens at the daily steps
./rig calibrate --new    # new table map: the table or a marker moved
./rig calibrate --solve  # solve again from the saved data, no hardware
```

| step in the window | arm | time | when |
|---|---|---|---|
| 1 Views: the camera at 5 or more positions | not needed | 5 min | table map |
| 2 Touch: fingertips on marker corners | unpowered, moved by hand | 8 min | table map |
| 3 Sweep: fingertips slid over the table | unpowered, moved by hand | 5 min | table map |
| Solve: result on screen, map and report written | not needed | 10 s | table map |
| 4 Arm: wrist zero; rest and initial poses | unpowered, moved by hand | 1 min | every day |
| 5 Ball: the ball colour at the tracking size | not needed | 1 min | when the light or the ball changed |

ENTER goes to the next step, BACKSPACE back, `1`-`5` jump, `q` quits. Everything is saved as it is
recorded, so you can quit at any time and continue later. Nothing is powered: the CAN bus is opened
read-only. Support the unpowered arm while moving it: joints 2 and 3 drop. It also starts without
the arm (off, or no CAN adapter): steps 2-4 then show `ARM NOT CONNECTED` and pick the arm up by
themselves once it answers, without a restart. `./rig check` does the same.

The quality rules below never block: every reading is recorded, and one that breaks a rule gets a
red `CHECK:` line saying what is weak (`u` undoes it). The solve and camera location work the same
way: they refuse only when nothing can be computed (no marker in view, no marker with two touched
corners), and otherwise warn.

### Table map (steps 1-3, once)

**1 Views.** Arm and hands away from the markers. Put the camera somewhere, wait until it is still,
SPACE (30 frames are averaged). Move it and repeat: at least 5 views, better 6-8, from the left,
right and front, low (about 30°) to high (about 60°), near and far. Best with 3 or more markers in
each view and the camera still (corners steady within 0.5 px while averaging). A view that does not
link to a touched marker is left out of the solve, with a warning.

**2 Touch.** The window circles one corner in yellow (TL, TR, BR, BL; TL has the filled dot). Rest
the **closed** fingertips on exactly that corner, pressed on the table, hold still, SPACE. Any
gripper angle. After the 4 corners `n` goes to the next marker (`p` back); pass the ones you cannot
reach with `n`. Best: all 4 corners of 2 or more markers (the solve needs 2 corners of one). A red
`CHECK:` line means the point does not fit the others (wrong corner, tip off the table) or the tip
moved while recording: `u` and touch it again.

**3 Sweep.** **Hold SPACE** and slide the closed fingertips slowly over the table, at any gripper
angle (if holding the key does not record, `r` toggles recording). Cover the yellow box around the
intercept line to 70 % or more, then the table around it. `u` removes the last stroke. Samples where
the tip left the table are found and dropped by the solve; the report says how many.

**Solve.** ENTER after step 3 solves, writes `config/rig_map.json` and the report in
`outputs/rig_calib/<date>/report/`, and shows the result. Commit the map when it is clean.

| in the report | accept | if not |
|---|---|---|
| reprojection RMS | < 1.5 px (expect 0.3-0.7) | `report/reproj_view_XX.png`: one bad view, record the views again; all bad, a marker lifted off the table |
| corner touches RMS | < 5 mm (expect 2-3) | `report.txt` lists every touch: redo the ones marked `<-- check` (step 2: select the marker with `n`, `u`, SPACE) |
| scale check | within ±0.5 % | the printed markers are not the size in `markers.py` |
| sweep RMS | < 1.5 mm | `report/sweep_heatmap.png`: random speckle = lifted tip, sweep again; a smooth trend = arm model error |
| warnings | none | each says what to do |

Our 2026-11-25 map: 0.64 px reprojection, corner touches 1.7 mm in the table plane, sweep 1.4 mm,
table 5.4-5.7 mm (model frame) along the intercept line. The heights depend on the arm pose by
about 1-2 mm: a joint-level model error, left to the arm parameter calibration.

### Daily steps (4 and 5)

**4 Arm.** Three parts: `n` / `p` move between them, SPACE records, `u` goes back to the saved value.
ENTER keeps what is saved. Saved to `outputs/calibration/arm.json` (local, like the ball colour).

1. **Wrist level** (every day: the gripper is remounted). Open the jaw a few cm by hand and rest
   **both** fingertips on the table, fingers forward and down (not straight down), SPACE; then at a
   second spot. The readings should agree within about 1°. Both fingertips touching means the jaw
   is level, which gives the zero of joint 6 (the wrist roll; ours read about 15° off). Joint 6
   turns about the finger axis and the fingertip lies on that axis, so the zero moves no fingertip
   position (the map stays valid); it rolls the jaw, which matters at the blocking pose.
2. **Rest** (when it should change). Fold the arm into the pose where it lies on itself without
   support, SPACE. The powered tasks end here and release the motors here.
3. **Initial** (when it should change). Hold the arm where its work should start from: above the
   table, clear of the markers and the camera, SPACE. The powered tasks pass through it on the way
   to their work and back.

**5 Ball.** The camera switches to the tracking size. Put the ball **alone** on the table, still,
hands away, `a` tunes the colour. Roll it: the green ring must follow it, the mask (bottom right)
must show only the ball. The sliders in the "ball colour" window fine-tune it (raise S lo / V lo if
the table or skin shows up, lower them if the ball breaks up in shadow). Leaving the step saves a
changed colour to `outputs/calibration/ball_hsv.json`.

## Checks and shadow tracking

```bash
./rig check --label camA   # live check, arm unpowered (or off)
./rig track                # ball tracking, arm not powered
```

Both locate the camera from the markers when they start (hands and arm off the markers for a
second) and again when you press `c`, after the camera was moved.

`./rig check` shows the map (magenta) over the detected markers (green), with the error per marker;
a magenta ring at the arm's reported fingertip, which should sit on the real fingertips as you move
the unpowered arm (rest them on a marker corner: the error shows, SPACE records it); and with `b`,
the tracked ball against known points (marker centres, and spots added with `t`). Accept: markers
< 1.5 px and < 5 mm, fingertip on a corner < 5 mm, ball on a known point < 5 mm. Records go to
`outputs/calibration/validation.csv`, `s` saves a screenshot.

`./rig track` shows the detection, the track, the predicted crossing of the intercept line, where
the jaw would go, and a render of the pose the planner would command. Each roll is scored: the
crossing predicted 0.5 / 0.3 / 0.15 s before the line against where the ball really crossed (the
open jaw catches within about ±28 mm). A red **CAMERA MOVED?** means the markers no longer sit where
the pose says: press `c`.

| message | do |
|---|---|
| `camera located: OK` (< 1.5 px, < 5 mm) | go on |
| `camera located: WARN` (1.5-3 px) | usable; check that the camera is fixed, press `c` |
| `CHECK: only N map marker(s) in view` | works; the pose is better with 3 or more: move the camera closer or higher |
| `CHECK: no marker far from the arm in view` | works; the pose is loose far down the table: aim further down |
| `CHECK: marker ... does not fit the map` | it moved on the table: fine for today (left out), calibrate again before relying on it |
| `CANNOT LOCATE THE CAMERA: no map marker visible` | aim the camera at the table; hands and arm off the markers |

## Powered runs

```bash
./rig move               # the jaw along the intercept line at 60, 35 and 20 mm above the table
./rig move --heights 60  # one high pass first
./rig catch              # the jaw follows the ball
```

Both scripts power all six arm joints in the motors' own position mode and go
**rest → initial pose → their work → initial pose → rest**, then release the motors. Without
recorded poses they start and end where the arm was resting.

- **ENTER** powers the arm. Before it, the arm should be at rest and the workspace clear.
- **Ctrl+C** (in the terminal) holds the arm where it is. **Support the arm**, then ENTER releases
  it: joints 2 and 3 drop when released. A second Ctrl+C releases at once. An error holds and
  asks the same way.
- The protection is the motors' own: a firmware speed limit (0.8 rad/s; `catch --joint-speed`,
  default 1.5) and a current cap per joint (6 / 18 / 14 / 5 / 4 / 4 A), written to RAM before
  enabling (gone at power-off; nothing goes to the motors' flash). There are no software stop
  checks: keep a hand near the arm's power switch.

`./rig move` poses the jaw as when blocking and passes it along the intercept line (x = 0.40 m),
highest first, alternating direction, then lifts before going back. It prints the fingertip's
height above the table at the start and end of each pass: check it with a ruler.

`./rig catch` locates the camera, then after ENTER waits at the intercept line and follows the
planner on the tracked ball, one target per camera frame. Open the jaw by hand first (the gripper
motor is not powered). `q` in the window finishes, `r` resets the tracker. Commands and ball states
are saved to `outputs/hardware/catch_<time>.npz`.

The planner (`eaiproject/real_intercept.py`) is the simulation's policy in real time: straight-line
crossing prediction, the jaw facing the ball, rate limits on the jaw pose (0.6 m/s sideways), IK in
the MuJoCo model, heights from the measured table plane.

## Validation experiment

3 camera positions × 5 ball positions with known coordinates, error in mm in the arm frame.

1. Once: in `./rig check`, rest the fingertips on 2 spots on the intercept line (e.g. y = ±0.15),
   press `t` at each and mark it with a pen dot. The 5 ball positions are these 2 dots and the
   centres of 3 markers.
2. For each camera position (front, left, right and higher): move the camera,
   `./rig check --label camA` (`camB`, `camC`), touch 2 marker corners (SPACE each), then `b` and
   the ball on each of the 5 points (SPACE each), `s` for a screenshot.
3. The rows are in `outputs/calibration/validation.csv`. A ball on a marker centre tests locating
   the camera; a ball on a pen dot and the corner touches also test the map in the arm frame.
   Pass: mean error < 5 mm for every camera position, no systematic shift between them.

`outputs/` is not in git, and its camera frames can show people: copy only the reports, the CSV
and the screenshots you need.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `No motor answered` / `transmit queue is full` / joints not answering | arm power off, or the CAN cable is not connected |
| `PcanCanInitializationError ... handle is invalid` | the PCAN-USB adapter is unplugged |
| camera server: `failed to set power state` | another program holds the camera: quit video apps, replug the camera, start again. A stuck earlier server is killed by `start_camera.sh` itself |
| camera server: `Frame didn't arrive` | the camera re-enumerated: replug it, `./rig camera stop`, start again |
| `the camera server did not switch` | `./rig camera stop`, then the task again |
| small markers not detected | too small or too edge-on in the image: move the camera closer or higher |
| ball not found | tune the colour: `./rig calibrate --step ball` |

## Protocol notes

RobStride private CAN protocol, 29-bit extended IDs, comm type in bits 24-28 (`eaiproject/robstride.py`):

| type | meaning | id layout | data |
|---|---|---|---|
| 1 | MIT command | `1<<24 \| torque_u16<<8 \| motor` | pos, vel, kp, kd (u16 big-endian) |
| 2 | feedback | `2<<24 \| mode/faults<<16 \| motor<<8 \| host` | pos, vel, torque, temp×10 |
| 3 / 4 | enable / stop | `t<<24 \| host<<8 \| motor` | zeros |
| 17 / 18 | read / write parameter | `t<<24 \| host<<8 \| motor` | index (u16 LE), value at bytes 4-7 |

Scaling: position ±4π rad. RS06: ±50 rad/s, ±36 Nm, kp 0-5000, kd 0-100. RS00: ±33 rad/s, ±14 Nm,
kp 0-500, kd 0-5. The torque ranges match the motors' own `limit_torque` parameter.

Useful parameters: 0x7005 run_mode (0 MIT, 1 position), 0x7016 loc_ref, 0x7017 limit_spd,
0x7018 limit_cur, 0x7019 mechPos, 0x701A iqf, 0x701B mechVel, 0x701C VBUS.
The gripper travel is 0 (closed) to 5.0 rad (open), from Seeed's config.

`RobStrideBus(movable=...)` refuses enable / command / parameter writes to any motor not listed:
the calibration and check tools open it with `movable=()`, the powered tasks with the six arm joints.
