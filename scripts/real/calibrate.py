"""Calibration of the rig, in one guided window. Read-only on CAN: nothing is ever powered. Starts
without the arm too: steps 2-4 say ARM NOT CONNECTED until it answers, then work by themselves.

    ./rig calibrate                     # continue the latest session (daily: opens at step 4)
    ./rig calibrate --new               # new table map: the table or a marker moved
    ./rig calibrate --solve             # solve again from the saved data (no hardware)
    ./rig calibrate --step ball         # open at a step: views, touch, sweep, arm, ball

  table map (once; again with --new when the table or a marker moved)
  1 VIEWS   Move the camera to >= 5 positions (angle and distance). At each, with arm and hands
            clear of the markers, SPACE averages every marker corner over 30 frames.   u undo
  2 TOUCH   Rest the CLOSED fingertips of the unpowered arm on the corner circled in yellow, SPACE.
            Corners go TL, TR, BR, BL (TL has the filled dot).
                       n / p next / previous marker, s skip corner, c marker centre, u undo
  3 SWEEP   HOLD SPACE and slide the closed fingertips slowly over the table, any gripper angle
            (r toggles recording instead).                                 u undo last stroke
  SOLVE     Fits the map, writes config/rig_map.json and a report, shows the result.

  daily (a minute; ENTER keeps the saved values)
  4 ARM     WRIST LEVEL (the gripper is remounted every day): both fingertips of the open jaw on the
            table, SPACE, at two spots. REST and INITIAL poses only when they should change.
                                            SPACE record, n / p next / previous part, u undo
  5 BALL    Ball colour at the tracking resolution: ball alone on the table, a = auto-tune, roll it
            and watch the mask. Sliders fine-tune.   u back to the saved colour, c locate camera

ENTER next step, BACKSPACE previous, 1-5 jump, q quit. Everything is saved as it is recorded.
"""
from __future__ import annotations

import signal
import sys
signal.signal(signal.SIGINT, lambda *_: sys.exit(130))     # Ctrl+C while starting up: quiet (cli.run takes over)
import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eaiproject import cli  # noqa: E402
from eaiproject import arm as arm_mod  # noqa: E402
from eaiproject import markers, rig_report  # noqa: E402
from eaiproject import rig_calib as rc  # noqa: E402
from eaiproject.rig_draw import (GREEN, GREY, MAGENTA, RED, WHITE, YELLOW, TopDown, ipt, outline, put,  # noqa: E402
                                 window_closed)

N_FRAMES = 30                # frames averaged per view
# Quality checks: every reading is recorded; breaking one only adds a CHECK note (u undoes the reading).
MAX_JITTER_PX = 0.5          # view: the corners moved more than this while averaging (camera not still?)
STILL_MM = 1.5               # touch: the fingertip moved more than this in 0.25 s
STILL_DEG = 1.0              # wrist level: a joint moved more than this while reading (rest / initial: no check)
MIN_SENS = 0.4               # wrist level: below this the fingers point nearly straight down -> a weak reading
NO_SENS = 0.05               # ... and below this the roll cannot be read at all: the only refusal
HOLD_GAP = 1.0               # s; SPACE repeats arriving within this gap count as "key held"
GRID = 0.02                  # m, sweep coverage cell
TARGET_X, TARGET_Y = (rc.GATE_X - 0.06, rc.GATE_X + 0.06), (-0.26, 0.26)      # box to cover around the intercept line


def sweep_frame(sess: Path) -> rc.TableFrame:
    """Table axes for the sweep's guide (its box and top view), before this session has a map: the table
    is where the corners touched in step 2 are; without touches, as in the existing map; else in front."""
    path = sess / "touches.json"
    pts = [t["p"] for t in json.loads(path.read_text())["touches"]] if path.exists() else []
    if pts:
        return rc.TableFrame(rc.table_yaw_deg(pts))
    return rc.RigMap.load(rc.MAP_PATH).frame if rc.MAP_PATH.exists() else rc.TableFrame()
VGA = (640, 480, 60)                            # camera: the ball step runs at the tracking size

WIN = "rig calibration"
NAV = "ENTER next step   BACKSPACE previous   1-5 jump (4 arm, 5 ball)   q quit (everything is saved)"
STEPS = ("views", "touch", "sweep", "solve", "arm", "ball")


def open_camera():
    from eaiproject.realsense_client import RealSenseClient
    try:
        return RealSenseClient()
    except OSError:
        print("Camera server not running: ./rig camera start")
        return None


def to_mode(cam, mode):
    """The camera client at this size (the server restreams; no password)."""
    from eaiproject.realsense_client import RealSenseClient, switch_mode
    try:
        return switch_mode(cam, *mode)
    except TimeoutError as e:
        print(e, flush=True)
        return RealSenseClient()


def no_readings(arm) -> str:
    """Why there is no joint reading: the arm is not connected, or it stopped answering."""
    return arm.status() or "no joint readings (arm power?)"


def nav(key: int) -> str | None:
    """Step navigation shared by all steps: where to go for this key, or None."""
    if key == ord("q") or window_closed(WIN) or cli.ctrl_c():                  # Ctrl+C in the terminal = q
        return "quit"
    if key in (13, 10):
        return "next"
    if key in (8, 127):
        return "prev"
    return {ord("1"): "views", ord("2"): "touch", ord("3"): "sweep", ord("4"): "arm", ord("5"): "ball"}.get(key)


def footer(img, msg: str = "") -> None:
    if msg:
        put(img, msg, (10, img.shape[0] - 34), RED, 0.5)
    put(img, NAV, (10, img.shape[0] - 10), GREY, 0.45)


def big(img, width: int = 1280) -> tuple[np.ndarray, float]:
    """The frame upscaled to at least `width` px, so the instruction lines fit; and the scale."""
    s = max(1.0, width / img.shape[1])
    return (cv2.resize(img, None, fx=s, fy=s) if s > 1 else img.copy()), s


def grab_average(cam, last_fn, n: int = N_FRAMES, timeout: float = 4.0):
    """Table-marker corners over n fresh frames. Returns (corners, jitter, n_frames, last frame);
    grab_average.rates then holds how often each marker was read (a flickering marker still counts)."""
    frames, fn, t_end, g = [], last_fn, time.monotonic() + timeout, None
    while len(frames) < n and time.monotonic() < t_end:
        g = cam.read(timeout=2.0, newer_than=fn)
        fn = g.frame_number
        frames.append(markers.detect(g.img))
    mean, jitter = rc.average_corners(frames)
    grab_average.rates = rc.read_rates(frames)
    return mean, jitter, len(frames), g


def map_pose(cam, last_fn):
    """Camera pose from the existing map for the sweep overlay, or None (no map / markers hidden)."""
    if not rc.MAP_PATH.exists():
        return None, last_fn
    mean, _, _, g = grab_average(cam, last_fn, 10, 2.0)
    try:
        res = rc.relocalize(rc.RigMap.load(rc.MAP_PATH), cam.K, cam.dist, mean)
    except rc.CalibrationError:
        return None, g.frame_number
    return (res.rvec, res.tvec), g.frame_number


# ----------------------------------------------------------------------------------- views
def run_views(cam, sess: Path) -> str:
    path = sess / "views.json"
    views = json.loads(path.read_text())["views"] if path.exists() else []
    msg, last_fn = "", None
    recent: deque = deque(maxlen=N_FRAMES)             # the last detections: a far marker is not read in every frame
    while True:
        f = cam.read(timeout=2.0, newer_than=last_fn)
        last_fn = f.frame_number
        img = f.img.copy()
        recent.append(markers.detect(f.img))
        rates = rc.read_rates(list(recent))
        steady = {k: r for k, r in rates.items() if r >= 0.3}          # what a recording would keep
        for k in steady:
            c = next(d[k] for d in reversed(recent) if k in d)         # its latest reading: no flicker on screen
            outline(img, c, GREEN if steady[k] >= 0.7 else YELLOW, 2, f"{k} {100 * steady[k]:.0f}%")
        n = len(views)
        put(img, f"1 VIEWS: {n} recorded ({rc.MIN_VIEWS}+ wanted, 6-8 is better). Hands + arm off the markers, camera still, SPACE",
            (10, 24), YELLOW if n < rc.MIN_VIEWS else GREEN)
        put(img, f"markers in view: {len(steady)}  {sorted(steady)}   (% = read in the last {len(recent)} frames; yellow = small "
                 "in the image, closer is better)", (10, 48), GREEN if len(steady) >= 3 else YELLOW, 0.45)
        seen = {k: sum(k in v["corners"] for v in views) for k in markers.SIZES}
        put(img, "views per marker: " + "  ".join(f"{k}: {c}" for k, c in seen.items()), (10, 72))
        put(img, "then move the camera: left / right / front, low (30 deg) to high (60 deg), near and far   (u = undo)",
            (10, 96), WHITE, 0.45)
        footer(img, msg)
        cv2.imshow(WIN, img)
        key = cv2.waitKey(1) & 0xFF
        if nav(key):
            return nav(key)
        if key == ord("u") and views:
            v = views.pop()
            path.write_text(json.dumps({"views": views}, indent=1))
            msg = f"removed {v['name']}"
        if key == ord(" "):
            mean, jitter, nf, g = grab_average(cam, last_fn)
            last_fn = g.frame_number
            if not mean:
                msg = "no table marker stable in view: nothing to record"
                continue
            name = f"view_{int(views[-1]['name'].split('_')[1]) + 1 if views else 0:02d}"
            cv2.imwrite(str(sess / f"{name}.png"), g.img)
            views.append({"name": name, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "serial": cam.serial,
                          "width": cam.width, "height": cam.height, "K": cam.K.tolist(), "dist": cam.dist.tolist(),
                          "n_frames": nf, "corners": {k: c.tolist() for k, c in mean.items()},
                          "read_rate": {k: round(grab_average.rates.get(k, 0.0), 2) for k in mean},
                          "jitter_px": {k: round(j, 3) for k, j in jitter.items()}, "frame": f"{name}.png"})
            path.write_text(json.dumps({"views": views}, indent=1))
            weak = [k for k in mean if grab_average.rates.get(k, 1.0) < 0.7]
            notes = ([f"only {len(mean)} marker(s) in this view (3+ link the map better)"] if len(mean) < 3 else []) + \
                    ([f"marker {', '.join(weak)} read in under 70 % of the frames (small in the image: a closer view reads better)"]
                     if weak else []) + \
                    ([f"corners moved {max(jitter.values()):.2f} px while averaging (camera still?)"]
                     if max(jitter.values()) > MAX_JITTER_PX else [])
            msg = f"CHECK {name}: " + "; ".join(notes) + "  (u = undo)" if notes else ""
            print(f"{name}: {sorted(mean)}  jitter max {max(jitter.values()):.2f} px  {msg}", flush=True)


# ----------------------------------------------------------------------------------- touch
def touch_sanity(touches: list, new: dict, size: float, moved_mm: float = 0.0) -> str:
    """Catches a wrong corner or a moving fingertip right away (the touch is still recorded; u undoes it)."""
    out = [f"the fingertip moved {moved_mm:.1f} mm while recording"] if moved_mm > STILL_MM else []
    zs = [t["p"][2] for t in touches]
    if len(zs) >= 2 and abs(new["p"][2] - np.median(zs)) > 0.012:
        out.append(f"height differs {1000 * (new['p'][2] - np.median(zs)):+.0f} mm from the other touches")
    loc = rc.local_corners(size)
    for t in touches:
        if t["key"] == new["key"] and t["corner"] != new["corner"]:
            want = np.linalg.norm(loc[t["corner"]] - loc[new["corner"]])
            got = np.linalg.norm(np.array(t["p"]) - new["p"])
            if abs(got - want) > 0.008:
                out.append(f"{1000 * got:.0f} mm from {rc.CORNER_NAMES[t['corner']]} (expected {1000 * want:.0f})")
    return ("CHECK: " + "; ".join(out) + "  (u = undo)") if out else ""


def run_touch(cam, arm, sess: Path) -> str:
    path = sess / "touches.json"
    touches = json.loads(path.read_text())["touches"] if path.exists() else []
    seen: dict[str, np.ndarray] = {}                # last known corners (the hand hides markers while touching)
    cur, skipped, msg, last_fn = None, set(), "", None

    def done(k):
        return {t["corner"] for t in touches if t["key"] == k}

    def next_corner(k):
        return next((c for c in range(4) if c not in done(k) and (k, c) not in skipped), None)

    def save():
        path.write_text(json.dumps({"touches": touches}, indent=1))

    while True:
        f = cam.read(timeout=2.0, newer_than=last_fn)
        last_fn = f.frame_number
        img = f.img.copy()
        for k, c in markers.detect(f.img).items():
            near = k in seen and np.abs(seen[k] - c).max() < 3.0
            seen[k] = 0.7 * seen[k] + 0.3 * c if near else c
        keys = sorted(seen)
        if cur not in keys:
            cur = next((k for k in keys if next_corner(k) is not None), keys[0] if keys else None)
        cc = next_corner(cur) if cur else None
        s = arm.latest()
        for k in keys:
            c = seen[k]
            outline(img, c, YELLOW if k == cur else GREY, 2 if k == cur else 1, k)
            for j in range(4):
                if j in done(k):
                    cv2.circle(img, ipt(c[j]), 5, GREEN, -1)
                elif k == cur:
                    cv2.circle(img, ipt(c[j]), 5, YELLOW, 1)
                    put(img, rc.CORNER_NAMES[j], c[j] + 0.25 * (c[j] - c.mean(axis=0)) + [-8, 4], GREY, 0.4)
            if rc.CENTRE in done(k):
                cv2.circle(img, ipt(c.mean(axis=0)), 5, GREEN, -1)
        if cur and cc is not None:
            tgt = seen[cur][cc]
            cv2.circle(img, ipt(tgt), 16, YELLOW, 2, cv2.LINE_AA)
            cv2.drawMarker(img, ipt(tgt), YELLOW, cv2.MARKER_CROSS, 44, 1, cv2.LINE_AA)
            put(img, f"touch here: {rc.CORNER_NAMES[cc]}", tgt + [20, -14], YELLOW, 0.6, 2)
            put(img, f"2 TOUCH: closed fingertips on corner {rc.CORNER_NAMES[cc]} of marker {cur}, hold still, SPACE", (10, 24), YELLOW)
        elif cur:
            put(img, f"2 TOUCH: marker {cur} done. n = next marker you can reach (p = previous), ENTER when finished", (10, 24), GREEN)
        else:
            put(img, "2 TOUCH: no marker seen yet: aim the camera at the markers", (10, 24), RED)
        y = 50
        for k in keys:
            d = done(k)
            txt = " ".join(n if i in d else "--" for i, n in enumerate(rc.CORNER_NAMES))
            put(img, f"{'>' if k == cur else ' '} {k:4s} {txt}", (10, y), YELLOW if k == cur else (GREEN if len(d) >= 4 else WHITE), 0.45)
            y += 20
        n_mk = sum(1 for k in keys if len(done(k) - {rc.CENTRE}) >= 2)
        put(img, f"{len(touches)} touches on {n_mk} marker(s). Best: all 4 corners of every marker within reach, 2+ markers",
            (10, y + 4), GREEN if n_mk >= 2 else WHITE, 0.45)
        put(img, "s = skip this corner   c = record the marker centre (optional)   u = undo", (10, y + 24), GREY, 0.42)
        if s is not None:
            put(img, f"fingertip x {s.p[0] * 100:6.1f}  y {s.p[1] * 100:+6.1f} cm  z {s.p[2] * 1000:6.1f} mm",
                (10, img.shape[0] - 58))
        else:
            put(img, no_readings(arm), (10, img.shape[0] - 58), RED)
        footer(img, msg)
        cv2.imshow(WIN, img)
        key = cv2.waitKey(1) & 0xFF
        if nav(key):
            return nav(key)
        if key in (ord("n"), ord("p")) and keys:
            cur = keys[(keys.index(cur) + (-1 if key == ord("p") else 1)) % len(keys)]
            msg = ""
        if key == ord("s") and cur and cc is not None:
            skipped.add((cur, cc))
            msg = f"skipped {rc.CORNER_NAMES[cc]} of {cur}"
        if key == ord("u") and touches:
            t = touches.pop()
            skipped.discard((t["key"], t["corner"]))
            cur = t["key"]
            save()
            msg = f"removed {t['key']} {rc.CORNER_NAMES[t['corner']]}"
        if key in (ord(" "), ord("c")) and cur:
            corner = rc.CENTRE if key == ord("c") else cc
            st = arm.steady()
            if corner is None:
                msg = "all 4 corners of this marker are done: n = next marker, ENTER = next step"
            elif st is None:
                msg = no_readings(arm)
            else:
                new = {"key": cur, "corner": corner, "p": st[0].p.tolist(), "q": st[0].q.tolist(),
                       "time": time.strftime("%Y-%m-%d %H:%M:%S")}
                msg = touch_sanity(touches, {**new, "p": st[0].p}, markers.SIZES[cur], st[1] * 1000)
                touches[:] = [t for t in touches if (t["key"], t["corner"]) != (cur, corner)] + [new]
                save()
                print(f"touch {cur} {rc.CORNER_NAMES[corner]}: ({st[0].p[0]:.4f}, {st[0].p[1]:+.4f}, {st[0].p[2]:.4f}) m  {msg}",
                      flush=True)


# ----------------------------------------------------------------------------------- sweep
def run_sweep(cam, arm, sess: Path) -> str:
    path = sess / "sweep.npz"
    strokes: list[list] = []                       # each stroke: list of (t, q, p)
    if path.exists():
        z = np.load(path)
        strokes.append(list(zip(z["t"], z["q"], z["p"])))
    frame = sweep_frame(sess)
    top = TopDown(frame, x=(0.10, 0.75), y=(-0.40, 0.40), scale=600.0)
    pose, last_fn = map_pose(cam, None)            # image overlay only when a map exists already
    toggled, t_key, t_commit, stroke_open, msg = False, -1e9, None, False, ""

    def save():
        s = [x for st in strokes for x in st]
        if s:
            np.savez(path, t=np.array([x[0] for x in s]), q=np.array([x[1] for x in s]), p=np.array([x[2] for x in s]))

    def commit(t0, t1):
        new = [(s.t, s.q, s.p) for s in arm.between(t0, t1)]
        if new:
            strokes[-1].extend(new)

    while True:
        f = cam.read(timeout=2.0, newer_than=last_fn)
        last_fn = f.frame_number
        now = time.monotonic()
        img = f.img.copy()
        s = arm.latest()
        holding = toggled or now - t_key < HOLD_GAP
        if toggled and stroke_open:                # toggle mode: commit continuously
            commit(t_commit, now)
            t_commit = now
        if not holding and stroke_open:            # key released: samples after the last key event are dropped
            stroke_open = False
            if not strokes[-1]:
                strokes.pop()
            save()

        allp = np.array([x[2] for st in strokes for x in st]).reshape(-1, 3)
        tabp = frame.table_xy(allp)                            # coverage in table axes
        cells = {(int(np.floor(p[0] / GRID)), int(np.floor(p[1] / GRID))) for p in tabp}
        tx = range(int(np.floor(TARGET_X[0] / GRID)), int(np.ceil(TARGET_X[1] / GRID)))
        ty = range(int(np.floor(TARGET_Y[0] / GRID)), int(np.ceil(TARGET_Y[1] / GRID)))
        cover = sum((i, j) in cells for i in tx for j in ty) / (len(tx) * len(ty))

        tv = top.canvas(rc.GATE_X)                 # top view: needs no camera pose
        top.rect(tv, TARGET_X, TARGET_Y, YELLOW, 1)
        for (i, j) in cells:
            top.cell(tv, frame.arm_xy([[(i + 0.5) * GRID, (j + 0.5) * GRID]])[0], GRID, (0, 170, 0))
        if s is not None:
            cv2.drawMarker(tv, ipt(top.px(s.p[:2])[0]), MAGENTA, cv2.MARKER_CROSS, 18, 2)
        put(tv, f"top view, table axes (base turned {frame.yaw_deg:+.0f} deg). green = covered", (8, 16), WHITE, 0.42)
        put(tv, "yellow box = cover this well", (8, 34), YELLOW, 0.42)

        if pose is not None:                       # the same on the camera image
            pr = lambda P: rc.project(cam.K, cam.dist, *pose, np.asarray(P, float).reshape(-1, 3))
            z0 = float(np.median(allp[:, 2])) if len(allp) else (float(s.p[2]) if s is not None else 0.0)
            on_table = lambda xy: np.column_stack([frame.arm_xy(xy), np.full(len(xy), z0)])
            box = pr(on_table([[TARGET_X[0], TARGET_Y[0]], [TARGET_X[1], TARGET_Y[0]],
                               [TARGET_X[1], TARGET_Y[1]], [TARGET_X[0], TARGET_Y[1]]]))
            cv2.polylines(img, [box.round().astype(np.int32)], True, YELLOW, 1, cv2.LINE_AA)
            for c in pr(on_table([[(i + 0.5) * GRID, (j + 0.5) * GRID] for i, j in cells])) if cells else []:
                cv2.circle(img, ipt(c), 3, (0, 200, 0), -1)
            if s is not None:
                cv2.drawMarker(img, ipt(pr(s.p)[0]), MAGENTA, cv2.MARKER_CROSS, 18, 2)
        if s is not None:
            put(img, f"fingertip x {s.p[0] * 100:5.1f} y {s.p[1] * 100:+5.1f} cm  z {s.p[2] * 1000:5.1f} mm   "
                     "(any gripper angle: keep the closed fingertips on the table)", (10, 50), GREEN, 0.5)
        else:
            put(img, no_readings(arm), (10, 50), RED)
        put(img, ("3 SWEEP: RECORDING" if holding else "3 SWEEP: hold SPACE and slide the closed fingertips over the table"),
            (10, 24), RED if holding else YELLOW, 0.65, 2 if holding else 1)
        put(img, f"{len(allp)} samples   {len(cells)} cells   box around the intercept line: {100 * cover:.0f} % covered "
                 "(aim >= 70 %)", (10, 76), GREEN if cover >= 0.7 else WHITE, 0.5)
        put(img, "r = recording on/off without holding   u = undo the last stroke", (10, 98), GREY, 0.42)
        footer(img, msg)
        h = img.shape[0]
        tv = cv2.resize(tv, (int(tv.shape[1] * h / tv.shape[0]), h))
        cv2.imshow(WIN, np.hstack([img, tv]))

        key = cv2.waitKey(1) & 0xFF
        if nav(key) and not holding:
            save()
            return nav(key)
        if key == ord(" ") and not toggled:
            tk = time.monotonic()
            if not stroke_open:
                strokes.append([])
                stroke_open = True
            else:
                commit(t_commit, tk)               # everything up to this key event was with the key down
            t_key = t_commit = tk
        if key == ord("r"):
            toggled = not toggled
            if toggled:
                strokes.append([])
                stroke_open, t_commit = True, time.monotonic()
            msg = "recording ON (r stops)" if toggled else ""
        if key == ord("u") and strokes and not holding:
            n = len(strokes.pop())
            if not strokes and path.exists():
                path.unlink()
            save()
            msg = f"removed the last stroke ({n} samples)"


# ------------------------------------------------------------------------------------- arm
ARM_PARTS = ("level", "rest", "initial")
ARM_TEXT = {
    "level": "WRIST LEVEL: open the jaw a few cm by hand, rest BOTH fingertips on the table, fingers forward and "
             "down, SPACE. Two spots.",
    "rest": "REST: fold the arm into the pose where it lies on itself without support, SPACE",
    "initial": "INITIAL: hold the arm where its work should start (above the table, clear of the markers), SPACE. Roughly",
}


def table_plane() -> np.ndarray:
    if rc.MAP_PATH.exists():
        return np.asarray(rc.RigMap.load(rc.MAP_PATH).plane, float)
    return np.array([0.005, 0.0, 0.0])            # no map yet: a level table, 5 mm up in the arm's frame


def motor_reading(reader, seconds: float = 0.5) -> tuple[np.ndarray, float] | None:
    """Mean MOTOR reading over the next `seconds` and how much the largest joint moved meanwhile (deg)."""
    t0 = time.monotonic()
    time.sleep(seconds)
    ss = reader.between(t0, time.monotonic())
    if len(ss) < 3:
        return None
    Q = np.array([s.q for s in ss]) - np.asarray(reader.offset, float)
    return Q.mean(axis=0), math.degrees(float(np.ptp(Q, axis=0).max()))


def arm_summary(cfg: arm_mod.ArmConfig) -> list[str]:
    """One line per recorded item, with its date."""
    when = lambda k: cfg.info.get(k, cfg.time or "?")
    roll = cfg.offset[arm_mod.ROLL]
    return [f"wrist zero {math.degrees(roll):+.1f} deg ({when('level_time')})" if "level_time" in cfg.info
            else "wrist zero: not measured",
            f"rest pose ({when('rest_time')})" if cfg.rest is not None else "rest pose: not recorded",
            f"initial pose ({when('initial_time')})" if cfg.initial is not None else "initial pose: not recorded"]


def run_arm(cam, arm, sess: Path) -> str:
    """Step 4: wrist zero, rest pose and initial pose -> arm.json (saved after every record)."""
    R = arm_mod.ROLL
    cfg = arm_mod.load()
    plane = table_plane()
    before = {"offset": cfg.offset.copy(), "rest": cfg.rest, "initial": cfg.initial, "info": dict(cfg.info)}
    part, level, msg, last_fn = "level", [], "", None

    def level_now(q_motor):                        # total joint-6 zero if the jaw is level at this reading
        d, sens = arm_mod.level_offset(cfg.to_model(q_motor), plane)
        return float(cfg.offset[R] + d), sens

    def save(what: str | None = None):
        if what:
            cfg.info[f"{what}_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        cfg.save()
        arm.offset = cfg.offset.copy()             # the reader uses it from now on

    while True:
        f = cam.read(timeout=2.0, newer_than=last_fn)
        last_fn = f.frame_number
        img, _ = big((f.img * 0.4).astype(np.uint8))
        s = arm.latest()
        put(img, "4 ARM: the gripper is remounted every day -> WRIST LEVEL again. ENTER keeps what is saved",
            (10, 24), YELLOW, 0.6)
        for i, (p_, line) in enumerate(zip(ARM_PARTS, arm_summary(cfg))):
            today = ("   today: " + " ".join(f"{math.degrees(v):+.1f}" for v in level)) if p_ == "level" and level else ""
            put(img, ("> " if p_ == part else "  ") + line + today, (20, 56 + 24 * i), GREEN if p_ == part else WHITE, 0.55)
        put(img, ARM_TEXT[part], (10, 140), WHITE, 0.5)
        if s is not None and part == "level":
            corr, sens = level_now(s.q - arm.offset)
            put(img, "fingers straight down: the roll cannot be read, tilt them forward" if sens < NO_SENS else
                f"if both fingertips touch now: wrist zero {math.degrees(corr):+.1f} deg"
                + ("  (fingers steep: a weak reading, tilt them forward)" if sens < MIN_SENS else ""),
                (10, 166), RED if sens < NO_SENS else GREEN if sens >= MIN_SENS else YELLOW, 0.5)
        elif s is None:
            put(img, no_readings(arm), (10, 166), RED)
        put(img, "SPACE record   n / p next / previous part   u undo this part", (10, 192), GREY, 0.45)
        footer(img, msg)
        cv2.imshow(WIN, img)
        key = cv2.waitKey(1) & 0xFF
        if nav(key):
            return nav(key)
        if key in (ord("n"), ord("p")):
            part, msg = ARM_PARTS[(ARM_PARTS.index(part) + (1 if key == ord("n") else -1)) % len(ARM_PARTS)], ""
        if key == ord("u"):                         # back to what was saved before this run
            if part == "level":
                level.clear()
                cfg.offset = before["offset"].copy()
                for k in ("level_time", "level_readings_deg"):
                    cfg.info.pop(k, None)
                    if k in before["info"]:
                        cfg.info[k] = before["info"][k]
            else:
                setattr(cfg, part, before[part])
                cfg.info.pop(f"{part}_time", None)
                if f"{part}_time" in before["info"]:
                    cfg.info[f"{part}_time"] = before["info"][f"{part}_time"]
            save()
            msg = f"{part}: back to the saved value"
        if key == ord(" "):
            r = motor_reading(arm)
            last_fn = None
            if r is None:
                msg = no_readings(arm)
                continue
            q, moved = r
            if part == "level":
                corr, sens = level_now(q)
                if sens < NO_SENS:
                    msg = "the fingers point straight down: the roll cannot be read, tilt them forward, SPACE again"
                    continue
                notes = ([f"moved {moved:.1f} deg while reading"] if moved > STILL_DEG else []) + \
                        (["fingers steep: a weak reading"] if sens < MIN_SENS else [])
                level.append(corr)
                cfg.offset = cfg.offset.copy()
                cfg.offset[R] = float(np.mean(level))
                cfg.info["level_readings_deg"] = [round(math.degrees(v), 2) for v in level]
                save("level")
                spread = math.degrees(float(np.ptp(level)))
                if spread > 3:
                    notes.append("readings more than 3 deg apart - were both fingertips touching?")
                msg = (f"wrist zero {math.degrees(cfg.offset[R]):+.1f} deg; "
                       + ("one more at another spot" if len(level) == 1 else f"readings agree within {spread:.1f} deg")
                       + ("  CHECK: " + "; ".join(notes) + " (u = start the level again)" if notes else ""))
                print(f"wrist zero {math.degrees(cfg.offset[R]):+.2f} deg ({len(level)} reading(s))", flush=True)
            else:
                setattr(cfg, part, q)
                save(part)
                msg = f"{part} pose saved"
                print(msg, flush=True)


# ------------------------------------------------------------------------------------ ball
BALL_WIN = "ball colour"
SLIDERS = (("H lo", 0, 0, 179), ("H hi", 1, 0, 179), ("S lo", 0, 1, 255), ("V lo", 0, 2, 255))   # name, lo/hi, channel, max


def run_ball(cam, sess: Path) -> str:
    """Step 5: the ball colour, seen at the tracking resolution; saved for check / track / catch."""
    from eaiproject.real_intercept import (auto_hsv, camera_calib, hsv_from_selection, hsv_saved_time, load_hsv,
                                           locate_camera, make_tracker, save_hsv)
    saved = tuple(tuple(int(v) for v in x) for x in load_hsv())
    when = hsv_saved_time()
    cur = [list(saved[0]), list(saved[1])]
    seen, msg, last_fn, trk = deque(maxlen=60), "", None, None
    sel = {"start": None, "now": None, "done": None, "used": None}   # mouse: drag a box around the ball
    sized = False                                   # window resized to the image's shape (see below)

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            sel["start"] = sel["now"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and sel["start"] is not None:
            sel["now"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and sel["start"] is not None:
            sel["done"], sel["start"] = (sel["start"], (x, y)), None

    cv2.setMouseCallback(WIN, on_mouse)

    def set_sliders():
        for name, which, ch, _ in SLIDERS:
            cv2.setTrackbarPos(name, BALL_WIN, int(cur[which][ch]))

    def locate():
        nonlocal trk, msg
        try:
            rig_map = rc.RigMap.load(rc.MAP_PATH)
            first = trk is None
            res = locate_camera(cam, rig_map)
            trk = make_tracker(camera_calib(cam, res), rig_map, hsv=saved)
            msg = ("camera located" + (f" - CHECK: {res.warnings[0]}" if res.warnings else ""))[:150]
            for w in res.warnings:
                print(f"camera: {w}", flush=True)
        except rc.CalibrationError as e:
            msg = f"cannot locate the camera: {e} (c = try again)"
            return
        if first:
            cv2.namedWindow(BALL_WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(BALL_WIN, 420, 160)
            for name, which, ch, mx in SLIDERS:
                cv2.createTrackbar(name, BALL_WIN, int(cur[which][ch]), mx, lambda _v: None)

    if rc.MAP_PATH.exists():
        locate()
    else:
        msg = "the ball step needs the table map: steps 1-3 first"
    try:
        while True:
            f = cam.read(timeout=2.0, newer_than=last_fn)
            last_fn = f.frame_number
            img, sc = big(f.img)
            if not sized:
                # The window is 16:9 for the map steps; this image is 4:3. macOS's OpenCV window then shows it
                # with side bars but reports mouse positions as if there were none, so the selection would
                # sample the wrong pixels. Same shape as the image: no bars, mouse and image agree.
                cv2.resizeWindow(WIN, int(720 * img.shape[1] / img.shape[0]), 720)   # fits a laptop screen
                sized = True
            if trk is not None:
                for name, which, ch, _ in SLIDERS:
                    cur[which][ch] = cv2.getTrackbarPos(name, BALL_WIN)
                trk.lo, trk.hi = np.array(cur[0], np.uint8), np.array(cur[1], np.uint8)
                det = trk.detect(f.img)
                seen.append(det is not None)
                if det is not None:
                    cv2.circle(img, ipt(det.uv * sc), int(max(det.radius_px * sc, 4)) + 4, GREEN, 2)
                    put(img, f"ball at x {det.xy[0]:.3f} y {det.xy[1]:+.3f} m", ipt(det.uv * sc + [12, -12]), GREEN, 0.45)
                if trk.last_mask is not None:
                    mk, (ox, oy) = trk.last_mask
                    full = np.zeros(f.img.shape[:2], np.uint8)
                    full[oy:oy + mk.shape[0], ox:ox + mk.shape[1]] = mk
                    inset = cv2.resize(cv2.cvtColor(full, cv2.COLOR_GRAY2BGR), (img.shape[1] // 3, img.shape[0] // 3))
                    img[-inset.shape[0] - 40:-40, -inset.shape[1]:] = inset
                    put(img, "colour mask", (img.shape[1] - inset.shape[1] + 4, img.shape[0] - inset.shape[0] - 46), WHITE, 0.4)
            if sel["used"] is not None:                 # where the last selection sampled
                ux0, uy0, ux1, uy1 = sel["used"]
                cv2.ellipse(img, ipt(((ux0 + ux1) / 2, (uy0 + uy1) / 2)),
                            (max(1, int(abs(ux1 - ux0) / 2)), max(1, int(abs(uy1 - uy0) / 2))), 0, 0, 360, MAGENTA, 1, cv2.LINE_AA)
            if sel["start"] is not None:                # dragging: show the box
                cv2.rectangle(img, sel["start"], sel["now"], MAGENTA, 1, cv2.LINE_AA)
            if sel["done"] is not None and trk is not None:
                (ax, ay), (bx, by) = sel["done"]
                sel["done"] = None
                if abs(bx - ax) < 6 and abs(by - ay) < 6:       # a click: a small disc around it
                    r = 8 * sc
                    ax, ay, bx, by = ax - r, ay - r, ax + r, ay + r
                res_sel = hsv_from_selection(f.img, ax / sc, ay / sc, bx / sc, by / sc)
                if res_sel is None:
                    msg = "selection outside the image: drag a box around the ball"
                else:
                    cur[:] = [list(res_sel[0]), list(res_sel[1])]
                    set_sliders()
                    sel["used"] = (ax, ay, bx, by)
                    msg = (f"from the selection (magenta ellipse = the pixels used): lo {tuple(cur[0])} hi {tuple(cur[1])}"
                           "; roll the ball to check")
            elif sel["done"] is not None:
                sel["done"] = None
                msg = "locate the camera first (c), then select the ball"
            changed = (tuple(cur[0]), tuple(cur[1])) != saved
            put(img, f"5 BALL COLOUR ({cam.width}x{cam.height}, as for tracking): " +
                (f"saved {when}" if when else "default (never saved)") +
                (" - CHANGED, saved when you leave" if changed else " - ENTER keeps it"), (10, 24), YELLOW, 0.55)
            put(img, "MOUSE: drag a box around the ball (or click on it) = its colour.  a = auto-tune.  Then roll it: "
                     "the ring must follow it, the mask must show only the ball", (10, 48), WHITE, 0.45)
            put(img, f"found in {100 * np.mean(seen) if seen else 0:.0f} % of the last 2 s   lo {tuple(cur[0])} hi {tuple(cur[1])}   "
                     "sliders: 'ball colour' window   u saved colour   c locate camera", (10, 70), WHITE, 0.45)
            footer(img, msg)
            cv2.imshow(WIN, img)
            key = cv2.waitKey(1) & 0xFF
            go = nav(key)
            if go:
                if changed:
                    save_hsv(*cur)
                    print(f"ball colour saved: lo {tuple(cur[0])} hi {tuple(cur[1])}", flush=True)
                return go
            if key == ord("a") and trk is not None:
                r = auto_hsv(f.img, trk.table_mask)
                if r is None:
                    msg = "auto-tune needs exactly ONE orange blob on the table (ball still, hands away)"
                else:
                    cur[:] = [list(map(int, r[0])), list(map(int, r[1]))]
                    set_sliders()
                    msg = f"tuned: lo {tuple(cur[0])} hi {tuple(cur[1])}; roll the ball to check"
            if key == ord("u") and trk is not None:
                cur[:] = [list(saved[0]), list(saved[1])]
                set_sliders()
                msg = "back to the saved colour"
            if key == ord("c") and rc.MAP_PATH.exists():
                locate()
    finally:
        cv2.setMouseCallback(WIN, lambda *_: None)    # the other steps share the window
        if sized:
            cv2.resizeWindow(WIN, 1280, 720)
        if trk is not None:
            cv2.destroyWindow(BALL_WIN)


# ----------------------------------------------------------------------------------- solve
def solve(sess: Path) -> tuple[rc.RigSolution | None, str]:
    """Solve from the saved observations; write the report and config/rig_map.json."""
    obs = rc.load_observations(sess)
    obs.sources["session"] = str(sess.relative_to(ROOT)) if sess.is_relative_to(ROOT) else str(sess)
    try:
        sol = rc.solve_rig(obs)
    except rc.CalibrationError as e:
        return None, f"CANNOT SOLVE: {e}"
    out, text = rig_report.save_report(sol, obs, sess)
    sol.rig_map.save(rc.MAP_PATH)
    return sol, text + f"\n\nreport: {out}\nwrote {rc.MAP_PATH}  (commit it)"


def show_result(sess: Path, sol, text: str) -> str:
    """The solve result in the window: the layout picture, the headline numbers and the warnings."""
    if sol is None:
        img = np.full((720, 1280, 3), 32, np.uint8)
        lines = [text]
        put(img, "NO MAP WRITTEN", (20, 70), RED, 0.8, 2)
    else:
        img = cv2.imread(str(sess / "report" / "layout.png"))
        r, ok = sol.report, not sol.warnings
        put(img, "CALIBRATION OK - config/rig_map.json written" if ok else "MAP WRITTEN, BUT CHECK THE WARNINGS",
            (20, 70), GREEN if ok else YELLOW, 0.8, 2)
        lines = [f"reprojection {r['reproj_rms_px']} px (< {rc.OK_PX})   touches {r['touch_rms_mm']} mm (< {rc.OK_MM})   "
                 f"sweep {r['sweep_rms_mm']} mm (< 1.5)"] + [f"! {w}" for w in sol.warnings]
    y = 100
    for line in lines:
        for k in range(0, len(line), 125):                  # wrap long warnings
            put(img, line[k:k + 125], (20, y), YELLOW if line.startswith("!") else WHITE, 0.45)
            y += 20
    put(img, f"full report: {sess / 'report' / 'report.txt'}", (60, img.shape[0] - 64), GREY, 0.45)
    put(img, "ENTER = next: 4 arm, 5 ball      BACKSPACE or 1 / 2 / 3 = go back and fix      q = done",
        (60, img.shape[0] - 42), WHITE, 0.5)
    while True:
        cv2.imshow(WIN, img)
        go = nav(cv2.waitKey(30) & 0xFF)
        if go:
            return go


# ------------------------------------------------------------------------------------ main
def pick_session(args) -> Path:
    """--session if given; a new one (named after today) with --new or when there is none; else the
    newest one (its data stay valid while the markers stay where they are)."""
    today = time.strftime("%Y%m%d")
    if args.session:
        return rc.SESSIONS_DIR / args.session
    last = rc.latest_session()
    if last is not None and not args.new:
        n_v = len(json.loads((last / "views.json").read_text())["views"]) if (last / "views.json").exists() else 0
        n_t = len(json.loads((last / "touches.json").read_text())["touches"]) if (last / "touches.json").exists() else 0
        print(f"continuing session {last.name}: {n_v} views, {n_t} touches, sweep {'yes' if (last / 'sweep.npz').exists() else 'no'}"
              "  (--new starts a fresh table map: do that when the table or a marker moved)")
        return last
    name, k = today, 2
    while (rc.SESSIONS_DIR / name).exists():
        name, k = f"{today}_{k}", k + 1
    return rc.SESSIONS_DIR / name


def start_step(sess: Path, args) -> str:
    """--step; else the daily steps when the table map is done; else the first map step with data missing."""
    if args.step:
        return args.step
    if (sess / "report").exists() and rc.MAP_PATH.exists():
        return "arm"
    n_views = len(json.loads((sess / "views.json").read_text())["views"]) if (sess / "views.json").exists() else 0
    return "views" if n_views < rc.MIN_VIEWS else "touch" if not (sess / "touches.json").exists() else "sweep"


def daily_summary() -> str:
    from eaiproject.real_intercept import hsv_saved_time
    when = hsv_saved_time()
    return "arm: " + "; ".join(arm_summary(arm_mod.load())) + "\nball colour: " + (f"saved {when}" if when else "default")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", action="store_true", help="start a new table map session (the table or a marker moved)")
    ap.add_argument("--session", help="folder under outputs/rig_calib/ (default: the newest)")
    ap.add_argument("--step", choices=("views", "touch", "sweep", "arm", "ball"), help="open at this step")
    ap.add_argument("--solve", action="store_true", help="only solve from the saved observations (no hardware)")
    args = ap.parse_args()
    if args.solve:
        sess = rc.SESSIONS_DIR / args.session if args.session else rc.latest_session()
        if sess is None or not sess.exists():
            print(f"no saved session under {rc.SESSIONS_DIR}")
            return 2
        sol, text = solve(sess)
        print(text)
        return 0 if sol is not None else 1

    sess = pick_session(args)
    sess.mkdir(parents=True, exist_ok=True)
    cam = open_camera()
    if cam is None:
        return 2
    print(f"session: {sess}")
    arm = arm_mod.ArmLink()                         # connects whenever the arm answers on CAN
    if arm.status():
        print(f"{arm.status()} (steps 2-4 need it; the others work without it)", flush=True)
    step, code = start_step(sess, args), 0
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    cli.ctrl_c_quits()
    try:
        while step != "quit":
            # No size switch for the map steps: every switch restarts the camera, and macOS's camera
            # service crashed on that (27 Nov 2026: the Mac froze). They run at the size the server
            # has (./rig camera hd beforehand reads small markers better). Only the ball colour is
            # tuned at the tracking size.
            if step == "ball":
                cam = to_mode(cam, VGA)
            if step == "views":
                go = run_views(cam, sess)
            elif step == "touch":
                go = run_touch(cam, arm, sess)
            elif step == "sweep":
                go = run_sweep(cam, arm, sess)
            elif step == "arm":
                go = run_arm(cam, arm, sess)
            elif step == "ball":
                go = run_ball(cam, sess)
            else:
                sol, text = solve(sess)
                print("\n" + text, flush=True)
                code = 0 if sol is not None else 1
                go = show_result(sess, sol, text)
            i = STEPS.index(step)
            if go in STEPS or go == "quit":
                step = go
            elif go == "next":
                step = "quit" if step == "ball" else STEPS[i + 1]
            else:                                                # back; from the arm step not into a new solve
                step = "sweep" if step == "arm" else STEPS[max(i - 1, 0)]
    finally:
        cv2.destroyAllWindows()
        cam.close()
        arm.close()
    print(daily_summary())
    return code


if __name__ == "__main__":
    cli.run(main)
