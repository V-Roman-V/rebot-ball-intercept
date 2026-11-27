#!/bin/bash
# Start the RealSense frame server as root (macOS password dialog), once per session. Usually called
# by ./rig, which also switches a running server between sizes without a password (./rig camera hd|vga).
# Root cannot read ~/Documents (TCC), so the server runs from ~/realsense_tools with its own venv.
# Extra arguments go to rs_server.py, e.g.:  scripts/real/start_camera.sh --fps 60 --exposure 3000
# Stop it with:  ./rig camera stop
set -e
T="$HOME/realsense_tools"
# Apps that have the RealSense open, other than macOS's own camera service and our server. A browser
# or a call app holding it leaves the server without frames (27 Nov 2026: Yandex).
holders() {
  ioreg -l -w0 -r -n "Intel(R) RealSense(TM) Depth Camera 435i" 2>/dev/null | grep IOUserClientCreator \
    | sed -E 's/.*"pid ([0-9]+), ([^"]+)".*/\2 (pid \1)/' | grep -v -E '^(UVCAssistant|WindowServer|python[0-9.]*) ' | sort -u | paste -sd, -
}
advice() {
  H="$(holders)"
  if [ -n "$H" ]; then
    echo "The camera is held by: $H"
    echo "  -> quit it completely (Cmd+Q), unplug and replug the RealSense, then start again."
  else
    echo "  -> quit apps that may use a camera (browsers, Teams, Zoom, Telegram, FaceTime), unplug and replug the"
    echo "     RealSense (a USB 3 port of the Mac, not a hub if possible), then start again."
  fi
}
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$T"
cp "$HERE/rs_server.py" "$T/rs_server.py"
if [ ! -x "$T/.venv/bin/python" ]; then
  echo "first run: setting up $T/.venv (numpy, pyrealsense2-macosx 2.54.2)"
  UV="$(command -v uv || true)"
  [ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ] && UV="$HOME/.local/bin/uv"
  if [ -n "$UV" ]; then
    "$UV" venv --python 3.12 "$T/.venv" -q
    "$UV" pip install -q --python "$T/.venv/bin/python" numpy "pyrealsense2-macosx==2.54.2"
  elif command -v python3.12 >/dev/null; then
    python3.12 -m venv "$T/.venv"
    "$T/.venv/bin/pip" install -q numpy "pyrealsense2-macosx==2.54.2"
  else
    echo "need uv (brew install uv) or python3.12 to set up the camera server"; exit 1
  fi
fi
if nc -z 127.0.0.1 8770 2>/dev/null; then
  echo "camera server already running:"; head -n 1 "$T/rs_server.log" 2>/dev/null
  echo "(another size: ./rig camera hd | vga switches it without a password)"; exit 0
fi
ARGS="$*"
# A stuck earlier server (camera held, port closed) is killed by the root command below. Only a
# number that is still an rs_server.py process is passed on: this value ends up in a root shell.
OLD="$(cat "$T/rs_server.pid" 2>/dev/null || true)"
case "$OLD" in ''|*[!0-9]*) OLD="" ;; esac
if [ -n "$OLD" ] && ! ps -p "$OLD" -o command= 2>/dev/null | grep -q "rs_server.py"; then OLD=""; fi
rm -f "$T/rs_server.log" "$T/rs_server.pid"      # root's files, but in our folder: we may delete them (not write them)
H="$(holders)"
[ -n "$H" ] && echo "CHECK: the camera is already open in: $H. If the start fails, quit it (Cmd+Q) and replug the camera."
echo "starting the camera server as root (macOS asks for your password; then up to ~20 s) ..."
# The server is started directly (no subshell around it), with all its output in the log, so that
# the password helper returns at once. The helper still runs in the background, and this script
# watches the port, so a helper that does not return cannot hang the terminal.
KILL_OLD=""
[ -n "$OLD" ] && KILL_OLD="kill -9 $OLD 2>/dev/null; sleep 1;"
osascript -e "do shell script \"$KILL_OLD cd $T; .venv/bin/python rs_server.py $ARGS < /dev/null > $T/rs_server.log 2>&1 &\" with administrator privileges" &
OSA=$!
trap 'kill "$OSA" 2>/dev/null; exit 130' INT TERM   # Ctrl+C while waiting: close the password helper too
started=""
for i in $(seq 1 600); do                       # up to 2.5 min: time to type the password
  if nc -z 127.0.0.1 8770 2>/dev/null; then started=1; break; fi
  if [ -s "$T/rs_server.pid" ] && ! ps -p "$(cat "$T/rs_server.pid")" >/dev/null 2>&1; then break; fi   # server ended
  if ! kill -0 "$OSA" 2>/dev/null && [ ! -s "$T/rs_server.pid" ]; then   # helper done, no server: cancelled?
    [ -z "$done_at" ] && done_at=$i
    [ $((i - done_at)) -gt 20 ] && break
  fi
  sleep 0.25
done
kill "$OSA" 2>/dev/null                          # the helper only; the server runs on by itself
if [ -n "$started" ]; then
  sleep 0.3; cat "$T/rs_server.log"; exit 0
fi
echo "server did not start; log:"
if [ -s "$T/rs_server.log" ]; then
  grep -v -E '^(Thread|Current thread|  File|Extension modules)' "$T/rs_server.log" || true
  grep -q '^Timeout' "$T/rs_server.log" && echo "The server got stuck talking to the camera and ended itself."
  advice
else
  echo "(no log: password cancelled?)"
fi
exit 1
