"""Ctrl+C in the real-rig scripts stops the program like closing its window: no traceback.

    cli.ctrl_c_quits()                # before the window loop: from now on Ctrl+C = q
    while True:
        ...
        if key == ord("q") or window_closed(win) or cli.ctrl_c():
            break                     # the code after the loop saves and closes as usual

    if __name__ == "__main__":
        cli.run(main)                 # a Ctrl+C before the loop (setup): "stopped (Ctrl+C)"

A second Ctrl+C stops at once. The powered scripts get theirs from eaiproject.arm.powered():
there the first Ctrl+C holds the arm and a second one drops it and stops. Ctrl+C alone always
stops a program.
"""
from __future__ import annotations

import signal
import sys

_pressed = False


def ctrl_c_quits() -> None:
    """From now on Ctrl+C asks the window loop to quit (it checks ctrl_c()); a second one stops at once."""
    global _pressed
    _pressed = False

    def handler(*_):
        global _pressed
        if _pressed:
            raise KeyboardInterrupt
        _pressed = True

    signal.signal(signal.SIGINT, handler)


def ctrl_c() -> bool:
    """Ctrl+C was pressed since ctrl_c_quits()."""
    return _pressed


def run(main, *args) -> None:
    """sys.exit(main(*args)); a Ctrl+C (or Ctrl+D at a prompt) that reaches here ends it quietly.
    The scripts' first lines make a Ctrl+C during their imports exit quietly; from here on it is the
    normal KeyboardInterrupt again (eaiproject.arm.powered() holds the arm on the first one)."""
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        code = main(*args)
    except (KeyboardInterrupt, EOFError):
        print("\nstopped (Ctrl+C)", flush=True)
        code = 130
    sys.exit(code)
