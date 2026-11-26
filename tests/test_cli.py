"""Ctrl+C in the real-rig scripts (eaiproject/cli.py), with real SIGINTs sent to this process.

    .venv/bin/python -m unittest tests.test_cli -v
"""
from __future__ import annotations

import io
import os
import signal
import sys
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eaiproject import cli  # noqa: E402


class CtrlCTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(signal.signal, signal.SIGINT, signal.getsignal(signal.SIGINT))
        self.addCleanup(setattr, cli, "_pressed", False)

    def test_first_press_asks_the_loop_to_quit_second_stops(self):
        cli.ctrl_c_quits()
        self.assertFalse(cli.ctrl_c())
        os.kill(os.getpid(), signal.SIGINT)
        self.assertTrue(cli.ctrl_c())                       # no exception: the loop quits as with q
        with self.assertRaises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.05)

    def test_run_ends_quietly(self):
        def main():
            raise KeyboardInterrupt

        out = io.StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit) as cm:
            cli.run(main)
        self.assertEqual(cm.exception.code, 130)
        self.assertEqual(out.getvalue().strip(), "stopped (Ctrl+C)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
