from __future__ import annotations

import subprocess
import sys
import unittest
from concurrent.futures import CancelledError

from scripts.process_group import ProcessGroup


class ProcessGroupTests(unittest.TestCase):
    def test_run_captures_both_streams_and_return_code(self) -> None:
        result = ProcessGroup().run(
            [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"],
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "out")
        self.assertEqual(result.stderr.strip(), "err")
        self.assertEqual(result.returncode, 3)

    def test_cancel_stops_active_processes_and_prevents_new_ones(self) -> None:
        group = ProcessGroup()
        command = [sys.executable, "-c", "import time; time.sleep(60)"]
        with group.open(command) as first, group.open(command) as second:
            self.assertIsNone(first.poll())
            self.assertIsNone(second.poll())
            group.cancel()
            self.assertIsNotNone(first.wait(timeout=5))
            self.assertIsNotNone(second.wait(timeout=5))
            with self.assertRaises(CancelledError):
                with group.open(command):
                    self.fail("cancelled group started another process")

    def test_exception_reaps_child_and_closes_pipes(self) -> None:
        group = ProcessGroup()
        with self.assertRaisesRegex(RuntimeError, "abort"):
            with group.open(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ) as process:
                raise RuntimeError("abort")
        self.assertIsNotNone(process.returncode)
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)


if __name__ == "__main__":
    unittest.main()
