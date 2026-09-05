"""Track child processes so a cancelled batch can stop active work."""

from __future__ import annotations

import subprocess
import threading
from concurrent.futures import CancelledError
from contextlib import contextmanager
from typing import Any, Iterator, Sequence


class ProcessGroup:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self._processes: set[subprocess.Popen] = set()

    def check_cancelled(self) -> None:
        if self._cancelled.is_set():
            raise CancelledError()

    def cancel(self) -> None:
        with self._lock:
            self._cancelled.set()
            for process in self._processes:
                if process.poll() is None:
                    process.kill()

    @contextmanager
    def open(self, command: Sequence[str], **kwargs: Any) -> Iterator[subprocess.Popen]:
        with self._lock:
            self.check_cancelled()
            process = subprocess.Popen(command, **kwargs)
            self._processes.add(process)
        try:
            yield process
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            for stream in (process.stdout, process.stderr, process.stdin):
                if stream is not None:
                    stream.close()
            with self._lock:
                self._processes.discard(process)

    def run(self, command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess:
        with self.open(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs
        ) as process:
            stdout, stderr = process.communicate()
            self.check_cancelled()
            return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
