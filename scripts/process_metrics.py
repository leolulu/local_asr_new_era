"""Low-overhead child-process resource monitoring for Windows."""

from __future__ import annotations

import ctypes
import os
import threading
import time
from ctypes import wintypes
from typing import Any


DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.25


class _FileTime(ctypes.Structure):
    _fields_ = [
        ("low", wintypes.DWORD),
        ("high", wintypes.DWORD),
    ]


class _ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("page_fault_count", wintypes.DWORD),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
        ("private_usage", ctypes.c_size_t),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("read_operation_count", ctypes.c_ulonglong),
        ("write_operation_count", ctypes.c_ulonglong),
        ("other_operation_count", ctypes.c_ulonglong),
        ("read_transfer_count", ctypes.c_ulonglong),
        ("write_transfer_count", ctypes.c_ulonglong),
        ("other_transfer_count", ctypes.c_ulonglong),
    ]


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("length", wintypes.DWORD),
        ("memory_load", wintypes.DWORD),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("available_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    ]


def _filetime_seconds(value: _FileTime) -> float:
    ticks = (int(value.high) << 32) | int(value.low)
    return ticks / 10_000_000


class ProcessResourceMonitor:
    """Sample one Windows process without changing how that process is executed."""

    def __init__(
        self,
        process_id: int,
        *,
        sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
    ) -> None:
        self._process_id = process_id
        self._sample_interval_seconds = sample_interval_seconds
        self._logical_processors = os.cpu_count() or 1
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle: int | None = None
        self._unavailable_reason: str | None = None
        self._previous_sample_time: float | None = None
        self._previous_cpu_seconds: float | None = None
        self._user_seconds = 0.0
        self._kernel_seconds = 0.0
        self._peak_logical_cores = 0.0
        self._peak_machine_percent = 0.0
        self._peak_working_set_bytes = 0
        self._working_set_bytes = 0
        self._peak_private_bytes_sampled = 0
        self._private_bytes = 0
        self._page_fault_count = 0
        self._io: dict[str, int] = {}
        self._system_memory_start: dict[str, int] | None = None
        self._system_memory_finish: dict[str, int] | None = None
        self._minimum_available_physical_bytes: int | None = None
        self._maximum_memory_load_percent = 0

        if os.name != "nt":
            self._unavailable_reason = "进程资源监控当前只支持 Windows"
            return

        try:
            self._configure_windows_api()
            process_query_information = 0x0400
            process_vm_read = 0x0010
            handle = self._open_process(
                process_query_information | process_vm_read,
                False,
                process_id,
            )
            if not handle:
                error_code = ctypes.get_last_error()
                raise OSError(error_code, "OpenProcess failed")
            self._handle = int(handle)
        except Exception as error:
            self._unavailable_reason = str(error)

    def start(self) -> None:
        if self._handle is None:
            return
        self._sample()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self, *, process_wall_seconds: float) -> dict[str, Any]:
        if self._handle is None:
            return {
                "available": False,
                "reason": self._unavailable_reason or "无法打开子进程",
            }

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._sample(update_current_memory=False)
        try:
            self._system_memory_finish = self._read_system_memory()
        except Exception:
            self._system_memory_finish = None
        self._close_handle(self._handle)
        self._handle = None

        total_cpu_seconds = self._user_seconds + self._kernel_seconds
        average_logical_cores = (
            total_cpu_seconds / process_wall_seconds if process_wall_seconds > 0 else 0.0
        )
        average_machine_percent = (
            average_logical_cores / self._logical_processors * 100
        )

        return {
            "available": True,
            "scope": "联合 VAD+ASR 子进程的完整生命周期",
            "sampling_interval_ms": round(self._sample_interval_seconds * 1000),
            "cpu": {
                "logical_processors": self._logical_processors,
                "process_user_seconds": round(self._user_seconds, 6),
                "process_kernel_seconds": round(self._kernel_seconds, 6),
                "process_total_seconds": round(total_cpu_seconds, 6),
                "average_logical_cores_used": round(average_logical_cores, 4),
                "average_machine_capacity_percent": round(
                    average_machine_percent, 4
                ),
                "peak_logical_cores_used_sampled": round(
                    self._peak_logical_cores, 4
                ),
                "peak_machine_capacity_percent_sampled": round(
                    self._peak_machine_percent, 4
                ),
            },
            "memory": {
                "peak_working_set_bytes": self._peak_working_set_bytes,
                "working_set_at_last_sample_bytes": self._working_set_bytes,
                "peak_private_bytes_sampled": self._peak_private_bytes_sampled,
                "private_bytes_at_last_sample": self._private_bytes,
                "page_fault_count": self._page_fault_count,
                "page_fault_note": (
                    "包含软缺页和硬缺页，不能直接解释为磁盘换页次数。"
                ),
            },
            "io": self._io,
            "system_memory": self._build_system_memory_result(),
        }

    def _configure_windows_api(self) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)

        self._open_process = kernel32.OpenProcess
        self._open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self._open_process.restype = wintypes.HANDLE

        self._close_handle = kernel32.CloseHandle
        self._close_handle.argtypes = [wintypes.HANDLE]
        self._close_handle.restype = wintypes.BOOL

        self._get_process_times = kernel32.GetProcessTimes
        self._get_process_times.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
        ]
        self._get_process_times.restype = wintypes.BOOL

        self._get_process_io_counters = kernel32.GetProcessIoCounters
        self._get_process_io_counters.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_IoCounters),
        ]
        self._get_process_io_counters.restype = wintypes.BOOL

        self._global_memory_status_ex = kernel32.GlobalMemoryStatusEx
        self._global_memory_status_ex.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
        self._global_memory_status_ex.restype = wintypes.BOOL

        self._get_process_memory_info = psapi.GetProcessMemoryInfo
        self._get_process_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_ProcessMemoryCountersEx),
            wintypes.DWORD,
        ]
        self._get_process_memory_info.restype = wintypes.BOOL

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self._sample_interval_seconds):
            self._sample()

    def _sample(self, *, update_current_memory: bool = True) -> None:
        if self._handle is None:
            return
        try:
            now = time.perf_counter()
            cpu = self._read_cpu_times()
            if cpu is not None:
                user_seconds, kernel_seconds = cpu
                total_cpu_seconds = user_seconds + kernel_seconds
                if (
                    self._previous_sample_time is not None
                    and self._previous_cpu_seconds is not None
                ):
                    elapsed = now - self._previous_sample_time
                    if elapsed > 0:
                        used_cores = max(
                            0.0,
                            (total_cpu_seconds - self._previous_cpu_seconds) / elapsed,
                        )
                        self._peak_logical_cores = max(
                            self._peak_logical_cores, used_cores
                        )
                        self._peak_machine_percent = max(
                            self._peak_machine_percent,
                            used_cores / self._logical_processors * 100,
                        )
                self._previous_sample_time = now
                self._previous_cpu_seconds = total_cpu_seconds
                self._user_seconds = max(self._user_seconds, user_seconds)
                self._kernel_seconds = max(self._kernel_seconds, kernel_seconds)

            memory = self._read_process_memory()
            if memory is not None:
                self._peak_working_set_bytes = max(
                    self._peak_working_set_bytes,
                    memory["peak_working_set_bytes"],
                )
                if update_current_memory:
                    self._working_set_bytes = memory["working_set_bytes"]
                self._peak_private_bytes_sampled = max(
                    self._peak_private_bytes_sampled,
                    memory["private_bytes"],
                )
                if update_current_memory:
                    self._private_bytes = memory["private_bytes"]
                self._page_fault_count = max(
                    self._page_fault_count,
                    memory["page_fault_count"],
                )

            io = self._read_io_counters()
            if io is not None:
                self._io = io

            system_memory = self._read_system_memory()
            if system_memory is not None:
                if self._system_memory_start is None:
                    self._system_memory_start = system_memory
                available = system_memory["available_physical_bytes"]
                if self._minimum_available_physical_bytes is None:
                    self._minimum_available_physical_bytes = available
                else:
                    self._minimum_available_physical_bytes = min(
                        self._minimum_available_physical_bytes, available
                    )
                self._maximum_memory_load_percent = max(
                    self._maximum_memory_load_percent,
                    system_memory["memory_load_percent"],
                )
        except Exception:
            # Observability must never make recognition fail.
            return

    def _read_cpu_times(self) -> tuple[float, float] | None:
        creation = _FileTime()
        exit_time = _FileTime()
        kernel = _FileTime()
        user = _FileTime()
        if not self._get_process_times(
            self._handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            return None
        return _filetime_seconds(user), _filetime_seconds(kernel)

    def _read_process_memory(self) -> dict[str, int] | None:
        counters = _ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        if not self._get_process_memory_info(
            self._handle,
            ctypes.byref(counters),
            counters.cb,
        ):
            return None
        return {
            "peak_working_set_bytes": int(counters.peak_working_set_size),
            "working_set_bytes": int(counters.working_set_size),
            "private_bytes": int(counters.private_usage),
            "page_fault_count": int(counters.page_fault_count),
        }

    def _read_io_counters(self) -> dict[str, int] | None:
        counters = _IoCounters()
        if not self._get_process_io_counters(self._handle, ctypes.byref(counters)):
            return None
        return {
            "read_bytes": int(counters.read_transfer_count),
            "write_bytes": int(counters.write_transfer_count),
            "other_bytes": int(counters.other_transfer_count),
            "read_operations": int(counters.read_operation_count),
            "write_operations": int(counters.write_operation_count),
            "other_operations": int(counters.other_operation_count),
        }

    def _read_system_memory(self) -> dict[str, int] | None:
        status = _MemoryStatusEx()
        status.length = ctypes.sizeof(status)
        if not self._global_memory_status_ex(ctypes.byref(status)):
            return None
        return {
            "total_physical_bytes": int(status.total_physical),
            "available_physical_bytes": int(status.available_physical),
            "memory_load_percent": int(status.memory_load),
        }

    def _build_system_memory_result(self) -> dict[str, Any] | None:
        if self._system_memory_start is None:
            return None
        finish = self._system_memory_finish or self._system_memory_start
        return {
            "total_physical_bytes": self._system_memory_start[
                "total_physical_bytes"
            ],
            "available_at_start_bytes": self._system_memory_start[
                "available_physical_bytes"
            ],
            "minimum_available_bytes_sampled": (
                self._minimum_available_physical_bytes
                if self._minimum_available_physical_bytes is not None
                else self._system_memory_start["available_physical_bytes"]
            ),
            "available_at_finish_bytes": finish["available_physical_bytes"],
            "maximum_memory_load_percent_sampled": self._maximum_memory_load_percent,
            "scope_note": "系统级数据可能同时受到其他进程影响。",
        }
