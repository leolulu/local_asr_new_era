"""Unified file and directory media-to-text entry point for local ASR models."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import threading
import time
import wave
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator
from uuid import uuid4

try:
    from .asr_common import (
        DEFAULT_NUM_THREADS,
        DEFAULT_VAD_MAX_SPEECH_DURATION,
        DEFAULT_VAD_MIN_SILENCE_DURATION,
        DEFAULT_VAD_MIN_SPEECH_DURATION,
        DEFAULT_VAD_NEG_THRESHOLD,
        DEFAULT_VAD_NUM_THREADS,
        DEFAULT_VAD_THRESHOLD,
        DEFAULT_VAD_WINDOW_SIZE,
        add_runtime_arguments,
        add_vad_arguments,
        resolve_file,
        vad_kwargs_from_args,
    )
    from .convert_audio import convert_audio_to_wav
    from .process_group import ProcessGroup
    from .sensevoice_asr import run_sensevoice
    from .zipformer_asr import run_zipformer
except ImportError:
    from asr_common import (
        DEFAULT_NUM_THREADS,
        DEFAULT_VAD_MAX_SPEECH_DURATION,
        DEFAULT_VAD_MIN_SILENCE_DURATION,
        DEFAULT_VAD_MIN_SPEECH_DURATION,
        DEFAULT_VAD_NEG_THRESHOLD,
        DEFAULT_VAD_NUM_THREADS,
        DEFAULT_VAD_THRESHOLD,
        DEFAULT_VAD_WINDOW_SIZE,
        add_runtime_arguments,
        add_vad_arguments,
        resolve_file,
        vad_kwargs_from_args,
    )
    from convert_audio import convert_audio_to_wav
    from process_group import ProcessGroup
    from sensevoice_asr import run_sensevoice
    from zipformer_asr import run_zipformer


MODEL_ORDER = ("sensevoice", "zipformer")
SUPPORTED_MODELS = frozenset(MODEL_ORDER)
ALL_MODELS_CHOICE = "all"
SUPPORTED_AUDIO_EXTENSIONS = frozenset(
    {
        ".aac",
        ".ac3",
        ".aif",
        ".aiff",
        ".amr",
        ".ape",
        ".au",
        ".caf",
        ".dts",
        ".eac3",
        ".flac",
        ".m4a",
        ".m4b",
        ".mka",
        ".mp2",
        ".mp3",
        ".oga",
        ".ogg",
        ".opus",
        ".snd",
        ".spx",
        ".wav",
        ".wma",
    }
)
SUPPORTED_VIDEO_EXTENSIONS = frozenset(
    {
        ".3gp",
        ".asf",
        ".avi",
        ".divx",
        ".flv",
        ".m2ts",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".mts",
        ".ogv",
        ".rm",
        ".rmvb",
        ".ts",
        ".vob",
        ".webm",
        ".wmv",
    }
)
SUPPORTED_MEDIA_EXTENSIONS = SUPPORTED_AUDIO_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS
StatusCallback = Callable[[str], None]
ProgressCallback = Callable[[float, float, int], None]


@dataclass(frozen=True)
class PreparedAudio:
    source_path: Path
    wav_path: Path
    conversion_performed: bool
    conversion_reason: str
    input_preparation_wall_seconds: float
    audio_conversion_wall_seconds: float
    audio_inspection_wall_seconds: float
    audio_metadata: dict[str, Any] | None


class ChineseHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Keep examples readable while showing complete Chinese option help."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("max_help_position", 32)
        kwargs.setdefault("width", 100)
        super().__init__(*args, **kwargs)


class ChineseArgumentParser(argparse.ArgumentParser):
    def format_usage(self) -> str:
        return super().format_usage().replace("usage: ", "用法: ", 1)

    def format_help(self) -> str:
        return super().format_help().replace("usage: ", "用法: ", 1)


class ConsoleProgress:
    """Render status and progress on stderr without contaminating ASR output."""

    def __init__(self, *, prefix: str = "", output_lock: Any = None) -> None:
        self._prefix = prefix
        self._interactive = sys.stderr.isatty() and not prefix
        self._lock = output_lock if output_lock is not None else threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active = False
        self._label = "准备处理"
        self._processed_seconds = 0.0
        self._total_seconds = 0.0
        self._segment_count = 0
        self._started_at = 0.0
        self._last_width = 0
        self._last_reported_bucket = -1
        self._spinner_index = 0

    def start(self) -> None:
        with self._lock:
            self._active = True
            self._started_at = time.perf_counter()
        if self._interactive:
            self._thread = threading.Thread(target=self._render_loop, daemon=True)
            self._thread.start()

    def status(self, message: str) -> None:
        with self._lock:
            self._label = message
            if self._interactive:
                self._clear_line_locked()
                print(f"{self._prefix}[状态] {message}", file=sys.stderr, flush=True)
            else:
                print(f"{self._prefix}[状态] {message}", file=sys.stderr, flush=True)

    def update(self, processed_seconds: float, total_seconds: float, segments: int) -> None:
        with self._lock:
            self._processed_seconds = processed_seconds
            self._total_seconds = total_seconds
            self._segment_count = segments
            if processed_seconds == 0 and total_seconds == 0 and segments == 0:
                self._last_reported_bucket = -1
                return
            if self._interactive:
                return
            percent = 100 * processed_seconds / total_seconds if total_seconds else 0
            bucket = min(9, int(percent // 10))
            if bucket <= self._last_reported_bucket:
                return
            self._last_reported_bucket = bucket
            print(
                f"{self._prefix}[进度] {percent:5.1f}% | 已处理 {processed_seconds:.1f}/"
                f"{total_seconds:.1f} 秒 | {segments} 个分段",
                file=sys.stderr,
                flush=True,
            )

    def finish(self) -> None:
        with self._lock:
            if not self._active:
                return
            self._processed_seconds = self._total_seconds
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            if self._interactive:
                self._render_locked(completed=True)
                print(file=sys.stderr, flush=True)
            elapsed = time.perf_counter() - self._started_at
            print(
                f"{self._prefix}[完成] 识别完成，共 {self._segment_count} 个分段，耗时 {elapsed:.1f} 秒",
                file=sys.stderr,
                flush=True,
            )
            self._active = False

    def abort(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            if self._interactive:
                self._clear_line_locked()
            self._active = False

    def _render_loop(self) -> None:
        while not self._stop_event.wait(0.2):
            with self._lock:
                if self._active:
                    self._render_locked(completed=False)

    def _render_locked(self, *, completed: bool) -> None:
        elapsed = time.perf_counter() - self._started_at
        if self._total_seconds > 0:
            fraction = min(1.0, self._processed_seconds / self._total_seconds)
            if completed:
                fraction = 1.0
            filled = int(30 * fraction)
            bar = "█" * filled + "─" * (30 - filled)
            line = (
                f"[进度] [{bar}] {fraction * 100:6.2f}% | "
                f"{self._processed_seconds:.1f}/{self._total_seconds:.1f} 秒 | "
                f"{self._segment_count} 段 | {elapsed:.1f} 秒"
            )
        else:
            spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[self._spinner_index % 10]
            self._spinner_index += 1
            line = f"[进度] {spinner} {self._label} | 已耗时 {elapsed:.1f} 秒"
        self._clear_line_locked()
        sys.stderr.write(line)
        sys.stderr.flush()
        self._last_width = len(line)

    def _clear_line_locked(self) -> None:
        if self._last_width:
            sys.stderr.write("\r" + " " * self._last_width + "\r")
            sys.stderr.flush()
            self._last_width = 0


def _is_model_ready_wav(path: Path) -> bool:
    if path.suffix.lower() != ".wav":
        return False
    try:
        with wave.open(str(path), "rb") as wav_file:
            return (
                wav_file.getnchannels() == 1
                and wav_file.getsampwidth() == 2
                and wav_file.getframerate() == 16_000
                and wav_file.getcomptype() == "NONE"
            )
    except (EOFError, OSError, wave.Error):
        return False


def _inspect_wav(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as wav_file:
        frame_count = wav_file.getnframes()
        sample_rate_hz = wav_file.getframerate()
        return {
            "duration_seconds": round(frame_count / sample_rate_hz, 6),
            "sample_rate_hz": sample_rate_hz,
            "channels": wav_file.getnchannels(),
            "sample_width_bits": wav_file.getsampwidth() * 8,
            "frame_count": frame_count,
            "compression_type": wav_file.getcomptype(),
        }


def _format_local_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="milliseconds")


def _describe_file(path: str | Path) -> dict[str, Any]:
    resolved_path = Path(path).expanduser().resolve()
    stat = resolved_path.stat()
    return {
        "path": str(resolved_path),
        "name": resolved_path.name,
        "size_bytes": stat.st_size,
        "modified_at": _format_local_timestamp(stat.st_mtime),
    }


@contextmanager
def _prepared_wav(
    input_path: str | Path,
    *,
    ffmpeg_executable: str | Path,
    status_callback: StatusCallback | None = None,
    collect_observability: bool = False,
    process_group: ProcessGroup | None = None,
) -> Iterator[PreparedAudio]:
    preparation_started = time.perf_counter() if collect_observability else None
    if status_callback is not None:
        status_callback("正在检查输入文件和音频格式")
    source = resolve_file(input_path, "Input media file")
    if _is_model_ready_wav(source):
        preparation_elapsed = (
            time.perf_counter() - preparation_started
            if preparation_started is not None
            else 0.0
        )
        inspection_started = time.perf_counter() if collect_observability else None
        audio_metadata = _inspect_wav(source) if collect_observability else None
        inspection_elapsed = (
            time.perf_counter() - inspection_started
            if inspection_started is not None
            else 0.0
        )
        if status_callback is not None:
            status_callback("输入已经是模型可用的 WAV，跳过 FFmpeg 转换")
        yield PreparedAudio(
            source_path=source,
            wav_path=source,
            conversion_performed=False,
            conversion_reason="already_compatible_wav",
            input_preparation_wall_seconds=preparation_elapsed,
            audio_conversion_wall_seconds=0.0,
            audio_inspection_wall_seconds=inspection_elapsed,
            audio_metadata=audio_metadata,
        )
        return

    preparation_elapsed = (
        time.perf_counter() - preparation_started
        if preparation_started is not None
        else 0.0
    )
    with tempfile.TemporaryDirectory(prefix="local-asr-") as temp_dir:
        wav_path = Path(temp_dir) / "input.wav"
        if status_callback is not None:
            status_callback("正在使用 FFmpeg 转换为 16 kHz 单声道 PCM WAV")
        conversion_started = time.perf_counter() if collect_observability else None
        convert_audio_to_wav(
            source,
            wav_path,
            sample_rate=16_000,
            ffmpeg_executable=ffmpeg_executable,
            process_group=process_group,
        )
        conversion_elapsed = (
            time.perf_counter() - conversion_started
            if conversion_started is not None
            else 0.0
        )
        inspection_started = time.perf_counter() if collect_observability else None
        audio_metadata = _inspect_wav(wav_path) if collect_observability else None
        inspection_elapsed = (
            time.perf_counter() - inspection_started
            if inspection_started is not None
            else 0.0
        )
        if status_callback is not None:
            status_callback("音频转换完成")
        yield PreparedAudio(
            source_path=source,
            wav_path=wav_path,
            conversion_performed=True,
            conversion_reason="input_not_model_ready_wav",
            input_preparation_wall_seconds=preparation_elapsed,
            audio_conversion_wall_seconds=conversion_elapsed,
            audio_inspection_wall_seconds=inspection_elapsed,
            audio_metadata=audio_metadata,
        )


def run_asr(
    input_path: str | Path,
    *,
    model: str = "sensevoice",
    ffmpeg_executable: str | Path = "ffmpeg",
    num_threads: int = DEFAULT_NUM_THREADS,
    debug: bool = False,
    provider: str = "cpu",
    language: str = "auto",
    use_itn: bool = True,
    decoding_method: str = "greedy_search",
    vad_threshold: float = DEFAULT_VAD_THRESHOLD,
    vad_min_silence_duration: float = DEFAULT_VAD_MIN_SILENCE_DURATION,
    vad_min_speech_duration: float = DEFAULT_VAD_MIN_SPEECH_DURATION,
    vad_max_speech_duration: float = DEFAULT_VAD_MAX_SPEECH_DURATION,
    vad_window_size: int = DEFAULT_VAD_WINDOW_SIZE,
    vad_neg_threshold: float = DEFAULT_VAD_NEG_THRESHOLD,
    vad_num_threads: int = DEFAULT_VAD_NUM_THREADS,
    vad_debug: bool = False,
    vad_provider: str = "cpu",
    status_callback: StatusCallback | None = None,
    progress_callback: ProgressCallback | None = None,
    collect_observability: bool = False,
    process_group: ProcessGroup | None = None,
) -> dict[str, Any]:
    """Convert one media file as needed, then recognize it with the chosen model."""
    task_started = time.perf_counter() if collect_observability else None
    task_started_at = datetime.now().astimezone() if collect_observability else None
    normalized_model = model.lower()
    if normalized_model not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(f"Unsupported model {model!r}. Expected one of: {supported}")

    vad_options = {
        "vad_threshold": vad_threshold,
        "vad_min_silence_duration": vad_min_silence_duration,
        "vad_min_speech_duration": vad_min_speech_duration,
        "vad_max_speech_duration": vad_max_speech_duration,
        "vad_window_size": vad_window_size,
        "vad_neg_threshold": vad_neg_threshold,
        "vad_num_threads": vad_num_threads,
        "vad_debug": vad_debug,
        "vad_provider": vad_provider,
    }
    with _prepared_wav(
        input_path,
        ffmpeg_executable=ffmpeg_executable,
        status_callback=status_callback,
        collect_observability=collect_observability,
        process_group=process_group,
    ) as prepared_audio:
        if status_callback is not None:
            status_callback(f"正在加载 {normalized_model} 并执行 VAD 分段识别")
        if normalized_model == "sensevoice":
            result = run_sensevoice(
                prepared_audio.wav_path,
                num_threads=num_threads,
                language=language,
                use_itn=use_itn,
                debug=debug,
                provider=provider,
                use_vad=True,
                progress_callback=progress_callback,
                collect_observability=collect_observability,
                process_group=process_group,
                **vad_options,
            )
        else:
            result = run_zipformer(
                prepared_audio.wav_path,
                num_threads=num_threads,
                decoding_method=decoding_method,
                status_callback=status_callback,
                debug=debug,
                provider=provider,
                progress_callback=progress_callback,
                collect_observability=collect_observability,
                process_group=process_group,
                **vad_options,
            )

    if status_callback is not None:
        status_callback("模型推理与分段识别完成")
    final_result = {"model": normalized_model, **result}
    if collect_observability:
        internal_observability = final_result.pop("_observability", {})
        task_finished_at = datetime.now().astimezone()
        total_wall_seconds = time.perf_counter() - (task_started or 0.0)
        metadata_started = time.perf_counter()
        metadata = _build_metadata(
            source_path=prepared_audio.source_path,
            prepared_audio=prepared_audio,
            model=normalized_model,
            ffmpeg_executable=ffmpeg_executable,
            num_threads=num_threads,
            debug=debug,
            provider=provider,
            language=language,
            use_itn=use_itn,
            decoding_method=decoding_method,
            vad_options=vad_options,
            result=final_result,
            internal_observability=internal_observability,
            task_started_at=task_started_at,
            task_finished_at=task_finished_at,
            total_wall_seconds=total_wall_seconds,
        )
        retry = internal_observability.get("retry")
        if retry is not None:
            metadata["retry"] = retry
            effective_vad = dict(metadata["configuration"]["vad"])
            effective_vad["min_silence_duration"] = retry["attempts"][-1][
                "vad_min_silence_duration"
            ]
            effective_vad["max_speech_duration"] = retry["attempts"][-1][
                "vad_max_speech_duration"
            ]
            metadata["configuration"]["vad_effective"] = effective_vad
            failed_wall = retry["failed_attempts_wall_seconds"]
            metadata["timing"]["failed_attempts_wall_seconds"] = failed_wall
            metadata["timing"]["other_overhead_wall_seconds"] = round(
                max(0.0, metadata["timing"]["other_overhead_wall_seconds"] - failed_wall),
                6,
            )
        metadata_finished = time.perf_counter()
        metadata["timing"]["metadata_generation_wall_seconds"] = round(
            metadata_finished - metadata_started, 6
        )
        metadata["timing"]["total_until_metadata_ready_seconds"] = round(
            metadata_finished - (task_started or 0.0), 6
        )
        final_result["metadata"] = metadata
    return final_result


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return round(
        ordered[lower_index]
        + (ordered[upper_index] - ordered[lower_index]) * fraction,
        6,
    )


def _build_result_statistics(
    result: dict[str, Any],
    *,
    audio_duration_seconds: float,
) -> dict[str, Any]:
    text = str(result.get("text", ""))
    segments = result.get("segments", [])
    durations = [
        max(0.0, float(segment["end"]) - float(segment["start"]))
        for segment in segments
        if isinstance(segment, dict) and "start" in segment and "end" in segment
    ]
    represented_audio_seconds = sum(durations)
    unrepresented_audio_seconds = max(
        0.0, audio_duration_seconds - represented_audio_seconds
    )
    duration_statistics = {
        "minimum_seconds": round(min(durations), 6) if durations else None,
        "average_seconds": (
            round(represented_audio_seconds / len(durations), 6)
            if durations
            else None
        ),
        "p50_seconds": _percentile(durations, 0.50),
        "p95_seconds": _percentile(durations, 0.95),
        "maximum_seconds": round(max(durations), 6) if durations else None,
    }
    return {
        "segment_count": len(segments) if isinstance(segments, list) else 0,
        "text_character_count": len(text),
        "text_non_whitespace_character_count": sum(
            not character.isspace() for character in text
        ),
        "text_line_count": text.count("\n") + 1 if text else 0,
        "audio_duration_seconds": round(audio_duration_seconds, 6),
        "recognized_segment_audio_seconds": round(
            represented_audio_seconds, 6
        ),
        "unrepresented_audio_seconds": round(unrepresented_audio_seconds, 6),
        "recognized_segment_audio_ratio": _safe_ratio(
            represented_audio_seconds, audio_duration_seconds
        ),
        "segment_duration": duration_statistics,
        "coverage_note": (
            "仅根据联合 EXE 输出的非空识别分段计算；未覆盖时长可能同时包含静音和"
            "未产生文本的语音分段。"
        ),
    }


def _describe_runtime_file(path: str | Path) -> dict[str, Any]:
    try:
        return _describe_file(path)
    except OSError as error:
        return {
            "path": str(Path(path).expanduser()),
            "metadata_available": False,
            "reason": str(error),
        }


def _build_runtime_metadata(internal_runtime: dict[str, Any]) -> dict[str, Any]:
    executable_path = internal_runtime.get("executable_path")
    vad_model_path = internal_runtime.get("vad_model_path")
    model_files = internal_runtime.get("model_files", {})
    runtime: dict[str, Any] = {
        "python_version": platform.python_version(),
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "processor": os.environ.get("PROCESSOR_IDENTIFIER")
        or platform.processor()
        or None,
    }
    if executable_path:
        runtime["asr_executable"] = _describe_runtime_file(executable_path)
    if vad_model_path:
        runtime["vad_model"] = _describe_runtime_file(vad_model_path)
    if isinstance(model_files, dict):
        runtime["model_files"] = {
            name: _describe_runtime_file(path)
            for name, path in model_files.items()
        }
    return runtime


def _build_metadata(
    *,
    source_path: Path,
    prepared_audio: PreparedAudio,
    model: str,
    ffmpeg_executable: str | Path,
    num_threads: int,
    debug: bool,
    provider: str,
    language: str,
    use_itn: bool,
    decoding_method: str,
    vad_options: dict[str, Any],
    result: dict[str, Any],
    internal_observability: dict[str, Any],
    task_started_at: datetime | None,
    task_finished_at: datetime,
    total_wall_seconds: float,
) -> dict[str, Any]:
    audio_metadata = prepared_audio.audio_metadata or {}
    audio_duration_seconds = float(audio_metadata.get("duration_seconds", 0.0))
    internal_timing = internal_observability.get("timing", {})
    vad_asr_wall_seconds = float(
        internal_timing.get("vad_asr_process_wall_seconds", 0.0)
    )
    measured_stage_seconds = (
        prepared_audio.input_preparation_wall_seconds
        + prepared_audio.audio_conversion_wall_seconds
        + prepared_audio.audio_inspection_wall_seconds
        + vad_asr_wall_seconds
    )
    other_overhead_seconds = max(0.0, total_wall_seconds - measured_stage_seconds)

    input_metadata = _describe_file(source_path)
    input_metadata["extension"] = source_path.suffix.lower()
    configuration = {
        "audio_conversion": {
            "ffmpeg_executable_requested": str(ffmpeg_executable),
            "target_sample_rate_hz": 16_000,
            "target_channels": 1,
            "target_sample_width_bits": 16,
        },
        "asr": {
            "model": model,
            "provider": provider,
            "num_threads": num_threads,
            "debug": debug,
            "language": language if model == "sensevoice" else None,
            "use_itn": use_itn if model == "sensevoice" else None,
            "decoding_method": (
                decoding_method if model == "zipformer" else None
            ),
        },
        "vad": {
            "model": "silero_vad.onnx",
            "provider": vad_options["vad_provider"],
            "num_threads": vad_options["vad_num_threads"],
            "debug": vad_options["vad_debug"],
            "threshold": vad_options["vad_threshold"],
            "neg_threshold": vad_options["vad_neg_threshold"],
            "min_silence_duration": vad_options[
                "vad_min_silence_duration"
            ],
            "min_speech_duration": vad_options["vad_min_speech_duration"],
            "max_speech_duration": vad_options["vad_max_speech_duration"],
            "window_size": vad_options["vad_window_size"],
        },
    }
    timing = {
        "processing_started_at": (
            task_started_at.isoformat(timespec="milliseconds")
            if task_started_at is not None
            else None
        ),
        "processing_finished_at": task_finished_at.isoformat(timespec="milliseconds"),
        "total_processing_wall_seconds": round(total_wall_seconds, 6),
        "input_preparation_wall_seconds": round(
            prepared_audio.input_preparation_wall_seconds, 6
        ),
        "audio_conversion_wall_seconds": round(
            prepared_audio.audio_conversion_wall_seconds, 6
        ),
        "audio_inspection_wall_seconds": round(
            prepared_audio.audio_inspection_wall_seconds, 6
        ),
        "vad_asr_combined_wall_seconds": round(vad_asr_wall_seconds, 6),
        "other_overhead_wall_seconds": round(other_overhead_seconds, 6),
        "time_to_first_result_seconds": internal_timing.get(
            "time_to_first_result_seconds"
        ),
        "time_to_last_result_seconds": internal_timing.get(
            "time_to_last_result_seconds"
        ),
        "engine_reported_post_initialization_seconds": internal_timing.get(
            "engine_reported_post_initialization_seconds"
        ),
        "engine_reported_rtf": internal_timing.get("engine_reported_rtf"),
        "end_to_end_rtf": _safe_ratio(total_wall_seconds, audio_duration_seconds),
        "end_to_end_speed_x": _safe_ratio(audio_duration_seconds, total_wall_seconds),
        "vad_asr_combined_rtf": _safe_ratio(
            vad_asr_wall_seconds, audio_duration_seconds
        ),
        "vad_asr_combined_speed_x": _safe_ratio(
            audio_duration_seconds, vad_asr_wall_seconds
        ),
        "measurement_notes": {
            "vad_asr_combined": (
                "包含联合 EXE 的进程启动、模型初始化、WAV 读取、VAD、ASR 和退出。"
            ),
            "engine_reported_post_initialization": (
                "由联合 EXE 内部报告；从模型初始化完成后开始，仍同时包含 WAV 读取、"
                "VAD 和 ASR。"
            ),
            "time_to_result": "表示 Python 从联合 EXE 的 stdout 收到识别分段的时间。",
        },
    }
    return {
        "schema_version": "1.1",
        "invocation": {},
        "input": input_metadata,
        "audio": {
            "conversion_performed": prepared_audio.conversion_performed,
            "conversion_reason": prepared_audio.conversion_reason,
            **audio_metadata,
        },
        "configuration": configuration,
        "timing": timing,
        "resources": internal_observability.get("resources", {}),
        "result_statistics": _build_result_statistics(
            result,
            audio_duration_seconds=audio_duration_seconds,
        ),
        "runtime": _build_runtime_metadata(
            internal_observability.get("runtime", {})
        ),
    }


def _format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        raise ValueError("字幕时间不能小于 0 秒")
    total_milliseconds = int(seconds * 1000 + 0.5)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def _format_srt(segments: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    subtitle_number = 1
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = float(segment["start"])
        end = float(segment["end"])
        if end < start:
            raise ValueError(f"字幕结束时间 {end} 早于开始时间 {start}")
        blocks.append(
            "\n".join(
                [
                    str(subtitle_number),
                    f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}",
                    text,
                ]
            )
        )
        subtitle_number += 1
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _write_selected_outputs(
    source_path: Path,
    result: dict[str, Any],
    *,
    write_txt: bool,
    write_srt: bool,
    write_json: bool,
    overwrite: bool = False,
    model_name: str | None = None,
) -> list[Path]:
    requested_paths = _selected_output_paths(
        source_path,
        write_txt=write_txt,
        write_srt=write_srt,
        write_json=write_json,
        model_name=model_name,
    )
    if not overwrite:
        existing_paths = [path for path in requested_paths if path.exists()]
        if existing_paths:
            existing = ", ".join(str(path) for path in existing_paths)
            raise FileExistsError(f"输出文件已经存在：{existing}")

    output_paths: list[Path] = []
    if write_txt:
        text = result.get("text")
        if not isinstance(text, str):
            raise ValueError("识别结果中缺少可写入 TXT 的 text 字段")
        output_path = _result_output_path(source_path, ".txt", model_name)
        output_path.write_text(text, encoding="utf-8")
        output_paths.append(output_path)

    if write_srt:
        segments = result.get("segments")
        if not isinstance(segments, list):
            raise ValueError("识别结果中缺少可写入 SRT 的 segments 字段")
        output_path = _result_output_path(source_path, ".srt", model_name)
        output_path.write_text(_format_srt(segments), encoding="utf-8")
        output_paths.append(output_path)

    if write_json:
        output_path = _result_output_path(source_path, ".json", model_name)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_paths.append(output_path)
    return output_paths


def _result_output_path(
    source_path: Path,
    output_suffix: str,
    model_name: str | None = None,
) -> Path:
    if model_name is None:
        return source_path.with_suffix(output_suffix)
    return source_path.with_name(f"{source_path.stem}.{model_name}{output_suffix}")


def _selected_output_paths(
    source_path: Path,
    *,
    write_txt: bool,
    write_srt: bool,
    write_json: bool,
    model_name: str | None = None,
) -> list[Path]:
    output_paths: list[Path] = []
    if write_txt:
        output_paths.append(_result_output_path(source_path, ".txt", model_name))
    if write_srt:
        output_paths.append(_result_output_path(source_path, ".srt", model_name))
    if write_json:
        output_paths.append(_result_output_path(source_path, ".json", model_name))
    return output_paths


def _collect_input_files(input_path: str | Path) -> tuple[list[Path], bool]:
    resolved_input = Path(input_path).expanduser().resolve()
    if resolved_input.is_file():
        return [resolved_input], False
    if not resolved_input.is_dir():
        raise FileNotFoundError(f"输入文件或文件夹不存在：{resolved_input}")

    media_files = sorted(
        (
            path
            for path in resolved_input.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if not media_files:
        raise FileNotFoundError(
            f"文件夹当前层级没有找到支持的音频或视频文件：{resolved_input}"
        )
    return media_files, True


def _validate_output_conflicts(
    input_files: list[Path],
    models_to_run: tuple[str, ...],
    *,
    write_txt: bool,
    write_srt: bool,
    write_json: bool,
) -> None:
    owners: dict[Path, list[Path]] = {}
    for source_path in input_files:
        for model in models_to_run:
            for output_path in _selected_output_paths(
                source_path,
                write_txt=write_txt,
                write_srt=write_srt,
                write_json=write_json,
                model_name=model if len(models_to_run) > 1 else None,
            ):
                owners.setdefault(output_path, []).append(source_path)
    conflicts = [
        f"  {output_path} <- {', '.join(str(source) for source in sources)}"
        for output_path, sources in owners.items()
        if len(sources) > 1
    ]
    if conflicts:
        raise ValueError(
            "多个输入文件的输出路径冲突，请先调整输入文件名：\n" + "\n".join(conflicts)
        )


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("数量必须大于 0")
    return number


def parse_args() -> argparse.Namespace:
    parser = ChineseArgumentParser(
        add_help=False,
        formatter_class=ChineseHelpFormatter,
        usage="uv run scripts/asr.py 输入文件或文件夹 [--model 模型] [其他选项]",
        description=(
            "统一的本地语音识别入口。可输入单个视频、音频或文件夹；文件夹模式会批量处理"
            "当前层级的常见音视频文件，不扫描子文件夹。脚本会自动转换为模型所需的 "
            "16 kHz、单声道、16-bit PCM WAV，再通过 Silero VAD 分段并调用指定的 ASR 模型。"
        ),
        epilog=(
            "使用示例：\n"
            "  uv run scripts/asr.py \"video.mp4\" --model sensevoice\n"
            "  uv run scripts/asr.py \"audio.m4a\" --model zipformer --srt\n"
            "  uv run scripts/asr.py \"video.mp4\" --model sensevoice --txt --json\n"
            "  uv run scripts/asr.py \"video.mp4\" --model sensevoice --all-output\n"
            "  uv run scripts/asr.py \"video.mp4\" --model all --all-output\n"
            "  uv run scripts/asr.py \"D:\\Media\" --model zipformer --all-output\n"
            "  uv run scripts/asr.py \"D:\\Media\" --model all --all-output --overwrite\n"
            "  uv run scripts/asr.py \"D:\\Media\" --model all --srt --file-workers 2\n\n"
            "输出说明：\n"
            "  未指定任何输出参数时，完整 JSON 结果打印到控制台。\n"
            "  文件夹模式或 --model all 模式下，控制台 JSON 是包含文件路径及识别结果的数组。\n"
            "  指定任意输出参数后，结果写到源文件旁边，控制台不再打印识别结果。\n"
            "  输出文件沿用源文件名称，并分别使用 .txt、.srt、.json 后缀。\n"
            "  --model all 会依次运行 sensevoice、zipformer，并生成例如 "
            "video.sensevoice.txt 和 video.zipformer.txt。\n"
            "  --all-output 等同于同时指定 --txt --srt --json。\n"
            "  批处理写文件前会检查输出路径冲突；同名不同扩展名的输入可能冲突，"
            "--overwrite 也不会放行。\n"
            "  默认遇到本次要生成的任一同名结果文件就跳过整个源文件；"
            "--overwrite 可以覆盖。\n"
            "  JSON 还包含输入、参数、分阶段耗时、联合 VAD+ASR 子进程资源和结果统计。\n"
            "  完整观测只在输出 JSON 或控制台 JSON 时启用；仅输出 TXT/SRT 时不采集。\n"
            "  ZipFormer 仅在确认长片段错误后逐级收紧 VAD 重试，最多尝试 4 次；"
            "首轮参数不变，全部失败后继续其他任务。\n"
            "  处理状态和进度写入标准错误流，不会进入 JSON、TXT 或 SRT 内容。"
        ),
    )
    parser._positionals.title = "位置参数"
    parser._optionals.title = "其他选项"

    parser.add_argument(
        "input",
        type=Path,
        metavar="输入文件或文件夹",
        help=(
            "要识别的音频、视频或文件夹路径。单文件由 FFmpeg 判断格式；文件夹只扫描"
            "当前层级中常见的音视频后缀，例如 MP4、MKV、MOV、MP3、M4A、FLAC 和 WAV，"
            "不会进入子文件夹，也支持中文路径。"
        ),
    )

    basic_group = parser.add_argument_group("基础选项")
    basic_group.add_argument(
        "-h", "--help", action="help", help="显示这份帮助信息并退出。"
    )
    basic_group.add_argument(
        "--model",
        choices=[*MODEL_ORDER, ALL_MODELS_CHOICE],
        default="sensevoice",
        metavar="{sensevoice,zipformer,all}",
        help=(
            "选择语音识别模型：sensevoice 支持普通话、粤语、英语、日语、韩语；"
            "zipformer 支持中文、英语；all 会按照 sensevoice、zipformer 的顺序"
            "依次运行全部模型。默认使用 sensevoice。"
        ),
    )
    basic_group.add_argument(
        "--file-workers",
        type=_positive_int,
        default=1,
        metavar="数量",
        help=(
            "同时处理的文件数上限，默认 1，逐个处理文件。大于 1 时按文件并发，"
            "同一文件的模型仍依次运行；不调整 ASR 和 VAD 的推理线程数。"
        ),
    )
    basic_group.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        metavar="路径",
        help=(
            "FFmpeg 可执行文件名或完整路径。输入已经是 16 kHz、单声道、16-bit PCM "
            "WAV 时会跳过转换；默认从系统 PATH 中查找 ffmpeg。"
        ),
    )
    output_group = parser.add_argument_group("输出选项")
    output_group.add_argument(
        "--txt",
        action="store_true",
        help=(
            "将完整识别文本写到源文件旁边的同名 .txt 文件。VAD 分段之间保留换行；"
            "指定后不在控制台打印识别结果。"
        ),
    )
    output_group.add_argument(
        "--srt",
        action="store_true",
        help=(
            "根据每个 VAD 分段的开始和结束时间生成同名 .srt 字幕文件；"
            "指定后不在控制台打印识别结果。"
        ),
    )
    output_group.add_argument(
        "--json",
        action="store_true",
        help=(
            "将模型名称、完整文本、全部时间分段以及输入、参数、耗时和资源观测信息"
            "写到同名 .json 文件；指定后不在控制台打印识别结果。"
        ),
    )
    output_group.add_argument(
        "--all-output",
        action="store_true",
        help="同时输出同名的 TXT、SRT 和 JSON 三种文件。",
    )
    output_group.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "允许覆盖本次准备生成的已有结果文件。默认关闭；默认情况下只要任一目标 "
            "TXT、SRT 或 JSON 已存在，就会在转换和识别前跳过对应的模型任务。"
        ),
    )

    runtime_group = parser.add_argument_group("ASR 推理参数")
    add_runtime_arguments(runtime_group)

    model_group = parser.add_argument_group("模型专属参数")
    model_group.add_argument(
        "--language",
        choices=["auto", "zh", "en", "ja", "ko", "yue"],
        default="auto",
        metavar="语言",
        help=(
            "SenseVoice 的输入语言。auto 自动判断；zh、en、ja、ko、yue 分别表示"
            "中文、英文、日文、韩文、粤语。默认 auto；ZipFormer 会忽略此参数。"
        ),
    )
    model_group.add_argument(
        "--use-itn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "是否为 SenseVoice 启用逆文本规范化，将口语数字等内容转换为更适合阅读的"
            "形式。默认开启；使用 --no-use-itn 可关闭。ZipFormer 会忽略此参数。"
        ),
    )
    model_group.add_argument(
        "--decoding-method",
        choices=["greedy_search", "modified_beam_search"],
        default="greedy_search",
        metavar="方法",
        help=(
            "ZipFormer 的解码方法。greedy_search 速度更快并作为默认值；"
            "modified_beam_search 使用多路径搜索。SenseVoice 会忽略此参数。"
        ),
    )

    vad_group = parser.add_argument_group("Silero VAD 参数")
    add_vad_arguments(vad_group)
    return parser.parse_args()


def main() -> int:
    invocation_started_at = datetime.now().astimezone()
    args = parse_args()
    try:
        input_files, directory_mode = _collect_input_files(args.input)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    write_txt = args.txt or args.all_output
    write_srt = args.srt or args.all_output
    write_json = args.json or args.all_output
    selected_output = write_txt or write_srt or write_json
    collect_observability = write_json or not selected_output
    invocation_id = str(uuid4()) if collect_observability else None
    models_to_run = MODEL_ORDER if args.model == ALL_MODELS_CHOICE else (args.model,)
    multi_model_mode = len(models_to_run) > 1
    try:
        _validate_output_conflicts(
            input_files,
            models_to_run,
            write_txt=write_txt,
            write_srt=write_srt,
            write_json=write_json,
        )
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    file_workers = min(args.file_workers, len(input_files))
    concurrent_mode = file_workers > 1
    process_group = ProcessGroup()
    output_lock = threading.Lock()

    def report(message: str) -> None:
        with output_lock:
            print(message, file=sys.stderr, flush=True)

    if directory_mode:
        print(
            f"[批处理] 当前目录共发现 {len(input_files)} 个支持的音视频文件，"
            f"将执行 {len(input_files) * len(models_to_run)} 个模型任务",
            file=sys.stderr,
            flush=True,
        )

    total_files = len(input_files)
    if concurrent_mode:
        report(f"[并发] 最多同时处理 {file_workers} 个文件，同一文件的模型依次运行")

    def process_file(
        index: int,
        source_path: Path,
    ) -> tuple[list[dict[str, Any]], int, int, int]:
        console_results: list[dict[str, Any]] = []
        succeeded = 0
        skipped = 0
        failed = 0
        process_group.check_cancelled()
        if directory_mode:
            report(f"[文件 {index}/{total_files}] {source_path.name}")

        for model_index, current_model in enumerate(models_to_run, start=1):
            process_group.check_cancelled()
            prefix = f"[{source_path.name}][{current_model}] " if concurrent_mode else ""
            if multi_model_mode:
                report(f"{prefix}[模型 {model_index}/{len(models_to_run)}] {current_model}")

            output_model_name = current_model if multi_model_mode else None
            requested_paths = _selected_output_paths(
                source_path,
                write_txt=write_txt,
                write_srt=write_srt,
                write_json=write_json,
                model_name=output_model_name,
            )
            existing_paths = [path for path in requested_paths if path.exists()]
            if existing_paths and not args.overwrite:
                existing = ", ".join(path.name for path in existing_paths)
                report(f"{prefix}[跳过][{current_model}] 已存在本次要生成的结果文件：{existing}")
                skipped += 1
                continue

            progress = ConsoleProgress(prefix=prefix, output_lock=output_lock)
            progress.start()
            try:
                result = run_asr(
                    source_path,
                    model=current_model,
                    ffmpeg_executable=args.ffmpeg,
                    num_threads=args.num_threads,
                    debug=args.debug,
                    provider=args.provider,
                    language=args.language,
                    use_itn=args.use_itn,
                    decoding_method=args.decoding_method,
                    status_callback=progress.status,
                    progress_callback=progress.update,
                    collect_observability=collect_observability,
                    process_group=process_group,
                    **vad_kwargs_from_args(args),
                )
                process_group.check_cancelled()
                if collect_observability:
                    result["metadata"]["invocation"] = {
                        "run_id": invocation_id,
                        "requested_model": args.model,
                        "model_sequence": list(models_to_run),
                        "model_index": model_index,
                        "model_count": len(models_to_run),
                        "file_index": index,
                        "file_count": total_files,
                        "file_workers_requested": args.file_workers,
                        "file_workers_effective": file_workers,
                        "command_started_at": invocation_started_at.isoformat(
                            timespec="milliseconds"
                        ),
                        "output_formats_requested": [
                            output_format
                            for output_format, enabled in (
                                ("txt", write_txt),
                                ("srt", write_srt),
                                ("json", write_json),
                            )
                            if enabled
                        ],
                        "overwrite": args.overwrite,
                    }
                progress.finish()
                if selected_output:
                    progress.status("正在写入输出文件")
                    output_paths = _write_selected_outputs(
                        source_path,
                        result,
                        write_txt=write_txt,
                        write_srt=write_srt,
                        write_json=write_json,
                        overwrite=args.overwrite,
                        model_name=output_model_name,
                    )
                    for output_path in output_paths:
                        progress.status(f"已写入 {output_path}")
                elif directory_mode or multi_model_mode:
                    console_results.append({"input": str(source_path), **result})
                else:
                    console_results.append(result)
                succeeded += 1
            except (OSError, ValueError, RuntimeError) as error:
                progress.abort()
                process_group.check_cancelled()
                report(f"[失败][{current_model}] {source_path}: {error}")
                failed += 1
            finally:
                progress.abort()
        return console_results, succeeded, skipped, failed

    file_results: dict[int, tuple[list[dict[str, Any]], int, int, int]] = {}
    executor: ThreadPoolExecutor | None = None
    try:
        if concurrent_mode:
            executor = ThreadPoolExecutor(max_workers=file_workers)
            futures = {
                executor.submit(process_file, index, source_path): index
                for index, source_path in enumerate(input_files, start=1)
            }
            pending = set(futures)
            while pending:
                # Periodically return to Python so Ctrl+C stays responsive on Windows.
                done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                for future in done:
                    file_results[futures[future]] = future.result()
        else:
            for index, source_path in enumerate(input_files, start=1):
                file_results[index] = process_file(index, source_path)
    except KeyboardInterrupt:
        process_group.cancel()
        report("[中断] 已停止批处理，正在清理子进程和临时文件")
        return 130
    except BaseException:
        process_group.cancel()
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    console_results: list[dict[str, Any]] = []
    succeeded = skipped = failed = 0
    for index in sorted(file_results):
        results, file_succeeded, file_skipped, file_failed = file_results[index]
        console_results.extend(results)
        succeeded += file_succeeded
        skipped += file_skipped
        failed += file_failed

    aggregate_console_output = directory_mode or multi_model_mode
    if not selected_output and (aggregate_console_output or console_results):
        print(
            json.dumps(
                console_results if aggregate_console_output else console_results[0],
                ensure_ascii=False,
                indent=2,
            )
        )

    if directory_mode or multi_model_mode:
        summary_label = "批处理完成" if directory_mode else "全部模型完成"
        print(
            f"[{summary_label}] 模型任务成功 {succeeded} 个，跳过 {skipped} 个，"
            f"失败 {failed} 个",
            file=sys.stderr,
            flush=True,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
