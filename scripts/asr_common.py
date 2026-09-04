"""Shared process, VAD, threading, and path helpers for local ASR scripts."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

try:
    from .process_metrics import ProcessResourceMonitor
except ImportError:
    from process_metrics import ProcessResourceMonitor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VAD_EXECUTABLE_PATH = (
    PROJECT_ROOT
    / "app"
    / "sherpa-onnx"
    / "bin"
    / "sherpa-onnx-vad-with-offline-asr.exe"
)
DEFAULT_VAD_MODEL_PATH = PROJECT_ROOT / "models" / "vad" / "silero_vad.onnx"

LOGICAL_PROCESSORS = os.cpu_count() or 1
DEFAULT_NUM_THREADS = max(1, min(12, (LOGICAL_PROCESSORS + 1) // 2))
DEFAULT_VAD_NUM_THREADS = max(1, min(4, (LOGICAL_PROCESSORS + 3) // 4))

DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_VAD_MIN_SILENCE_DURATION = 1.0
DEFAULT_VAD_MIN_SPEECH_DURATION = 0.25
DEFAULT_VAD_MAX_SPEECH_DURATION = 20.0
DEFAULT_VAD_WINDOW_SIZE = 512
DEFAULT_VAD_NEG_THRESHOLD = -1.0

SEGMENT_PATTERN = re.compile(
    r"^(?P<start>\d+(?:\.\d+)?) -- (?P<end>\d+(?:\.\d+)?):\s*(?P<text>.*)$"
)
ENGINE_ELAPSED_PATTERN = re.compile(
    r"Elapsed seconds:\s*(?P<seconds>\d+(?:\.\d+)?)\s*s"
)
ENGINE_RTF_PATTERN = re.compile(
    r"Real time factor \(RTF\):.*=\s*(?P<rtf>\d+(?:\.\d+)?)"
)


def resolve_file(path: str | Path, description: str) -> Path:
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {resolved_path}")
    return resolved_path


def resolve_audio(path: str | Path) -> Path:
    audio = resolve_file(path, "Input audio file")
    if audio.suffix.lower() != ".wav":
        raise ValueError(f"Input audio file must use the .wav extension: {audio}")
    return audio


def resolve_executable(executable: str | Path, description: str) -> str:
    executable_path = Path(executable).expanduser()
    if executable_path.is_file():
        return str(executable_path.resolve())

    resolved_executable = shutil.which(str(executable))
    if resolved_executable is None:
        raise FileNotFoundError(f"{description} was not found: {executable}")
    return resolved_executable


@contextmanager
def compatible_audio_path(audio: Path) -> Iterator[Path]:
    """Provide an ASCII-only path for local sherpa-onnx Windows executables."""
    if str(audio).isascii():
        yield audio
        return

    with tempfile.TemporaryDirectory(prefix="sherpa-onnx-audio-") as temp_dir:
        staged_audio = Path(temp_dir) / "input.wav"
        shutil.copyfile(audio, staged_audio)
        yield staged_audio


def _copy_wave_with_trailing_silence(source: Path, destination: Path) -> None:
    """Copy a PCM WAV and append one silent audio frame."""
    with wave.open(str(source), "rb") as source_wav:
        channels = source_wav.getnchannels()
        sample_width = source_wav.getsampwidth()
        frame_rate = source_wav.getframerate()
        compression_type = source_wav.getcomptype()
        compression_name = source_wav.getcompname()

        with wave.open(str(destination), "wb") as destination_wav:
            destination_wav.setnchannels(channels)
            destination_wav.setsampwidth(sample_width)
            destination_wav.setframerate(frame_rate)
            destination_wav.setcomptype(compression_type, compression_name)

            while frames := source_wav.readframes(1024 * 1024):
                destination_wav.writeframesraw(frames)
            destination_wav.writeframes(bytes(channels * sample_width))


@contextmanager
def vad_compatible_audio_path(audio: Path, window_size: int) -> Iterator[Path]:
    """Provide a safe input path for the combined VAD and offline ASR executable.

    The bundled executable can leave its final active speech segment unflushed
    when the WAV frame count is exactly divisible by the VAD window size. Stage
    such inputs with one silent audio frame so the final window is incomplete.
    """
    with wave.open(str(audio), "rb") as audio_wav:
        requires_trailing_frame = audio_wav.getnframes() % window_size == 0

    if not requires_trailing_frame:
        with compatible_audio_path(audio) as compatible_audio:
            yield compatible_audio
        return

    with tempfile.TemporaryDirectory(prefix="sherpa-onnx-vad-audio-") as temp_dir:
        staged_audio = Path(temp_dir) / "input.wav"
        _copy_wave_with_trailing_silence(audio, staged_audio)
        yield staged_audio


def validate_num_threads(num_threads: int, vad_num_threads: int | None = None) -> None:
    if num_threads <= 0:
        raise ValueError("num_threads must be greater than 0")
    if vad_num_threads is not None and vad_num_threads <= 0:
        raise ValueError("vad_num_threads must be greater than 0")


def validate_vad_options(
    *,
    threshold: float,
    min_silence_duration: float,
    min_speech_duration: float,
    max_speech_duration: float,
    window_size: int,
    neg_threshold: float,
) -> None:
    if not 0 < threshold < 1:
        raise ValueError("vad_threshold must be between 0 and 1")
    if neg_threshold != -1 and not 0 < neg_threshold < 1:
        raise ValueError("vad_neg_threshold must be -1 or between 0 and 1")
    if min_silence_duration <= 0:
        raise ValueError("vad_min_silence_duration must be greater than 0")
    if min_speech_duration <= 0:
        raise ValueError("vad_min_speech_duration must be greater than 0")
    if max_speech_duration <= 0:
        raise ValueError("vad_max_speech_duration must be greater than 0")
    if window_size <= 0:
        raise ValueError("vad_window_size must be greater than 0")


def _run_process(command: Sequence[str], failure_label: str) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if process.returncode != 0:
        details = process.stderr.strip() or process.stdout.strip() or "Unknown error"
        raise RuntimeError(
            f"{failure_label} failed with exit code {process.returncode}: {details}"
        )
    return process


def run_json_asr(
    audio_path: str | Path,
    *,
    executable_path: str | Path,
    model_arguments: Sequence[str],
    num_threads: int,
    debug: bool,
    provider: str,
    failure_label: str,
) -> dict[str, Any]:
    """Run a direct offline recognizer and parse its single JSON result."""
    audio = resolve_audio(audio_path)
    executable = resolve_executable(executable_path, f"{failure_label} executable")
    validate_num_threads(num_threads)

    with compatible_audio_path(audio) as compatible_audio:
        command = [
            executable,
            *model_arguments,
            f"--num-threads={num_threads}",
            f"--debug={str(debug).lower()}",
            f"--provider={provider}",
            "--print-args=false",
            str(compatible_audio),
        ]
        process = _run_process(command, failure_label)

    output = process.stdout.strip()
    if not output:
        raise RuntimeError(f"{failure_label} completed successfully but returned no JSON")
    try:
        result = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{failure_label} returned invalid JSON: {output}") from error
    if not isinstance(result, dict):
        raise RuntimeError(f"{failure_label} returned an unexpected JSON value: {output}")
    return result


def run_vad_asr(
    audio_path: str | Path,
    *,
    executable_path: str | Path,
    vad_model_path: str | Path,
    model_arguments: Sequence[str],
    num_threads: int,
    debug: bool,
    provider: str,
    vad_threshold: float = DEFAULT_VAD_THRESHOLD,
    vad_min_silence_duration: float = DEFAULT_VAD_MIN_SILENCE_DURATION,
    vad_min_speech_duration: float = DEFAULT_VAD_MIN_SPEECH_DURATION,
    vad_max_speech_duration: float = DEFAULT_VAD_MAX_SPEECH_DURATION,
    vad_window_size: int = DEFAULT_VAD_WINDOW_SIZE,
    vad_neg_threshold: float = DEFAULT_VAD_NEG_THRESHOLD,
    vad_num_threads: int = DEFAULT_VAD_NUM_THREADS,
    vad_debug: bool = False,
    vad_provider: str = "cpu",
    failure_label: str,
    progress_callback: Callable[[float, float, int], None] | None = None,
    collect_observability: bool = False,
) -> dict[str, Any]:
    """Run Silero VAD with an offline recognizer and parse timestamped segments."""
    audio = resolve_audio(audio_path)
    vad_model = resolve_file(vad_model_path, "Silero VAD model")
    executable = resolve_executable(executable_path, f"{failure_label} executable")
    validate_num_threads(num_threads, vad_num_threads)
    validate_vad_options(
        threshold=vad_threshold,
        min_silence_duration=vad_min_silence_duration,
        min_speech_duration=vad_min_speech_duration,
        max_speech_duration=vad_max_speech_duration,
        window_size=vad_window_size,
        neg_threshold=vad_neg_threshold,
    )

    with wave.open(str(audio), "rb") as wav_file:
        total_duration = wav_file.getnframes() / wav_file.getframerate()

    with vad_compatible_audio_path(audio, vad_window_size) as compatible_audio:
        command = [
            executable,
            f"--silero-vad-model={vad_model}",
            f"--silero-vad-threshold={vad_threshold}",
            f"--silero-vad-min-silence-duration={vad_min_silence_duration}",
            f"--silero-vad-min-speech-duration={vad_min_speech_duration}",
            f"--silero-vad-max-speech-duration={vad_max_speech_duration}",
            f"--silero-vad-window-size={vad_window_size}",
            f"--silero-vad-neg-threshold={vad_neg_threshold}",
            f"--vad-num-threads={vad_num_threads}",
            f"--vad-debug={str(vad_debug).lower()}",
            f"--vad-provider={vad_provider}",
            *model_arguments,
            f"--num-threads={num_threads}",
            f"--debug={str(debug).lower()}",
            f"--provider={provider}",
            "--print-args=false",
            str(compatible_audio),
        ]
        process_started = time.perf_counter() if collect_observability else None
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if process.stdout is None or process.stderr is None:
            process.kill()
            raise RuntimeError(f"{failure_label} could not capture process output")

        resource_monitor = (
            ProcessResourceMonitor(process.pid) if collect_observability else None
        )
        if resource_monitor is not None:
            resource_monitor.start()

        stderr_lines: list[str] = []

        def drain_stderr() -> None:
            stderr_lines.extend(process.stderr)

        stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
        stderr_thread.start()
        segments: list[dict[str, Any]] = []
        unexpected_lines: list[str] = []
        first_result_seconds: float | None = None
        last_result_seconds: float | None = None
        process_wall_seconds: float | None = None
        resource_metrics: dict[str, Any] | None = None
        try:
            for line in process.stdout:
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                match = SEGMENT_PATTERN.fullmatch(stripped_line)
                if match is None:
                    unexpected_lines.append(stripped_line)
                    continue
                segment = {
                    "start": float(match.group("start")),
                    "end": float(match.group("end")),
                    "text": match.group("text"),
                }
                segments.append(segment)
                if process_started is not None:
                    result_seconds = time.perf_counter() - process_started
                    if first_result_seconds is None:
                        first_result_seconds = result_seconds
                    last_result_seconds = result_seconds
                if progress_callback is not None:
                    progress_callback(
                        min(segment["end"], total_duration),
                        total_duration,
                        len(segments),
                    )
            return_code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            stderr_thread.join()
            if process_started is not None:
                process_wall_seconds = time.perf_counter() - process_started
            if resource_monitor is not None:
                try:
                    resource_metrics = resource_monitor.stop(
                        process_wall_seconds=process_wall_seconds or 0.0
                    )
                except Exception as error:
                    resource_metrics = {
                        "available": False,
                        "reason": f"资源监控失败：{error}",
                    }

    if return_code != 0:
        details = "".join(stderr_lines).strip()
        if not details:
            details = "\n".join(unexpected_lines) or "Unknown error"
        raise RuntimeError(
            f"{failure_label} failed with exit code {return_code}: {details}"
        )

    if unexpected_lines:
        output = "\n".join(unexpected_lines)
        raise RuntimeError(f"{failure_label} returned unexpected output: {output}")

    result: dict[str, Any] = {
        "text": "\n".join(segment["text"] for segment in segments),
        "segments": segments,
    }
    if collect_observability:
        stderr_output = "".join(stderr_lines)
        elapsed_match = ENGINE_ELAPSED_PATTERN.search(stderr_output)
        rtf_match = ENGINE_RTF_PATTERN.search(stderr_output)
        result["_observability"] = {
            "timing": {
                "vad_asr_process_wall_seconds": round(
                    process_wall_seconds or 0.0, 6
                ),
                "time_to_first_result_seconds": (
                    round(first_result_seconds, 6)
                    if first_result_seconds is not None
                    else None
                ),
                "time_to_last_result_seconds": (
                    round(last_result_seconds, 6)
                    if last_result_seconds is not None
                    else None
                ),
                "engine_reported_post_initialization_seconds": (
                    float(elapsed_match.group("seconds")) if elapsed_match else None
                ),
                "engine_reported_rtf": (
                    float(rtf_match.group("rtf")) if rtf_match else None
                ),
            },
            "resources": {
                "vad_asr_process": resource_metrics,
            },
            "runtime": {
                "executable_path": executable,
                "vad_model_path": str(vad_model),
            },
        }
    return result


def add_runtime_arguments(parser: Any) -> None:
    parser.add_argument(
        "--num-threads",
        type=int,
        default=DEFAULT_NUM_THREADS,
        metavar="数量",
        help=(
            "ASR 模型推理线程数。默认取逻辑处理器数量的一半并限制为最高 12；"
            f"当前机器的默认值为 {DEFAULT_NUM_THREADS}。"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="输出 ASR 模型加载信息和详细调试日志；默认关闭。",
    )
    parser.add_argument(
        "--provider",
        default="cpu",
        metavar="名称",
        help=(
            "ASR 模型使用的 ONNX Runtime 执行提供程序，例如 cpu、cuda 或 "
            "coreml；默认使用 cpu。"
        ),
    )


def add_vad_arguments(parser: Any) -> None:
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=DEFAULT_VAD_THRESHOLD,
        metavar="数值",
        help=(
            "Silero VAD 的语音概率阈值，范围为 0 到 1。降低会提高灵敏度，"
            f"升高会减少噪声误判；默认 {DEFAULT_VAD_THRESHOLD}。"
        ),
    )
    parser.add_argument(
        "--vad-min-silence-duration",
        type=float,
        default=DEFAULT_VAD_MIN_SILENCE_DURATION,
        metavar="秒",
        help=(
            "判定一个语音段结束前要求持续静音的最短时间。数值较大可减少句子"
            f"边界截断；默认 {DEFAULT_VAD_MIN_SILENCE_DURATION} 秒。"
        ),
    )
    parser.add_argument(
        "--vad-min-speech-duration",
        type=float,
        default=DEFAULT_VAD_MIN_SPEECH_DURATION,
        metavar="秒",
        help=(
            "保留语音段所需的最短持续时间，更短的声音会被当作瞬态噪声忽略；"
            f"默认 {DEFAULT_VAD_MIN_SPEECH_DURATION} 秒。"
        ),
    )
    parser.add_argument(
        "--vad-max-speech-duration",
        type=float,
        default=DEFAULT_VAD_MAX_SPEECH_DURATION,
        metavar="秒",
        help=(
            "长语音段的软切分触发时间。超过该时长后会提高检测阈值，以便尽快在"
            f"合适位置切分；默认 {DEFAULT_VAD_MAX_SPEECH_DURATION} 秒。"
        ),
    )
    parser.add_argument(
        "--vad-window-size",
        type=int,
        default=DEFAULT_VAD_WINDOW_SIZE,
        metavar="采样点",
        help=(
            "每次送入 Silero VAD 的采样点数量。当前模型和 16 kHz 音频建议保持 "
            f"{DEFAULT_VAD_WINDOW_SIZE}，通常无需修改。"
        ),
    )
    parser.add_argument(
        "--vad-neg-threshold",
        type=float,
        default=DEFAULT_VAD_NEG_THRESHOLD,
        metavar="数值",
        help=(
            "Silero VAD 的非语音阈值。-1 表示自动使用语音阈值减 0.15；"
            f"默认 {DEFAULT_VAD_NEG_THRESHOLD}。"
        ),
    )
    parser.add_argument(
        "--vad-num-threads",
        type=int,
        default=DEFAULT_VAD_NUM_THREADS,
        metavar="数量",
        help=(
            "Silero VAD 推理线程数。默认约为逻辑处理器数量的四分之一并限制为"
            f"最高 4；当前机器的默认值为 {DEFAULT_VAD_NUM_THREADS}。"
        ),
    )
    parser.add_argument(
        "--vad-debug",
        action="store_true",
        help="输出 Silero VAD 模型信息和详细调试日志；默认关闭。",
    )
    parser.add_argument(
        "--vad-provider",
        default="cpu",
        metavar="名称",
        help=(
            "Silero VAD 使用的 ONNX Runtime 执行提供程序，例如 cpu、cuda 或 "
            "coreml；默认使用 cpu。"
        ),
    )


def vad_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "vad_threshold": args.vad_threshold,
        "vad_min_silence_duration": args.vad_min_silence_duration,
        "vad_min_speech_duration": args.vad_min_speech_duration,
        "vad_max_speech_duration": args.vad_max_speech_duration,
        "vad_window_size": args.vad_window_size,
        "vad_neg_threshold": args.vad_neg_threshold,
        "vad_num_threads": args.vad_num_threads,
        "vad_debug": args.vad_debug,
        "vad_provider": args.vad_provider,
    }
