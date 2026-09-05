"""Run single-file speech recognition with Silero VAD and local ZipFormer."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable

try:
    from .process_group import ProcessGroup
except ImportError:
    from process_group import ProcessGroup

try:
    from .asr_common import (
        DEFAULT_NUM_THREADS,
        DEFAULT_VAD_EXECUTABLE_PATH,
        DEFAULT_VAD_MAX_SPEECH_DURATION,
        DEFAULT_VAD_MIN_SILENCE_DURATION,
        DEFAULT_VAD_MIN_SPEECH_DURATION,
        DEFAULT_VAD_MODEL_PATH,
        DEFAULT_VAD_NEG_THRESHOLD,
        DEFAULT_VAD_NUM_THREADS,
        DEFAULT_VAD_THRESHOLD,
        DEFAULT_VAD_WINDOW_SIZE,
        PROJECT_ROOT,
        resolve_file,
        run_vad_asr,
    )
except ImportError:
    from asr_common import (
        DEFAULT_NUM_THREADS,
        DEFAULT_VAD_EXECUTABLE_PATH,
        DEFAULT_VAD_MAX_SPEECH_DURATION,
        DEFAULT_VAD_MIN_SILENCE_DURATION,
        DEFAULT_VAD_MIN_SPEECH_DURATION,
        DEFAULT_VAD_MODEL_PATH,
        DEFAULT_VAD_NEG_THRESHOLD,
        DEFAULT_VAD_NUM_THREADS,
        DEFAULT_VAD_THRESHOLD,
        DEFAULT_VAD_WINDOW_SIZE,
        PROJECT_ROOT,
        resolve_file,
        run_vad_asr,
    )


DEFAULT_EXECUTABLE_PATH = DEFAULT_VAD_EXECUTABLE_PATH
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "zipformer"
DEFAULT_TOKENS_PATH = DEFAULT_MODEL_DIR / "tokens.txt"
DEFAULT_ENCODER_PATH = DEFAULT_MODEL_DIR / "encoder-epoch-99-avg-1.int8.onnx"
DEFAULT_DECODER_PATH = DEFAULT_MODEL_DIR / "decoder-epoch-99-avg-1.onnx"
DEFAULT_JOINER_PATH = DEFAULT_MODEL_DIR / "joiner-epoch-99-avg-1.int8.onnx"
SUPPORTED_DECODING_METHODS = {"greedy_search", "modified_beam_search"}
# Scale the caller's durations; attempt 1 always uses their original settings.
VAD_ATTEMPT_SCALES = ((1.0, 1.0), (0.2, 0.5), (0.1, 0.25), (0.05, 0.125))
ZIPFORMER_LENGTH_ERROR = re.compile(
    r"Name:'/encoder/0/layers\.0/self_attn_weights/Reshape_3'[^\r\n]*"
    r"Input shape:\{\s*1\s*,\s*(\d+)\s*,\s*16\s*\},\s*"
    r"requested shape:\{\s*-1\s*,\s*(\d+)\s*,\s*4\s*,\s*4\s*\}"
)


def _is_zipformer_length_error(error: RuntimeError) -> bool:
    """Match the verified overflow of this encoder's 3999-position table."""
    message = str(error)
    if not message.startswith("ZipFormer VAD recognition failed with exit code "):
        return False
    match = ZIPFORMER_LENGTH_ERROR.search(message)
    if match is None:
        return False
    available, requested = (int(value) for value in match.groups())
    if requested <= 3999 or requested % 2 != 1:
        return False
    sequence_length = (requested + 1) // 2
    # Negative slice starts wrap around the fixed table, then clamp at its start.
    return available == min(sequence_length - 2000, 3999)


def run_zipformer(
    audio_path: str | Path,
    *,
    executable_path: str | Path = DEFAULT_EXECUTABLE_PATH,
    vad_model_path: str | Path = DEFAULT_VAD_MODEL_PATH,
    tokens_path: str | Path = DEFAULT_TOKENS_PATH,
    encoder_path: str | Path = DEFAULT_ENCODER_PATH,
    decoder_path: str | Path = DEFAULT_DECODER_PATH,
    joiner_path: str | Path = DEFAULT_JOINER_PATH,
    num_threads: int = DEFAULT_NUM_THREADS,
    decoding_method: str = "greedy_search",
    debug: bool = False,
    provider: str = "cpu",
    vad_threshold: float = DEFAULT_VAD_THRESHOLD,
    vad_min_silence_duration: float = DEFAULT_VAD_MIN_SILENCE_DURATION,
    vad_min_speech_duration: float = DEFAULT_VAD_MIN_SPEECH_DURATION,
    vad_max_speech_duration: float = DEFAULT_VAD_MAX_SPEECH_DURATION,
    vad_window_size: int = DEFAULT_VAD_WINDOW_SIZE,
    vad_neg_threshold: float = DEFAULT_VAD_NEG_THRESHOLD,
    vad_num_threads: int = DEFAULT_VAD_NUM_THREADS,
    vad_debug: bool = False,
    vad_provider: str = "cpu",
    progress_callback: Callable[[float, float, int], None] | None = None,
    collect_observability: bool = False,
    process_group: ProcessGroup | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Recognize a WAV file using optimized Silero VAD and ZipFormer."""
    tokens = resolve_file(tokens_path, "Tokens file")
    encoder = resolve_file(encoder_path, "ZipFormer encoder")
    decoder = resolve_file(decoder_path, "ZipFormer decoder")
    joiner = resolve_file(joiner_path, "ZipFormer joiner")

    if decoding_method not in SUPPORTED_DECODING_METHODS:
        supported = ", ".join(sorted(SUPPORTED_DECODING_METHODS))
        raise ValueError(
            f"Unsupported decoding method {decoding_method!r}. Expected one of: {supported}"
        )

    model_arguments = [
        f"--tokens={tokens}",
        f"--encoder={encoder}",
        f"--decoder={decoder}",
        f"--joiner={joiner}",
        f"--decoding-method={decoding_method}",
        "--model-type=transducer",
    ]
    attempts: list[dict[str, Any]] = []
    for attempt, (silence_scale, duration_scale) in enumerate(VAD_ATTEMPT_SCALES, start=1):
        if process_group is not None:
            process_group.check_cancelled()
        min_silence = vad_min_silence_duration * silence_scale
        max_speech = vad_max_speech_duration * duration_scale
        if attempt > 1:
            if status_callback is not None:
                status_callback(
                    f"[重试 {attempt}/{len(VAD_ATTEMPT_SCALES)}] ZipFormer 长片段错误，"
                    f"重新识别当前文件：VAD 静音等待 {min_silence:g} 秒，"
                    f"软时长阈值 {max_speech:g} 秒"
                )
            if progress_callback is not None:
                # Discard the failed attempt's displayed progress before restarting.
                progress_callback(0.0, 0.0, 0)
        started = time.perf_counter() if collect_observability else None
        try:
            result = run_vad_asr(
                audio_path,
                executable_path=executable_path,
                vad_model_path=vad_model_path,
                model_arguments=model_arguments,
                num_threads=num_threads,
                debug=debug,
                provider=provider,
                vad_threshold=vad_threshold,
                vad_min_silence_duration=min_silence,
                vad_min_speech_duration=vad_min_speech_duration,
                vad_max_speech_duration=max_speech,
                vad_window_size=vad_window_size,
                vad_neg_threshold=vad_neg_threshold,
                vad_num_threads=vad_num_threads,
                vad_debug=vad_debug,
                vad_provider=vad_provider,
                failure_label="ZipFormer VAD recognition",
                progress_callback=progress_callback,
                collect_observability=collect_observability,
                process_group=process_group,
            )
        except RuntimeError as error:
            if process_group is not None:
                process_group.check_cancelled()
            if not _is_zipformer_length_error(error):
                raise
            if attempt == len(VAD_ATTEMPT_SCALES):
                raise RuntimeError(
                    f"ZipFormer 长片段错误：已尝试 {attempt} 次，仍然失败。\n{error}"
                ) from error
            outcome = "length_error"
        else:
            outcome = "success"
        if started is not None:
            attempts.append({
                "attempt": attempt,
                "outcome": outcome,
                "vad_min_silence_duration": min_silence,
                "vad_max_speech_duration": max_speech,
                "wall_seconds": round(time.perf_counter() - started, 6),
            })
        if outcome == "success":
            break
    if collect_observability:
        if len(attempts) > 1:
            result["_observability"]["retry"] = {
                "attempt_count": len(attempts),
                "max_attempts": len(VAD_ATTEMPT_SCALES),
                "attempts": attempts,
                "failed_attempts_wall_seconds": round(
                    sum(item["wall_seconds"] for item in attempts[:-1]), 6
                ),
                "scope_note": (
                    "各次尝试重新执行当前文件的 VAD+ZipFormer，复用已准备的 WAV。"
                    "进程资源和引擎耗时仅对应最后成功的一次；总处理耗时包含失败尝试。"
                ),
            }
        result["_observability"]["runtime"]["model_files"] = {
            "tokens": str(tokens),
            "encoder": str(encoder),
            "decoder": str(decoder),
            "joiner": str(joiner),
        }
    return result
