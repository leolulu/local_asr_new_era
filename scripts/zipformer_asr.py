"""Run single-file speech recognition with Silero VAD and local ZipFormer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

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
    result = run_vad_asr(
        audio_path,
        executable_path=executable_path,
        vad_model_path=vad_model_path,
        model_arguments=model_arguments,
        num_threads=num_threads,
        debug=debug,
        provider=provider,
        vad_threshold=vad_threshold,
        vad_min_silence_duration=vad_min_silence_duration,
        vad_min_speech_duration=vad_min_speech_duration,
        vad_max_speech_duration=vad_max_speech_duration,
        vad_window_size=vad_window_size,
        vad_neg_threshold=vad_neg_threshold,
        vad_num_threads=vad_num_threads,
        vad_debug=vad_debug,
        vad_provider=vad_provider,
        failure_label="ZipFormer VAD recognition",
        progress_callback=progress_callback,
        collect_observability=collect_observability,
    )
    if collect_observability:
        result["_observability"]["runtime"]["model_files"] = {
            "tokens": str(tokens),
            "encoder": str(encoder),
            "decoder": str(decoder),
            "joiner": str(joiner),
        }
    return result
