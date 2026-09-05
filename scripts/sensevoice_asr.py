"""Run single-file speech recognition with the local SenseVoice model."""

from __future__ import annotations

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
        run_json_asr,
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
        run_json_asr,
        run_vad_asr,
    )


DEFAULT_EXECUTABLE_PATH = (
    PROJECT_ROOT / "app" / "sherpa-onnx" / "bin" / "sherpa-onnx-offline.exe"
)
DEFAULT_TOKENS_PATH = PROJECT_ROOT / "models" / "sensevoice" / "tokens.txt"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "sensevoice" / "model.int8.onnx"
SUPPORTED_LANGUAGES = {"auto", "zh", "en", "ja", "ko", "yue"}


def run_sensevoice(
    audio_path: str | Path,
    *,
    executable_path: str | Path = DEFAULT_EXECUTABLE_PATH,
    tokens_path: str | Path = DEFAULT_TOKENS_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    num_threads: int = DEFAULT_NUM_THREADS,
    language: str = "auto",
    use_itn: bool = True,
    debug: bool = False,
    provider: str = "cpu",
    use_vad: bool = True,
    vad_executable_path: str | Path = DEFAULT_VAD_EXECUTABLE_PATH,
    vad_model_path: str | Path = DEFAULT_VAD_MODEL_PATH,
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
) -> dict[str, Any]:
    """Recognize a WAV file with SenseVoice, using optimized VAD by default.

    VAD mode returns ``text`` and timestamped ``segments``. Set
    ``use_vad=False`` to retain the direct recognizer's original JSON fields.
    """
    tokens = resolve_file(tokens_path, "Tokens file")
    model = resolve_file(model_path, "SenseVoice model")

    normalized_language = language.lower()
    if normalized_language not in SUPPORTED_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ValueError(f"Unsupported language {language!r}. Expected one of: {supported}")

    model_arguments = [
        f"--tokens={tokens}",
        f"--sense-voice-model={model}",
        f"--sense-voice-language={normalized_language}",
        f"--sense-voice-use-itn={str(use_itn).lower()}",
    ]
    if use_vad:
        result = run_vad_asr(
            audio_path,
            executable_path=vad_executable_path,
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
            failure_label="SenseVoice VAD recognition",
            progress_callback=progress_callback,
            collect_observability=collect_observability,
            process_group=process_group,
        )
        if collect_observability:
            result["_observability"]["runtime"]["model_files"] = {
                "tokens": str(tokens),
                "sensevoice_model": str(model),
            }
        return result

    return run_json_asr(
        audio_path,
        executable_path=executable_path,
        model_arguments=model_arguments,
        num_threads=num_threads,
        debug=debug,
        provider=provider,
        failure_label="SenseVoice",
        process_group=process_group,
    )
