"""Convert one audio file to a mono 16-bit PCM WAV file with FFmpeg."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from .process_group import ProcessGroup
except ImportError:
    from process_group import ProcessGroup


def convert_audio_to_wav(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    sample_rate: int = 16_000,
    ffmpeg_executable: str | Path = "ffmpeg",
    overwrite: bool = False,
    process_group: ProcessGroup | None = None,
) -> Path:
    """Convert a single audio file to mono, 16-bit PCM WAV.

    Args:
        input_path: Source audio file. FFmpeg determines its input format.
        output_path: Destination WAV path. Defaults to ``<stem>.16k-mono.wav``.
        sample_rate: Output sample rate in Hz.
        ffmpeg_executable: FFmpeg executable name or path.
        overwrite: Replace an existing output file when true.

    Returns:
        The absolute path of the converted WAV file.
    """
    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input audio file does not exist: {source}")

    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than 0")

    if output_path is None:
        destination = source.with_name(f"{source.stem}.16k-mono.wav")
    else:
        destination = Path(output_path).expanduser().resolve()

    if destination.suffix.lower() != ".wav":
        raise ValueError(f"Output file must use the .wav extension: {destination}")
    if source == destination:
        raise ValueError("Input and output paths must be different")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"Output directory does not exist: {destination.parent}"
        )
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {destination}. Use overwrite=True to replace it."
        )

    ffmpeg = str(ffmpeg_executable)
    if not Path(ffmpeg).is_file():
        resolved_ffmpeg = shutil.which(ffmpeg)
        if resolved_ffmpeg is None:
            raise FileNotFoundError(f"FFmpeg executable was not found: {ffmpeg}")
        ffmpeg = resolved_ffmpeg

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y" if overwrite else "-n",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]

    if process_group is None:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    else:
        result = process_group.run(command, text=True)
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip() or "Unknown FFmpeg error"
        raise RuntimeError(f"FFmpeg conversion failed: {details}")

    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one audio file to a mono 16-bit PCM WAV file."
    )
    parser.add_argument("input", type=Path, help="source audio file")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="output WAV file (default: <input stem>.16k-mono.wav)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        help="output sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="FFmpeg executable name or path (default: ffmpeg from PATH)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the output file if it already exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output = convert_audio_to_wav(
            args.input,
            args.output,
            sample_rate=args.sample_rate,
            ffmpeg_executable=args.ffmpeg,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
