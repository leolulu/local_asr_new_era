from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from scripts.asr_common import vad_compatible_audio_path


def write_test_wav(path: Path, frame_count: int) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\x01\x02" * frame_count)


class VadCompatibleAudioPathTests(unittest.TestCase):
    def test_divisible_frame_count_gets_one_silent_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.wav"
            write_test_wav(source, 1024)
            original_bytes = source.read_bytes()

            with vad_compatible_audio_path(source, 512) as staged:
                self.assertNotEqual(staged, source)
                with wave.open(str(staged), "rb") as wav_file:
                    self.assertEqual(wav_file.getnframes(), 1025)
                    wav_file.setpos(1024)
                    self.assertEqual(wav_file.readframes(1), b"\x00\x00")

            self.assertEqual(source.read_bytes(), original_bytes)
            self.assertFalse(staged.exists())

    def test_non_divisible_frame_count_is_left_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.wav"
            write_test_wav(source, 1025)
            original_bytes = source.read_bytes()

            with vad_compatible_audio_path(source, 512) as staged:
                self.assertEqual(staged, source)
                with wave.open(str(staged), "rb") as wav_file:
                    self.assertEqual(wav_file.getnframes(), 1025)

            self.assertEqual(source.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
