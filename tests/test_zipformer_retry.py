from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import unittest
import wave
from concurrent.futures import CancelledError
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import asr, zipformer_asr
from scripts.process_group import ProcessGroup


LENGTH_ERROR = (
    "ZipFormer VAD recognition failed with exit code 3221226505: "
    "Non-zero status code returned while running Reshape node. "
    "Name:'/encoder/0/layers.0/self_attn_weights/Reshape_3' "
    "The input tensor cannot be reshaped to the requested shape. "
    "Input shape:{1,76,16}, requested shape:{-1,4151,4,4}"
)


def success(*args, **kwargs):
    result = {"text": "成功", "segments": [{"start": 0.0, "end": 1.0, "text": "成功"}]}
    if kwargs.get("collect_observability"):
        result["_observability"] = {"runtime": {}, "timing": {}, "resources": {}}
    return result


def write_wav(path):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 16001)


class ZipformerRetryTests(unittest.TestCase):
    def test_verified_length_signatures_and_saturated_position_table_retry(self):
        for available, requested in ((76, 4151), (185, 4369), (84, 4167), (2, 4003), (3999, 16001)):
            message = LENGTH_ERROR.replace("76,16", f"{available},16").replace("4151", str(requested))
            with self.subTest(available=available, requested=requested), patch.object(
                zipformer_asr, "run_vad_asr", side_effect=[RuntimeError(message), success()],
            ) as run:
                zipformer_asr.run_zipformer("unused.wav")
            self.assertEqual(run.call_count, 2)

    def test_first_success_preserves_settings_and_result(self):
        with patch.object(zipformer_asr, "run_vad_asr", side_effect=success) as run:
            result = zipformer_asr.run_zipformer(
                "unused.wav", vad_threshold=0.6, vad_min_silence_duration=0.7,
                vad_max_speech_duration=15, num_threads=3, vad_num_threads=2,
            )
        self.assertEqual(result, success())
        run.assert_called_once()
        self.assertEqual(run.call_args.kwargs["vad_threshold"], 0.6)
        self.assertEqual(run.call_args.kwargs["vad_min_silence_duration"], 0.7)
        self.assertEqual(run.call_args.kwargs["vad_max_speech_duration"], 15)
        self.assertEqual(run.call_args.kwargs["num_threads"], 3)
        self.assertEqual(run.call_args.kwargs["vad_num_threads"], 2)

    def test_stops_at_first_success_and_discards_failed_progress(self):
        def recognize(*args, **kwargs):
            kwargs["progress_callback"](30.0, 100.0, 5)
            if run.call_count < 3:
                raise RuntimeError(LENGTH_ERROR)
            return success(*args, **kwargs)

        updates = []
        messages = []
        with patch.object(zipformer_asr, "run_vad_asr", side_effect=recognize) as run:
            result = zipformer_asr.run_zipformer(
                "unused.wav", progress_callback=lambda *values: updates.append(values),
                status_callback=messages.append, collect_observability=True,
            )
        self.assertEqual(run.call_count, 3)
        self.assertEqual(result["segments"], success()["segments"])
        self.assertEqual(updates, [(30.0, 100.0, 5), (0.0, 0.0, 0),
                                  (30.0, 100.0, 5), (0.0, 0.0, 0), (30.0, 100.0, 5)])
        self.assertIn("2/4", messages[0])
        self.assertIn("3/4", messages[1])
        retry = result["_observability"]["retry"]
        self.assertEqual(retry["attempt_count"], 3)
        self.assertEqual([item["outcome"] for item in retry["attempts"]],
                         ["length_error", "length_error", "success"])

    def test_four_attempt_limit_and_parameter_isolation(self):
        with patch.object(zipformer_asr, "run_vad_asr", side_effect=RuntimeError(LENGTH_ERROR)) as run:
            with self.assertRaisesRegex(RuntimeError, "已尝试 4 次") as error:
                zipformer_asr.run_zipformer("unused.wav", num_threads=3, vad_num_threads=2)
        self.assertEqual(run.call_count, 4)
        self.assertIn("Reshape_3", str(error.exception))
        self.assertEqual(
            [(call.kwargs["vad_min_silence_duration"], call.kwargs["vad_max_speech_duration"])
             for call in run.call_args_list],
            [(1.0, 20.0), (0.2, 10.0), (0.1, 5.0), (0.05, 2.5)],
        )
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["num_threads"], 3)
            self.assertEqual(call.kwargs["vad_num_threads"], 2)
            self.assertEqual(call.kwargs["vad_threshold"], 0.5)
            self.assertEqual(call.kwargs["vad_neg_threshold"], -1.0)
        with patch.object(zipformer_asr, "run_vad_asr", side_effect=success) as next_run:
            zipformer_asr.run_zipformer("next.wav")
        self.assertEqual(next_run.call_args.kwargs["vad_min_silence_duration"], 1.0)

    def test_custom_durations_are_scaled_without_being_loosened(self):
        with patch.object(zipformer_asr, "run_vad_asr", side_effect=[
            RuntimeError(LENGTH_ERROR), success(),
        ]) as run:
            zipformer_asr.run_zipformer("unused.wav", vad_min_silence_duration=0.1,
                                       vad_max_speech_duration=2)
        self.assertAlmostEqual(run.call_args.kwargs["vad_min_silence_duration"], 0.02)
        self.assertEqual(run.call_args.kwargs["vad_max_speech_duration"], 1.0)

    def test_other_errors_never_trigger_retry(self):
        errors = [
            RuntimeError("out of memory"),
            RuntimeError(LENGTH_ERROR.replace("Reshape_3", "Reshape_2")),
            RuntimeError(LENGTH_ERROR.replace("4151", "3997")),
            RuntimeError(LENGTH_ERROR.replace("76,16", "77,16")),
            RuntimeError(LENGTH_ERROR.replace("ZipFormer VAD recognition failed", "unexpected output")),
            RuntimeError(LENGTH_ERROR.split("Name:")[0]),
            ValueError("bad parameter"), CancelledError(), KeyboardInterrupt(),
        ]
        for error in errors:
            with self.subTest(error=repr(error)), patch.object(
                zipformer_asr, "run_vad_asr", side_effect=error,
            ) as run:
                with self.assertRaises(type(error)) as raised:
                    zipformer_asr.run_zipformer("unused.wav")
                self.assertIs(raised.exception, error)
                run.assert_called_once()

    def test_new_error_during_retry_stops_immediately(self):
        with patch.object(zipformer_asr, "run_vad_asr", side_effect=[
            RuntimeError(LENGTH_ERROR), RuntimeError("out of memory"),
        ]) as run:
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                zipformer_asr.run_zipformer("unused.wav")
        self.assertEqual(run.call_count, 2)

    def test_cancel_after_failed_attempt_prevents_retry(self):
        group = ProcessGroup()

        def recognize(*args, **kwargs):
            group.cancel()
            raise RuntimeError(LENGTH_ERROR)

        with patch.object(zipformer_asr, "run_vad_asr", side_effect=recognize) as run:
            with self.assertRaises(CancelledError):
                zipformer_asr.run_zipformer("unused.wav", process_group=group)
        run.assert_called_once()

    def test_wav_is_prepared_once_and_metadata_records_effective_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.mp3"
            source.touch()

            def convert(source, destination, **kwargs):
                write_wav(destination)
                return destination

            def recognize(*args, **kwargs):
                self.assertTrue(Path(args[0]).exists())
                if run.call_count == 1:
                    raise RuntimeError(LENGTH_ERROR)
                return success(*args, **kwargs)

            with (
                patch.object(asr, "convert_audio_to_wav", side_effect=convert) as conversion,
                patch.object(zipformer_asr, "run_vad_asr", side_effect=recognize) as run,
            ):
                result = asr.run_asr(source, model="zipformer", collect_observability=True)
            conversion.assert_called_once()
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args_list[0].args[0], run.call_args_list[1].args[0])
            self.assertFalse(Path(run.call_args.args[0]).exists())
        metadata = result["metadata"]
        self.assertEqual(metadata["retry"]["attempt_count"], 2)
        self.assertEqual(metadata["configuration"]["vad"]["min_silence_duration"], 1.0)
        self.assertEqual(metadata["configuration"]["vad_effective"]["min_silence_duration"], 0.2)
        self.assertIn("failed_attempts_wall_seconds", metadata["timing"])

    def test_batch_continues_after_retry_exhaustion_in_serial_and_parallel_modes(self):
        for workers in (1, 2):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for name in ("a.wav", "b.wav"):
                    write_wav(root / name)
                counts = {"a.wav": 0, "b.wav": 0}
                lock = threading.Lock()

                def recognize(source, **kwargs):
                    with lock:
                        counts[source.name] += 1
                    if source.name == "a.wav":
                        raise RuntimeError(LENGTH_ERROR)
                    self.assertEqual(kwargs["vad_min_silence_duration"], 1.0)
                    return success(source, **kwargs)

                stderr = io.StringIO()
                with (
                    patch.object(sys, "argv", ["asr", str(root), "--model", "all", "--json",
                                                "--file-workers", str(workers)]),
                    patch.object(zipformer_asr, "run_vad_asr", side_effect=recognize),
                    patch.object(asr, "run_sensevoice", side_effect=success) as sensevoice,
                    redirect_stdout(io.StringIO()), redirect_stderr(stderr),
                ):
                    code = asr.main()
                self.assertEqual(code, 1)
                self.assertEqual(counts, {"a.wav": 4, "b.wav": 1})
                self.assertEqual(sensevoice.call_count, 2)
                self.assertIn("成功 3 个，跳过 0 个，失败 1 个", stderr.getvalue())
                self.assertFalse((root / "a.zipformer.json").exists())
                self.assertTrue((root / "b.zipformer.json").exists())
                self.assertNotIn("retry", json.loads((root / "b.zipformer.json").read_text(encoding="utf-8"))["metadata"])

    def test_progress_reports_again_after_retry_reset(self):
        stream = io.StringIO()
        with redirect_stderr(stream):
            progress = asr.ConsoleProgress(prefix="[a.wav][zipformer] ")
            progress.start()
            progress.update(80, 100, 8)
            progress.update(0, 0, 0)
            progress.update(10, 100, 1)
            progress.abort()
        self.assertIn("80.0%", stream.getvalue())
        self.assertIn("10.0%", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
