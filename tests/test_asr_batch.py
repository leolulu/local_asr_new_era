from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import asr
from scripts.asr_common import DEFAULT_NUM_THREADS, DEFAULT_VAD_NUM_THREADS


def fake_result(source: Path, **kwargs: object) -> dict:
    text = f"{source.name}:{kwargs['model']}"
    kwargs["status_callback"]("正在识别")
    kwargs["progress_callback"](1.0, 1.0, 1)
    return {
        "model": kwargs["model"],
        "text": text,
        "segments": [{"start": 0.0, "end": 1.0, "text": text}],
        "metadata": {},
    }


class BatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("a.wav", "b.wav", "c.wav"):
            (self.root / name).touch()

    def invoke(self, *options: str, runner=fake_result, source=None):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch.object(sys, "argv", ["asr", str(source or self.root), *options]),
            patch.object(asr, "run_asr", side_effect=runner) as run,
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            code = asr.main()
        return code, stdout.getvalue(), stderr.getvalue(), run

    def test_default_runs_files_and_models_in_order_without_executor(self) -> None:
        with patch.object(asr, "ThreadPoolExecutor") as executor:
            code, stdout, stderr, run = self.invoke("--model", "all")
        executor.assert_not_called()
        self.assertEqual(code, 0)
        self.assertNotIn("[并发]", stderr)
        self.assertEqual(
            [(call.args[0].name, call.kwargs["model"]) for call in run.call_args_list],
            [(name, model) for name in ("a.wav", "b.wav", "c.wav") for model in asr.MODEL_ORDER],
        )
        self.assertTrue(all(
            result["metadata"]["invocation"]["file_workers_effective"] == 1
            for result in json.loads(stdout)
        ))

    def test_concurrency_limit_model_order_and_stable_result_order(self) -> None:
        barrier = threading.Barrier(2)
        b_finished = threading.Event()
        lock = threading.Lock()
        active: set[Path] = set()
        calls: list[tuple[str, str]] = []
        peak = 0

        def recognize(source, **kwargs):
            nonlocal peak
            with lock:
                self.assertNotIn(source, active)
                active.add(source)
                peak = max(peak, len(active))
                calls.append((source.name, kwargs["model"]))
            try:
                if source.name in ("a.wav", "b.wav") and kwargs["model"] == "sensevoice":
                    barrier.wait(timeout=5)
                if source.name == "a.wav" and kwargs["model"] == "sensevoice":
                    self.assertTrue(b_finished.wait(timeout=5))
                if source.name == "b.wav" and kwargs["model"] == "zipformer":
                    b_finished.set()
                return fake_result(source, **kwargs)
            finally:
                with lock:
                    active.remove(source)

        code, stdout, stderr, run = self.invoke(
            "--model", "all", "--file-workers", "2", runner=recognize,
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(peak, 2)
        self.assertLess(calls.index(("b.wav", "zipformer")), calls.index(("a.wav", "zipformer")))
        expected = [
            (name, model) for name in ("a.wav", "b.wav", "c.wav") for model in asr.MODEL_ORDER
        ]
        self.assertEqual(
            [(Path(item["input"]).name, item["model"]) for item in json.loads(stdout)],
            expected,
        )
        for name in ("a.wav", "b.wav", "c.wav"):
            self.assertEqual([model for file, model in calls if file == name], list(asr.MODEL_ORDER))
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["num_threads"], DEFAULT_NUM_THREADS)
            self.assertEqual(call.kwargs["vad_num_threads"], DEFAULT_VAD_NUM_THREADS)
        self.assertIn("[a.wav][sensevoice] [进度]", stderr)
        self.assertIn("[b.wav][zipformer] [完成]", stderr)
        self.assertNotIn("\r", stderr)

    def test_conflicts_fail_before_recognition_even_with_overwrite(self) -> None:
        (self.root / "a.mp3").touch()
        (self.root / "b.mp4").touch()
        for workers in ("1", "2"):
            for model in ("sensevoice", "all"):
                for overwrite in ([], ["--overwrite"]):
                    with self.subTest(workers=workers, model=model, overwrite=overwrite):
                        code, stdout, stderr, run = self.invoke(
                            "--file-workers", workers, "--model", model, "--all-output", *overwrite,
                        )
                        self.assertEqual(code, 1)
                        self.assertEqual(stdout, "")
                        run.assert_not_called()
                        for name in ("a.mp3", "a.wav", "b.mp4", "b.wav"):
                            self.assertIn(name, stderr)
                        suffix = ".sensevoice.srt" if model == "all" else ".srt"
                        self.assertIn("a" + suffix, stderr)
        self.assertEqual(list(self.root.glob("*.json")), [])

    def test_console_only_mode_has_no_output_path_conflict(self) -> None:
        (self.root / "a.mp3").touch()
        code, stdout, _, run = self.invoke("--file-workers", "2")
        self.assertEqual(code, 0)
        self.assertEqual(run.call_count, 4)
        self.assertEqual(len(json.loads(stdout)), 4)

    def test_failure_continues_other_models_and_files(self) -> None:
        def recognize(source, **kwargs):
            if source.name == "a.wav" and kwargs["model"] == "sensevoice":
                raise RuntimeError("broken input")
            return fake_result(source, **kwargs)

        code, stdout, stderr, run = self.invoke(
            "--model", "all", "--file-workers", "2", runner=recognize,
        )
        self.assertEqual(code, 1)
        self.assertEqual(run.call_count, 6)
        self.assertEqual(len(json.loads(stdout)), 5)
        self.assertIn("成功 5 个，跳过 0 个，失败 1 个", stderr)

    def test_skips_only_existing_model_and_writes_independent_outputs(self) -> None:
        existing = self.root / "a.sensevoice.srt"
        existing.write_text("keep", encoding="utf-8")
        code, stdout, stderr, run = self.invoke(
            "--model", "all", "--file-workers", "2", "--all-output",
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stdout, "")
        self.assertEqual(run.call_count, 5)
        self.assertEqual(existing.read_text(encoding="utf-8"), "keep")
        self.assertIn("成功 5 个，跳过 1 个，失败 0 个", stderr)
        for source in ("a", "b", "c"):
            for model in asr.MODEL_ORDER:
                if (source, model) == ("a", "sensevoice"):
                    continue
                stem = f"{source}.{model}"
                text = f"{source}.wav:{model}"
                self.assertEqual((self.root / f"{stem}.txt").read_text(encoding="utf-8"), text)
                self.assertIn(text, (self.root / f"{stem}.srt").read_text(encoding="utf-8"))
                result = json.loads((self.root / f"{stem}.json").read_text(encoding="utf-8"))
                self.assertEqual(result["text"], text)
                self.assertEqual(result["metadata"]["invocation"]["file_workers_effective"], 2)

    def test_explicit_thread_settings_and_overwrite_are_preserved(self) -> None:
        (self.root / "a.txt").write_text("old", encoding="utf-8")
        code, _, _, run = self.invoke(
            "--file-workers", "2", "--txt", "--overwrite",
            "--num-threads", "3", "--vad-num-threads", "2",
        )
        self.assertEqual(code, 0)
        self.assertEqual((self.root / "a.txt").read_text(encoding="utf-8"), "a.wav:sensevoice")
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["num_threads"], 3)
            self.assertEqual(call.kwargs["vad_num_threads"], 2)

    def test_single_file_keeps_single_result_shape_and_caps_workers(self) -> None:
        with patch.object(asr, "ThreadPoolExecutor") as executor:
            code, stdout, _, _ = self.invoke("--file-workers", "4", source=self.root / "a.wav")
        executor.assert_not_called()
        self.assertEqual(code, 0)
        result = json.loads(stdout)
        self.assertNotIn("input", result)
        self.assertEqual(result["metadata"]["invocation"]["file_workers_effective"], 1)

    def test_invalid_worker_count_is_rejected(self) -> None:
        for count in ("0", "-1", "bad"):
            with self.subTest(count=count), self.assertRaises(SystemExit) as error:
                self.invoke("--file-workers", count)
            self.assertEqual(error.exception.code, 2)

    def test_interrupt_stops_active_children_and_cleans_temporary_files(self) -> None:
        ready = threading.Barrier(3)
        children = []
        staged_directories = []
        lock = threading.Lock()

        def recognize(source, **kwargs):
            with tempfile.TemporaryDirectory() as staged:
                group = kwargs["process_group"]
                with group.open(
                    [sys.executable, "-c", "import time; time.sleep(60)"]
                ) as process:
                    with lock:
                        children.append(process)
                        staged_directories.append(Path(staged))
                    ready.wait(timeout=5)
                    process.wait(timeout=10)
                    group.check_cancelled()
            self.fail("cancelled recognition returned normally")

        def interrupt(*args, **kwargs):
            ready.wait(timeout=5)
            raise KeyboardInterrupt()

        with patch.object(asr, "wait", side_effect=interrupt):
            code, stdout, stderr, run = self.invoke(
                "--file-workers", "2", "--model", "all", "--all-output", runner=recognize,
            )
        self.assertEqual(code, 130)
        self.assertEqual(stdout, "")
        self.assertIn("[中断]", stderr)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(len(children), 2)
        self.assertTrue(all(process.poll() is not None for process in children))
        self.assertTrue(all(not path.exists() for path in staged_directories))
        self.assertEqual(list(self.root.glob("*.json")), [])

    def test_prefixed_progress_uses_lines_even_on_terminal(self) -> None:
        stream = io.StringIO()
        with redirect_stderr(stream), patch.object(stream, "isatty", return_value=True):
            progress = asr.ConsoleProgress(prefix="[a.wav][sensevoice] ")
            progress.start()
            progress.status("识别中")
            progress.update(1, 2, 1)
            progress.finish()
        self.assertNotIn("\r", stream.getvalue())
        self.assertTrue(all(
            line.startswith("[a.wav][sensevoice] ") for line in stream.getvalue().splitlines()
        ))


if __name__ == "__main__":
    unittest.main()
