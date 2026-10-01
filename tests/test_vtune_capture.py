"""Tests for vtune_capture.py: all subprocess interaction is mocked, no real VTune is invoked."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from profiling_rmsp_agent.vtune_capture import (
    VtuneCapture,
    VtuneCaptureError,
    WindowsVtuneProfiler,
    resolve_vtune_config,
)


class FakePopen:
    """Minimal stand-in for subprocess.Popen used by start_capture/stop_capture."""

    def __init__(
        self,
        args,
        *,
        exit_code: int | None = None,
        stdout: str = "",
        stderr: str = "",
        hang_on_wait: bool = False,
        **kwargs,
    ) -> None:
        self.args = list(args)
        self._exit_code = exit_code
        self._stdout_text = stdout
        self._stderr_text = stderr
        self._hang_on_wait = hang_on_wait
        self.stdout = _FakeStream(stdout)
        self.stderr = _FakeStream(stderr)
        self.killed = False
        self.waited = False

    def poll(self):
        return self._exit_code

    def wait(self, timeout=None):
        self.waited = True
        if self._hang_on_wait:
            raise subprocess.TimeoutExpired(cmd=self.args, timeout=timeout)
        return self._exit_code or 0

    def kill(self):
        self.killed = True
        self._exit_code = -9


class _FakeStream:
    def __init__(self, text: str) -> None:
        self._text = text

    def read(self) -> str:
        return self._text


class FakeRunner:
    """Records subprocess.run-style calls and returns scripted results in order."""

    def __init__(self, results: list[subprocess.CompletedProcess] | None = None) -> None:
        self.calls: list[list[str]] = []
        self._results = list(results or [])

    def __call__(self, args, **kwargs):
        self.calls.append(list(args))
        if self._results:
            return self._results.pop(0)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")


def _write_config(directory: Path, document: dict) -> Path:
    config_path = directory / "profile_config.json"
    config_path.write_text(json.dumps(document), encoding="utf-8")
    return config_path


class ResolveVtuneConfigTests(unittest.TestCase):
    def test_defaults_when_no_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            notebook_path = Path(directory) / "nb.ipynb"
            notebook_path.write_text("{}", encoding="utf-8")

            config = resolve_vtune_config(notebook_path, config_path=None)

            self.assertEqual(config.executable, "vtune")
            self.assertEqual(config.sampling_mode, "hw")
            self.assertEqual(config.search_dirs, ())
            self.assertEqual(config.source_search_dirs, ())
            self.assertEqual(
                config.artifact_directory, notebook_path.parent / "nb_deep_profile_artifacts"
            )
            self.assertEqual(config.timeout_seconds, 120.0)
            self.assertEqual(config.start_grace_seconds, 1.0)

    def test_defaults_when_config_exists_but_no_vtune_section(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            notebook_path = Path(directory) / "nb.ipynb"
            notebook_path.write_text("{}", encoding="utf-8")
            _write_config(Path(directory), {"min_total_time_seconds": 1.0})

            config = resolve_vtune_config(notebook_path, config_path=None)

            self.assertEqual(config.executable, "vtune")
            self.assertEqual(config.sampling_mode, "hw")

    def test_overrides_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            notebook_path = Path(directory) / "nb.ipynb"
            notebook_path.write_text("{}", encoding="utf-8")
            search_dir = Path(directory) / "symbols"
            search_dir.mkdir()
            config_path = _write_config(
                Path(directory),
                {
                    "vtune": {
                        "executable": "C:/tools/vtune.exe",
                        "sampling_mode": "SW",
                        "search_dirs": ["symbols"],
                        "timeout_seconds": 30,
                        "start_grace_seconds": 0.5,
                    }
                },
            )

            config = resolve_vtune_config(notebook_path, config_path=config_path)

            self.assertEqual(config.executable, "C:/tools/vtune.exe")
            self.assertEqual(config.sampling_mode, "sw")
            self.assertEqual(config.search_dirs, (str(search_dir.resolve()),))
            self.assertEqual(config.timeout_seconds, 30.0)
            self.assertEqual(config.start_grace_seconds, 0.5)

    def test_invalid_sampling_mode_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            notebook_path = Path(directory) / "nb.ipynb"
            notebook_path.write_text("{}", encoding="utf-8")
            config_path = _write_config(
                Path(directory), {"vtune": {"sampling_mode": "ultra"}}
            )

            with self.assertRaisesRegex(VtuneCaptureError, "sampling_mode"):
                resolve_vtune_config(notebook_path, config_path=config_path)

    def test_missing_search_dir_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            notebook_path = Path(directory) / "nb.ipynb"
            notebook_path.write_text("{}", encoding="utf-8")
            config_path = _write_config(
                Path(directory), {"vtune": {"search_dirs": ["does-not-exist"]}}
            )

            with self.assertRaisesRegex(VtuneCaptureError, "search_dirs"):
                resolve_vtune_config(notebook_path, config_path=config_path)


class PreflightTests(unittest.TestCase):
    def test_resolves_executable_via_path_and_creates_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_directory = Path(directory) / "artifacts"
            from profiling_rmsp_agent.vtune_capture import VtuneConfig

            config = VtuneConfig(
                executable="vtune",
                sampling_mode="hw",
                search_dirs=(),
                source_search_dirs=(),
                artifact_directory=artifact_directory,
                timeout_seconds=5.0,
                start_grace_seconds=0.0,
            )
            runner = FakeRunner()
            profiler = WindowsVtuneProfiler(
                config,
                popen=FakePopen,
                runner=runner,
                executable_lookup=lambda name: "/usr/bin/vtune" if name == "vtune" else None,
                platform_name="win32",
            )

            run_directory = profiler.preflight()

            self.assertTrue(run_directory.is_dir())
            self.assertTrue(str(run_directory).startswith(str(artifact_directory)))
            self.assertEqual(runner.calls[0][0], "/usr/bin/vtune")
            self.assertEqual(runner.calls[0][1:], ["-help", "collect", "hotspots"])

    def test_non_windows_platform_raises(self) -> None:
        from profiling_rmsp_agent.vtune_capture import VtuneConfig

        config = VtuneConfig(
            executable="vtune",
            sampling_mode="hw",
            search_dirs=(),
            source_search_dirs=(),
            artifact_directory=Path("artifacts"),
            timeout_seconds=5.0,
            start_grace_seconds=0.0,
        )
        profiler = WindowsVtuneProfiler(
            config, popen=FakePopen, runner=FakeRunner(), platform_name="linux"
        )

        with self.assertRaisesRegex(VtuneCaptureError, "Windows"):
            profiler.preflight()

    def test_executable_not_found_raises(self) -> None:
        from profiling_rmsp_agent.vtune_capture import VtuneConfig

        config = VtuneConfig(
            executable="vtune",
            sampling_mode="hw",
            search_dirs=(),
            source_search_dirs=(),
            artifact_directory=Path("artifacts"),
            timeout_seconds=5.0,
            start_grace_seconds=0.0,
        )
        profiler = WindowsVtuneProfiler(
            config,
            popen=FakePopen,
            runner=FakeRunner(),
            executable_lookup=lambda name: None,
            platform_name="win32",
        )

        with self.assertRaisesRegex(VtuneCaptureError, "not found on PATH"):
            profiler.preflight()


def _make_profiler(
    directory: Path,
    *,
    popen=FakePopen,
    runner: FakeRunner | None = None,
    start_grace_seconds: float = 0.0,
) -> WindowsVtuneProfiler:
    from profiling_rmsp_agent.vtune_capture import VtuneConfig

    config = VtuneConfig(
        executable="vtune",
        sampling_mode="hw",
        search_dirs=("C:/symbols",),
        source_search_dirs=("C:/src",),
        artifact_directory=directory / "artifacts",
        timeout_seconds=5.0,
        start_grace_seconds=start_grace_seconds,
    )
    profiler = WindowsVtuneProfiler(
        config,
        popen=popen,
        runner=runner or FakeRunner(),
        executable_lookup=lambda name: "/usr/bin/vtune",
        platform_name="win32",
    )
    profiler.preflight()
    return profiler


class StartCaptureTests(unittest.TestCase):
    def test_builds_expected_command_and_returns_recording_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            captured_args = {}

            def popen_factory(args, **kwargs):
                captured_args["args"] = args
                return FakePopen(args, exit_code=None)

            profiler = _make_profiler(Path(directory), popen=popen_factory)

            capture = profiler.start_capture(cell_number=3, notebook_cell_index=5, kernel_pid=4242)

            self.assertEqual(capture.status, "recording")
            self.assertEqual(capture.kernel_pid, 4242)
            self.assertEqual(capture.cell_number, 3)
            args = captured_args["args"]
            self.assertEqual(args[0], "/usr/bin/vtune")
            self.assertIn("-collect", args)
            self.assertEqual(args[args.index("-collect") + 1], "hotspots")
            self.assertIn("-knob", args)
            self.assertEqual(args[args.index("-knob") + 1], "sampling-mode=hw")
            self.assertIn("-target-pid", args)
            self.assertEqual(args[args.index("-target-pid") + 1], "4242")
            self.assertIn("-result-dir", args)
            self.assertIn("-duration", args)
            self.assertEqual(args[args.index("-duration") + 1], "unlimited")
            self.assertIn("-search-dir", args)
            self.assertEqual(args[args.index("-search-dir") + 1], "C:/symbols")
            self.assertIn("-source-search-dir", args)
            self.assertEqual(args[args.index("-source-search-dir") + 1], "C:/src")

    def test_fast_fail_when_process_exits_during_grace_period(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            def popen_factory(args, **kwargs):
                return FakePopen(args, exit_code=1, stdout="bad pid", stderr="error detail")

            profiler = _make_profiler(Path(directory), popen=popen_factory)

            with self.assertRaisesRegex(VtuneCaptureError, "exited immediately"):
                profiler.start_capture(cell_number=1, notebook_cell_index=0, kernel_pid=999)

    def test_sequential_only_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profiler = _make_profiler(Path(directory))
            profiler.start_capture(cell_number=1, notebook_cell_index=0, kernel_pid=111)

            with self.assertRaisesRegex(VtuneCaptureError, "already active"):
                profiler.start_capture(cell_number=2, notebook_cell_index=1, kernel_pid=222)

    def test_refuses_to_reuse_existing_result_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profiler = _make_profiler(Path(directory))
            assert profiler.run_directory is not None
            (profiler.run_directory / "cell-1").mkdir()

            with self.assertRaisesRegex(VtuneCaptureError, "already exists"):
                profiler.start_capture(cell_number=1, notebook_cell_index=0, kernel_pid=111)


class StopCaptureTests(unittest.TestCase):
    def test_stop_invokes_command_and_marks_captured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = FakeRunner()
            profiler = _make_profiler(Path(directory), runner=runner)
            capture = profiler.start_capture(cell_number=1, notebook_cell_index=0, kernel_pid=111)
            # Simulate VTune having written result files before -command stop completes.
            capture.result_dir.mkdir(parents=True, exist_ok=True)
            (capture.result_dir / "r001hs.vtune").write_text("x", encoding="utf-8")

            result_dir = profiler.stop_capture(capture)

            self.assertEqual(result_dir, capture.result_dir)
            self.assertEqual(capture.status, "captured")
            stop_call = runner.calls[-1]
            self.assertIn("-command", stop_call)
            self.assertEqual(stop_call[stop_call.index("-command") + 1], "stop")
            self.assertIn("-result-dir", stop_call)
            self.assertEqual(stop_call[stop_call.index("-result-dir") + 1], str(capture.result_dir))

    def test_refuses_to_stop_capture_it_does_not_own(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profiler = _make_profiler(Path(directory))
            foreign_capture = VtuneCapture(
                cell_number=99,
                notebook_cell_index=0,
                kernel_pid=1,
                result_dir=Path(directory) / "cell-99",
            )

            with self.assertRaisesRegex(VtuneCaptureError, "not owned by this run"):
                profiler.stop_capture(foreign_capture)

    def test_stop_command_failure_marks_stop_failed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = FakeRunner(
                results=[
                    subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
                    subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="boom"),
                ]
            )
            profiler = _make_profiler(Path(directory), runner=runner)
            capture = profiler.start_capture(cell_number=1, notebook_cell_index=0, kernel_pid=111)

            with self.assertRaisesRegex(VtuneCaptureError, "boom"):
                profiler.stop_capture(capture)

            self.assertEqual(capture.status, "stop_failed")

    def test_process_not_exiting_after_stop_is_killed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            def popen_factory(args, **kwargs):
                return FakePopen(args, exit_code=None, hang_on_wait=True)

            profiler = _make_profiler(Path(directory), popen=popen_factory)
            capture = profiler.start_capture(cell_number=1, notebook_cell_index=0, kernel_pid=111)
            capture.result_dir.mkdir(parents=True, exist_ok=True)
            (capture.result_dir / "r001hs.vtune").write_text("x", encoding="utf-8")

            with self.assertRaisesRegex(VtuneCaptureError, "force-killed"):
                profiler.stop_capture(capture)

            self.assertEqual(capture.status, "stop_failed")


if __name__ == "__main__":
    unittest.main()
