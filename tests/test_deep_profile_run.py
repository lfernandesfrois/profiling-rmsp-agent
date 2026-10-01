"""Tests for deep_profile_run.py: NotebookClient and the VTune profiler are both faked."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

import nbformat
from nbclient.exceptions import CellExecutionError

from profiling_rmsp_agent.deep_profile_run import _PID_PROBE_SOURCE, run_candidate_replay
from profiling_rmsp_agent.vtune_capture import VtuneCapture, VtuneCaptureError


class _Scenario:
    """Shared state between the fake NotebookClient and assertions in a test."""

    def __init__(self, pid: int | None) -> None:
        self.pid = pid
        self.executed_indices: list[int] = []
        self.failures: dict[int, BaseException] = {}
        self.kernel_options: dict[str, Any] = {}

    def pid_cell_output(self) -> dict[str, Any]:
        if self.pid is None:
            return {"outputs": []}
        return {
            "outputs": [
                {"output_type": "execute_result", "data": {"text/plain": str(self.pid)}}
            ]
        }


class _FakeNotebookClient:
    def __init__(self, scenario: _Scenario, notebook: Any, timeout: float | None = None) -> None:
        self._scenario = scenario
        self.notebook = notebook

    def setup_kernel(self, **kwargs):
        self._scenario.kernel_options = kwargs
        return contextlib.nullcontext()

    def execute_cell(self, cell, index: int):
        source = cell.get("source", "")
        if source == _PID_PROBE_SOURCE:
            return self._scenario.pid_cell_output()
        self._scenario.executed_indices.append(index)
        failure = self._scenario.failures.get(index)
        if failure is not None:
            raise failure
        return {}


class FakeVtuneProfiler:
    def __init__(self) -> None:
        self.start_calls: list[tuple[int, int, int]] = []
        self.stop_calls: list[int] = []
        self.start_failures: dict[int, Exception] = {}
        self.stop_failures: dict[int, Exception] = {}

    def start_capture(self, cell_number: int, notebook_cell_index: int, kernel_pid: int) -> VtuneCapture:
        self.start_calls.append((cell_number, notebook_cell_index, kernel_pid))
        if cell_number in self.start_failures:
            raise self.start_failures[cell_number]
        return VtuneCapture(
            cell_number=cell_number,
            notebook_cell_index=notebook_cell_index,
            kernel_pid=kernel_pid,
            result_dir=Path(f"/fake/cell-{cell_number}"),
            status="recording",
        )

    def stop_capture(self, capture: VtuneCapture) -> Path:
        self.stop_calls.append(capture.cell_number)
        if capture.cell_number in self.stop_failures:
            raise self.stop_failures[capture.cell_number]
        return capture.result_dir


def _write_notebook(directory: Path, sources: list[str]) -> Path:
    cells = [nbformat.v4.new_code_cell(source) for source in sources]
    notebook = nbformat.v4.new_notebook(cells=cells)
    path = directory / "nb.ipynb"
    nbformat.write(notebook, path)
    return path


def _patched_client(scenario: _Scenario):
    return patch(
        "profiling_rmsp_agent.deep_profile_run.NotebookClient",
        lambda notebook, timeout=None: _FakeNotebookClient(scenario, notebook, timeout),
    )


class RunCandidateReplayTests(unittest.TestCase):
    def test_no_candidates_short_circuits_without_reading_notebook(self) -> None:
        profiler = FakeVtuneProfiler()

        result = run_candidate_replay("does-not-exist.ipynb", [], {}, profiler)

        self.assertEqual(result.status, "no_candidates")
        self.assertEqual(result.cells, [])
        self.assertEqual(profiler.start_calls, [])

    def test_pid_probe_executed_first_and_threaded_into_start_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            notebook_path = _write_notebook(Path(directory), ["a = 1", "b = 2"])
            scenario = _Scenario(pid=4321)
            profiler = FakeVtuneProfiler()
            candidates = [{"cell_number": 2, "total_time_seconds": 5.0}]
            source_cell_indices = {1: 0, 2: 1}

            with _patched_client(scenario):
                result = run_candidate_replay(
                    notebook_path, candidates, source_cell_indices, profiler
                )

            self.assertEqual(result.status, "completed")
            self.assertEqual(profiler.start_calls, [(2, 1, 4321)])
            self.assertEqual(result.cells[0]["kernel_pid"], 4321)
            self.assertEqual(result.cells[0]["status"], "captured")
            self.assertEqual(scenario.executed_indices, [0, 1])

    def test_rmsp_path_is_passed_to_replay_kernel_without_changing_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notebook_path = _write_notebook(root, ["import rmsp", "rmsp.run()"])
            scenario = _Scenario(pid=4321)
            profiler = FakeVtuneProfiler()
            original_environment = os.environ.copy()

            with _patched_client(scenario):
                result = run_candidate_replay(
                    notebook_path, [{"cell_number": 2}], {2: 1}, profiler,
                    rmsp_path=root,
                )

            self.assertEqual(result.status, "completed")
            self.assertEqual(
                scenario.kernel_options["env"]["PYTHONPATH"].split(os.pathsep)[0],
                str(root),
            )
            self.assertEqual(scenario.executed_indices, [0, 1])
            self.assertEqual(profiler.start_calls, [(2, 1, 4321)])
            self.assertEqual(os.environ, original_environment)

    def test_missing_pid_probe_result_fails_before_replay_or_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            notebook_path = _write_notebook(Path(directory), ["a = 1"])
            scenario = _Scenario(pid=None)
            profiler = FakeVtuneProfiler()

            with _patched_client(scenario):
                result = run_candidate_replay(
                    notebook_path,
                    [{"cell_number": 1}],
                    {1: 0},
                    profiler,
                )

            self.assertEqual(result.status, "failed")
            self.assertIn("Could not determine", result.error)
            self.assertEqual(profiler.start_calls, [])
            self.assertEqual(scenario.executed_indices, [])

    def test_only_candidate_cells_are_bracketed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            notebook_path = _write_notebook(Path(directory), ["a = 1", "b = 2", "c = 3"])
            scenario = _Scenario(pid=111)
            profiler = FakeVtuneProfiler()
            candidates = [{"cell_number": 2, "total_time_seconds": 5.0}]
            source_cell_indices = {1: 0, 2: 1, 3: 2}

            with _patched_client(scenario):
                result = run_candidate_replay(
                    notebook_path, candidates, source_cell_indices, profiler
                )

            self.assertEqual(result.status, "completed")
            self.assertEqual(len(profiler.start_calls), 1)
            self.assertEqual(profiler.start_calls[0][0], 2)
            self.assertEqual(scenario.executed_indices, [0, 1, 2])

    def test_cell_exception_still_stops_capture_and_halts_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            notebook_path = _write_notebook(Path(directory), ["a = 1", "boom", "c = 3"])
            scenario = _Scenario(pid=111)
            scenario.failures[1] = CellExecutionError(
                traceback="", ename="ValueError", evalue="boom"
            )
            profiler = FakeVtuneProfiler()
            candidates = [{"cell_number": 2, "total_time_seconds": 5.0}]
            source_cell_indices = {1: 0, 2: 1, 3: 2}

            with _patched_client(scenario):
                result = run_candidate_replay(
                    notebook_path, candidates, source_cell_indices, profiler
                )

            self.assertEqual(result.status, "failed")
            self.assertEqual(profiler.stop_calls, [2])
            self.assertEqual(result.cells[0]["status"], "cell_failed")
            self.assertIsNotNone(result.failed_cell)
            self.assertEqual(result.failed_cell["ename"], "ValueError")
            # Replay halts: the third cell (index 2) must not execute.
            self.assertEqual(scenario.executed_indices, [0, 1])

    def test_capture_start_failure_halts_replay_before_executing_cell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            notebook_path = _write_notebook(Path(directory), ["a = 1", "b = 2", "c = 3"])
            scenario = _Scenario(pid=111)
            profiler = FakeVtuneProfiler()
            profiler.start_failures[2] = VtuneCaptureError("could not attach")
            candidates = [{"cell_number": 2, "total_time_seconds": 5.0}]
            source_cell_indices = {1: 0, 2: 1, 3: 2}

            with _patched_client(scenario):
                result = run_candidate_replay(
                    notebook_path, candidates, source_cell_indices, profiler
                )

            self.assertEqual(result.status, "failed")
            self.assertEqual(result.cells[0]["status"], "capture_failed")
            self.assertEqual(profiler.stop_calls, [])
            # Cell index 1 (the candidate) must never have been executed.
            self.assertEqual(scenario.executed_indices, [0])

    def test_stop_failure_marks_stop_failed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            notebook_path = _write_notebook(Path(directory), ["a = 1", "b = 2"])
            scenario = _Scenario(pid=111)
            profiler = FakeVtuneProfiler()
            profiler.stop_failures[2] = VtuneCaptureError("stop failed")
            candidates = [{"cell_number": 2, "total_time_seconds": 5.0}]
            source_cell_indices = {1: 0, 2: 1}

            with _patched_client(scenario):
                result = run_candidate_replay(
                    notebook_path, candidates, source_cell_indices, profiler
                )

            self.assertEqual(result.status, "failed")
            self.assertEqual(result.cells[0]["status"], "stop_failed")

    def test_mapping_failed_does_not_touch_profiler(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            notebook_path = _write_notebook(Path(directory), ["a = 1", "b = 2"])
            scenario = _Scenario(pid=111)
            profiler = FakeVtuneProfiler()
            candidates = [{"cell_number": 99, "total_time_seconds": 5.0}]
            source_cell_indices = {1: 0}

            with _patched_client(scenario):
                result = run_candidate_replay(
                    notebook_path, candidates, source_cell_indices, profiler
                )

            self.assertEqual(result.status, "failed")
            self.assertEqual(result.cells[0]["status"], "mapping_failed")
            self.assertEqual(profiler.start_calls, [])

    def test_candidate_mapped_to_markdown_cell_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            notebook = nbformat.v4.new_notebook(
                cells=[
                    nbformat.v4.new_code_cell("a = 1"),
                    nbformat.v4.new_markdown_cell("not executable"),
                ]
            )
            notebook_path = Path(directory) / "nb.ipynb"
            nbformat.write(notebook, notebook_path)
            scenario = _Scenario(pid=111)
            profiler = FakeVtuneProfiler()

            with _patched_client(scenario):
                result = run_candidate_replay(
                    notebook_path,
                    [{"cell_number": 2}],
                    {2: 1},
                    profiler,
                )

            self.assertEqual(result.status, "failed")
            self.assertEqual(result.cells[0]["status"], "mapping_failed")
            self.assertEqual(profiler.start_calls, [])


if __name__ == "__main__":
    unittest.main()
