import csv
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import nbformat

from profiling_rmsp_agent.deep_profile_run import run_candidate_replay
from profiling_rmsp_agent.etw_capture import EtlCapture, EtwConfig


class FakeExecutionError(Exception):
    def __init__(self, cell_number: int) -> None:
        super().__init__("cell failed")
        self.ename = "ValueError"
        self.evalue = "cell failed"
        self.cell_number = cell_number


class FakeNotebookClient:
    fail_index: int | None = None
    interrupt_index: int | None = None
    events: list[tuple[str, int]] = []

    def __init__(self, notebook: object, timeout: float | None = None) -> None:
        self.notebook = notebook
        self.km = SimpleNamespace(provisioner=SimpleNamespace(pid=7001))

    def setup_kernel(self):
        return nullcontext()

    def execute_cell(self, cell: object, index: int) -> object:
        self.events.append(("execute", index))
        if index == self.fail_index:
            raise FakeExecutionError(index)
        if index == self.interrupt_index:
            raise KeyboardInterrupt("interrupted")
        return cell


class FakeEtwProfiler:
    def __init__(self, directory: Path, events: list[tuple[str, int]]) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.events = events
        self.config = EtwConfig(
            wpr_executable="wpr.exe",
            wpr_profile="CPU",
            wpaexporter_executable="wpaexporter.exe",
            wpa_export_profile=directory / "cpu.wpaProfile",
            artifact_directory=directory,
            cpu_csv_pattern="*cpu-sampled.csv",
            process_id_column="PID",
            thread_id_column="TID",
            cpu_time_column="Weight (ms)",
            cpu_time_unit="ms",
        )

    def start_capture(
        self, cell_number: int, notebook_cell_index: int, kernel_pid: int
    ) -> EtlCapture:
        self.events.append(("start", notebook_cell_index))
        return EtlCapture(
            cell_number=cell_number,
            notebook_cell_index=notebook_cell_index,
            kernel_pid=kernel_pid,
            etl_path=self.directory / f"cell-{cell_number}.etl",
            instance_name=f"test-{cell_number}",
            status="recording",
        )

    def stop_capture(self, capture: EtlCapture) -> Path:
        self.events.append(("stop", capture.notebook_cell_index))
        capture.etl_path.touch()
        capture.status = "captured"
        return capture.etl_path

    def export_capture(self, etl_path: Path, output_directory: Path) -> list[Path]:
        self.events.append(("export", etl_path))
        output_directory.mkdir(parents=True)
        csv_path = output_directory / "cpu-sampled.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["PID", "TID", "Weight (ms)"])
            writer.writeheader()
            writer.writerow({"PID": "7001", "TID": "12", "Weight (ms)": "4"})
            writer.writerow({"PID": "9000", "TID": "99", "Weight (ms)": "100"})
        return [csv_path]


class CandidateReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        notebook = nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell("heading"),
                nbformat.v4.new_code_cell("setup_state = 1"),
                nbformat.v4.new_markdown_cell("more context"),
                nbformat.v4.new_code_cell("target_work()"),
                nbformat.v4.new_code_cell("later_work()"),
            ]
        )
        self.notebook_path = self.directory / "replay.ipynb"
        nbformat.write(notebook, self.notebook_path)
        self.events: list[tuple[str, int]] = []
        FakeNotebookClient.events = self.events
        FakeNotebookClient.fail_index = None
        FakeNotebookClient.interrupt_index = None
        self.profiler = FakeEtwProfiler(self.directory / "etw", self.events)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def run_replay(self, candidates: list[dict[str, object]], mapping: dict[int, int]):
        with patch(
            "profiling_rmsp_agent.deep_profile_run.NotebookClient", FakeNotebookClient
        ), patch(
            "profiling_rmsp_agent.deep_profile_run.CellExecutionError",
            FakeExecutionError,
        ):
            return run_candidate_replay(
                self.notebook_path,
                candidates,
                mapping,
                self.profiler,
            )

    def test_only_candidate_code_cell_is_bracketed_and_csv_is_aggregated(self) -> None:
        result = self.run_replay(
            [{"cell_number": 2, "deep_profile_candidate": True}],
            {2: 3},
        )

        self.assertIsNone(result.failed_cell)
        self.assertEqual(
            self.events,
            [
                ("execute", 1),
                ("start", 3),
                ("execute", 3),
                ("stop", 3),
                ("execute", 4),
                ("export", self.directory / "etw" / "cell-2.etl"),
            ],
        )
        capture = result.cells[0]
        self.assertEqual(capture["status"], "captured")
        self.assertEqual(capture["export_status"], "exported")
        self.assertEqual(capture["sampled_cpu_time_seconds"], 0.004)
        self.assertEqual(result.summary["total_sampled_cpu_time_seconds"], 0.004)

    def test_candidate_error_stops_replay_after_stopping_and_exporting_capture(self) -> None:
        FakeNotebookClient.fail_index = 3
        result = self.run_replay(
            [
                {"cell_number": 2, "deep_profile_candidate": True},
                {"cell_number": 3, "deep_profile_candidate": True},
            ],
            {2: 3, 3: 4},
        )

        self.assertEqual(result.failed_cell["cell_number"], 2)
        self.assertEqual(result.cells[0]["status"], "cell_failed")
        self.assertEqual(result.cells[0]["export_status"], "exported")
        self.assertEqual(result.cells[1]["status"], "not_reached")
        self.assertEqual(
            [event[0] for event in self.events],
            ["execute", "start", "execute", "stop", "export"],
        )

    def test_unmapped_candidate_is_reported_without_running_replay(self) -> None:
        result = self.run_replay(
            [{"cell_number": 99, "deep_profile_candidate": True}],
            {},
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.cells[0]["status"], "mapping_failed")
        self.assertEqual(self.events, [])

    def test_keyboard_interrupt_still_stops_and_exports_active_capture(self) -> None:
        FakeNotebookClient.interrupt_index = 3
        result = self.run_replay(
            [{"cell_number": 2, "deep_profile_candidate": True}],
            {2: 3},
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failed_cell["ename"], "KeyboardInterrupt")
        self.assertEqual(result.cells[0]["status"], "cell_failed")
        self.assertEqual(result.cells[0]["export_status"], "exported")
        self.assertEqual(
            [event[0] for event in self.events],
            ["execute", "start", "execute", "stop", "export"],
        )


if __name__ == "__main__":
    unittest.main()