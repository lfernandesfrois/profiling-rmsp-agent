import json
from pathlib import Path
import tempfile
import unittest

from profiling_rmsp_agent.deep_profile import DeepProfilePolicy
from profiling_rmsp_agent.report import FailedCell, build_report, write_report


POLICY = DeepProfilePolicy(
    min_total_time_seconds=1.0,
    min_native_python_ratio=1.0,
)


class BuildReportTests(unittest.TestCase):
    def test_completed_report_has_no_failed_cell(self) -> None:
        cells = [
            {
                "cell_number": 1,
                "total_time_seconds": 2.0,
                "native_time_seconds": 1.5,
                "python_time_seconds": 0.5,
            },
            {
                "cell_number": 2,
                "total_time_seconds": 3.0,
                "native_time_seconds": 0.5,
                "python_time_seconds": 1.0,
            },
        ]

        report = build_report(
            notebook_path=Path("nb.ipynb"),
            profiled_notebook_path=Path("nb_profile.ipynb"),
            cells=cells,
            failed_cell=None,
            deep_profile_policy=POLICY,
        )

        self.assertEqual(report["status"], "completed")
        self.assertIsNone(report["failed_cell"])
        self.assertEqual(report["cells"][0]["deep_profile_candidate"], True)
        self.assertEqual(report["cells"][1]["deep_profile_candidate"], False)
        self.assertNotIn("deep_profile_candidate", cells[0])
        self.assertEqual(report["cells"][0]["native_time_seconds"], 1.5)
        self.assertEqual(report["notebook"], "nb.ipynb")
        self.assertEqual(report["profiled_notebook"], "nb_profile.ipynb")

    def test_failed_report_includes_failed_cell_details(self) -> None:
        failed_cell = FailedCell(cell_number=3, ename="ValueError", evalue="boom")

        report = build_report(
            notebook_path=Path("nb.ipynb"),
            profiled_notebook_path=Path("nb_profile.ipynb"),
            cells=[
                {
                    "cell_number": 1,
                    "total_time_seconds": 2.0,
                    "native_time_seconds": 2.0,
                    "python_time_seconds": 0.0,
                },
                {"cell_number": 2},
            ],
            failed_cell=failed_cell,
            deep_profile_policy=POLICY,
        )

        self.assertEqual(report["status"], "failed")
        self.assertEqual(
            report["failed_cell"],
            {"cell_number": 3, "ename": "ValueError", "evalue": "boom"},
        )
        self.assertEqual(len(report["cells"]), 2)
        self.assertTrue(report["cells"][0]["deep_profile_candidate"])
        self.assertFalse(report["cells"][1]["deep_profile_candidate"])

    def test_deep_profile_section_preserves_timing_records(self) -> None:
        cell = {
            "cell_number": 4,
            "total_time_seconds": 2.0,
            "native_time_seconds": 1.5,
            "python_time_seconds": 0.5,
        }
        deep_profile = {
            "status": "completed",
            "summary": {"candidate_count": 1, "exported_count": 1},
            "cells": [{"cell_number": 4, "etl_path": "cell-4.etl"}],
        }

        report = build_report(
            notebook_path=Path("nb.ipynb"),
            profiled_notebook_path=Path("nb_profile.ipynb"),
            cells=[cell],
            failed_cell=None,
            deep_profile_policy=POLICY,
            deep_profile=deep_profile,
        )

        self.assertEqual(report["deep_profile"], deep_profile)
        self.assertEqual(report["cells"][0]["total_time_seconds"], 2.0)
        self.assertTrue(report["cells"][0]["deep_profile_candidate"])


class WriteReportTests(unittest.TestCase):
    def test_writes_valid_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "out.json"
            report = {"status": "completed", "cells": [], "failed_cell": None}

            write_report(report, output_path)

            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), report)


if __name__ == "__main__":
    unittest.main()
