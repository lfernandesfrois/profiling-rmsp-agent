"""Build and write the aggregated JSON report for a profiled notebook run."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from profiling_rmsp_agent.deep_profile import DeepProfilePolicy


@dataclass(frozen=True, slots=True)
class FailedCell:
    cell_number: int
    ename: str
    evalue: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_number": self.cell_number,
            "ename": self.ename,
            "evalue": self.evalue,
        }


def build_report(
    notebook_path: Path,
    profiled_notebook_path: Path,
    cells: list[dict[str, Any]],
    failed_cell: FailedCell | None,
    deep_profile_policy: DeepProfilePolicy,
    deep_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cell_records = []
    for cell in cells:
        record = dict(cell)
        record["deep_profile_candidate"] = deep_profile_policy.is_candidate(record)
        cell_records.append(record)

    report = {
        "notebook": str(notebook_path),
        "profiled_notebook": str(profiled_notebook_path),
        "status": "failed" if failed_cell is not None else "completed",
        "failed_cell": failed_cell.to_dict() if failed_cell is not None else None,
        "cells": cell_records,
    }
    if deep_profile is not None:
        report["deep_profile"] = deep_profile
    return report


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
