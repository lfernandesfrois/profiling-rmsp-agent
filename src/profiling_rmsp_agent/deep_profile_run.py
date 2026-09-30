"""Replay candidate cells and collect isolated WPR/WPA results."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from profiling_rmsp_agent.etw_capture import (
    EtwCaptureError,
    EtlCapture,
    WindowsEtwProfiler,
)
from profiling_rmsp_agent.wpa_export import summarize_kernel_cpu_csvs


@dataclass(frozen=True, slots=True)
class DeepProfileRunResult:
    status: str
    cells: list[dict[str, Any]]
    summary: dict[str, Any]
    failed_cell: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "failed_cell": self.failed_cell,
            "error": self.error,
            "cells": self.cells,
        }


def _new_capture_record(candidate: dict[str, Any], source_index: int | None) -> dict[str, Any]:
    return {
        "cell_number": candidate.get("cell_number"),
        "notebook_cell_index": source_index + 1 if source_index is not None else None,
        "kernel_pid": None,
        "status": "not_reached",
        "export_status": "not_started",
        "etl_path": None,
        "csv_files": [],
        "sampled_cpu_time_seconds": None,
        "matching_sample_rows": 0,
        "distinct_threads": 0,
        "error": None,
    }


def _failure_record(cell_number: int, error: CellExecutionError) -> dict[str, Any]:
    return {
        "cell_number": cell_number,
        "ename": error.ename,
        "evalue": error.evalue,
    }


def run_candidate_replay(
    notebook_path: str | Path,
    candidates: list[dict[str, Any]],
    source_cell_indices: dict[int, int],
    etw_profiler: WindowsEtwProfiler,
) -> DeepProfileRunResult:
    """Replay notebook code in a fresh kernel and capture candidate cells only."""
    path = Path(notebook_path)
    notebook = nbformat.read(path, as_version=4)
    capture_records: list[dict[str, Any]] = []
    candidates_by_index: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    global_error: str | None = None

    for candidate in candidates:
        cell_number = candidate.get("cell_number")
        source_index = source_cell_indices.get(cell_number)
        record = _new_capture_record(candidate, source_index)
        capture_records.append(record)
        if source_index is None or not 0 <= source_index < len(notebook.cells):
            record["status"] = "mapping_failed"
            record["error"] = "Could not map the candidate execution count to a source cell."
            global_error = record["error"]
            continue
        if notebook.cells[source_index].cell_type != "code":
            record["status"] = "mapping_failed"
            record["error"] = "The mapped candidate is not a code cell."
            global_error = record["error"]
            continue
        candidates_by_index[source_index] = (candidate, record)

    if not candidates:
        return DeepProfileRunResult(
            status="no_candidates",
            cells=[],
            summary=_summarize([], 0.0),
        )
    if global_error is not None:
        return DeepProfileRunResult(
            status="failed",
            cells=capture_records,
            summary=_summarize(capture_records, 0.0),
            error=global_error,
        )

    failed_cell: dict[str, Any] | None = None
    sampled_cpu_total = 0.0
    kernel_pid: int | None = None
    client = NotebookClient(notebook, timeout=None)
    try:
        with client.setup_kernel():
            provisioner = getattr(client.km, "provisioner", None)
            pid = getattr(provisioner, "pid", None)
            if not isinstance(pid, int) or pid < 1:
                raise EtwCaptureError("Could not determine the Jupyter kernel process ID.")
            kernel_pid = pid

            code_cell_number = 0
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type != "code":
                    continue
                code_cell_number += 1
                candidate_and_record = candidates_by_index.get(index)
                capture: EtlCapture | None = None

                if candidate_and_record is not None:
                    candidate, record = candidate_and_record
                    record["kernel_pid"] = kernel_pid
                    try:
                        capture = etw_profiler.start_capture(
                            cell_number=int(candidate["cell_number"]),
                            notebook_cell_index=index,
                            kernel_pid=kernel_pid,
                        )
                        record["status"] = "recording"
                        record["etl_path"] = str(capture.etl_path)
                    except EtwCaptureError as error:
                        record["status"] = "capture_failed"
                        record["error"] = str(error)
                        global_error = str(error)
                        break

                execution_error: BaseException | None = None
                try:
                    client.execute_cell(cell, index)
                except CellExecutionError as error:
                    execution_error = error
                    failed_cell = _failure_record(code_cell_number, error)
                except BaseException as error:
                    execution_error = error
                    failed_cell = {
                        "cell_number": code_cell_number,
                        "ename": type(error).__name__,
                        "evalue": str(error),
                    }
                finally:
                    if capture is not None:
                        try:
                            etw_profiler.stop_capture(capture)
                            if execution_error is None:
                                record["status"] = "captured"
                            else:
                                capture.status = "cell_failed"
                                if isinstance(execution_error, CellExecutionError):
                                    capture.error = (
                                        f"{execution_error.ename}: {execution_error.evalue}"
                                    )
                                else:
                                    capture.error = (
                                        f"{type(execution_error).__name__}: {execution_error}"
                                    )
                                record["status"] = "cell_failed"
                                record["error"] = capture.error
                        except EtwCaptureError as error:
                            record["status"] = "stop_failed"
                            record["error"] = str(error)
                            global_error = str(error)

                if execution_error is not None or global_error is not None:
                    break
    except EtwCaptureError as error:
        global_error = str(error)
    except Exception as error:
        global_error = f"Deep-profile replay failed: {type(error).__name__}: {error}"

    for index, (candidate, record) in candidates_by_index.items():
        if record["status"] not in {"captured", "cell_failed"}:
            continue
        etl_path = Path(record["etl_path"])
        csv_directory = etl_path.parent / f"cell-{index + 1:04d}-csv"
        try:
            csv_paths = etw_profiler.export_capture(etl_path, csv_directory)
            summary = summarize_kernel_cpu_csvs(
                csv_paths,
                etw_profiler.config,
                kernel_pid=int(record["kernel_pid"]),
            )
            record["csv_files"] = [str(csv_path) for csv_path in csv_paths]
            record["sampled_cpu_time_seconds"] = summary.sampled_cpu_time_seconds
            record["matching_sample_rows"] = summary.thread_rows
            record["distinct_threads"] = summary.distinct_threads
            record["export_status"] = "exported"
            sampled_cpu_total += summary.sampled_cpu_time_seconds
        except (EtwCaptureError, OSError) as error:
            record["export_status"] = "export_failed"
            record["error"] = "; ".join(
                message for message in (record["error"], str(error)) if message
            )

    summary = _summarize(capture_records, sampled_cpu_total)
    has_cell_failure = failed_cell is not None
    has_cell_errors = any(
        record["status"] in {"capture_failed", "stop_failed", "mapping_failed"}
        or record["export_status"] == "export_failed"
        for record in capture_records
    )
    status = (
        "failed"
        if global_error is not None or has_cell_failure
        else "completed_with_errors"
        if has_cell_errors
        else "completed"
    )
    return DeepProfileRunResult(
        status=status,
        cells=capture_records,
        summary=summary,
        failed_cell=failed_cell,
        error=global_error,
    )


def _summarize(captures: list[dict[str, Any]], sampled_cpu_total: float) -> dict[str, Any]:
    statuses = [capture["status"] for capture in captures]
    return {
        "candidate_count": len(captures),
        "captured_count": sum(
            status in {"captured", "cell_failed"} for status in statuses
        ),
        "exported_count": sum(
            capture["export_status"] == "exported" for capture in captures
        ),
        "capture_failed_count": sum(
            status in {"capture_failed", "stop_failed", "mapping_failed"}
            for status in statuses
        ),
        "export_failed_count": sum(
            capture["export_status"] == "export_failed" for capture in captures
        ),
        "not_reached_count": sum(status == "not_reached" for status in statuses),
        "total_sampled_cpu_time_seconds": sampled_cpu_total,
    }