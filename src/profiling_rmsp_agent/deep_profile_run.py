"""Replay candidate cells and attach Intel VTune Profiler by PID per candidate cell."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from profiling_rmsp_agent.notebook_run import _kernel_setup_options
from profiling_rmsp_agent.vtune_capture import VtuneCapture, VtuneCaptureError, WindowsVtuneProfiler


_PID_PROBE_SOURCE = "import os\nos.getpid()"


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
        "result_dir": None,
        "error": None,
    }


def _extract_int_result(executed_cell: Any) -> int | None:
    """Parse the PID probe cell's execute_result output (plain ``os.getpid()`` repr)."""
    for output in executed_cell.get("outputs", []):
        if output.get("output_type") == "execute_result":
            text = output.get("data", {}).get("text/plain")
            if text is not None:
                try:
                    value = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    return None
                return value if isinstance(value, int) and not isinstance(value, bool) else None
    return None


def _summarize(captures: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = [capture["status"] for capture in captures]
    return {
        "candidate_count": len(captures),
        "captured_count": sum(status in {"captured", "cell_failed"} for status in statuses),
        "capture_failed_count": sum(
            status in {"capture_failed", "stop_failed", "mapping_failed"} for status in statuses
        ),
        "not_reached_count": sum(status == "not_reached" for status in statuses),
    }


def run_candidate_replay(
    notebook_path: str | Path,
    candidates: list[dict[str, Any]],
    source_cell_indices: dict[int, int],
    vtune_profiler: WindowsVtuneProfiler,
    rmsp_path: Path | None = None,
) -> DeepProfileRunResult:
    """Replay *notebook_path* in a fresh kernel, sampling only candidate cells with VTune.

    The PID handed to VTune is obtained by executing a synthetic ``import os;
    os.getpid()`` cell as the very first thing in the fresh kernel: this is the
    exact OS process hosting the loaded C++ extension DLL, so no reliance on
    nbclient/kernel-manager internals is needed.
    """
    if not candidates:
        return DeepProfileRunResult(status="no_candidates", cells=[], summary=_summarize([]))

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
            record["error"] = "Could not map candidate execution count to source cell."
            global_error = record["error"]
            continue
        if notebook.cells[source_index].cell_type != "code":
            record["status"] = "mapping_failed"
            record["error"] = "The mapped candidate is not a code cell."
            global_error = record["error"]
            continue
        candidates_by_index[source_index] = (candidate, record)

    failed_cell: dict[str, Any] | None = None
    client = NotebookClient(notebook, timeout=None)

    try:
        with client.setup_kernel(**_kernel_setup_options(rmsp_path)):
            pid_cell = nbformat.v4.new_code_cell(source=_PID_PROBE_SOURCE)
            try:
                executed_pid_cell = client.execute_cell(pid_cell, 0)
            except Exception as error:
                raise VtuneCaptureError(
                    f"Could not execute PID probe cell: {type(error).__name__}: {error}"
                ) from error
            kernel_pid = _extract_int_result(executed_pid_cell)
            if kernel_pid is None:
                raise VtuneCaptureError("Could not determine the Jupyter kernel process ID.")

            code_cell_number = 0
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type != "code":
                    continue
                code_cell_number += 1

                candidate_and_record = candidates_by_index.get(index)
                capture: VtuneCapture | None = None

                if candidate_and_record is not None:
                    candidate, record = candidate_and_record
                    record["kernel_pid"] = kernel_pid
                    try:
                        capture = vtune_profiler.start_capture(
                            cell_number=int(candidate["cell_number"]),
                            notebook_cell_index=index,
                            kernel_pid=kernel_pid,
                        )
                        record["status"] = "recording"
                        record["result_dir"] = str(capture.result_dir)
                    except VtuneCaptureError as error:
                        record["status"] = "capture_failed"
                        record["error"] = str(error)
                        global_error = str(error)
                        break

                execution_error: BaseException | None = None
                try:
                    client.execute_cell(cell, index)
                except CellExecutionError as error:
                    execution_error = error
                    failed_cell = {
                        "cell_number": code_cell_number,
                        "ename": error.ename,
                        "evalue": error.evalue,
                    }
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
                            vtune_profiler.stop_capture(capture)
                            if execution_error is None:
                                record["status"] = "captured"
                            else:
                                record["status"] = "cell_failed"
                                if isinstance(execution_error, CellExecutionError):
                                    record["error"] = (
                                        f"{execution_error.ename}: {execution_error.evalue}"
                                    )
                                else:
                                    record["error"] = (
                                        f"{type(execution_error).__name__}: {execution_error}"
                                    )
                        except VtuneCaptureError as error:
                            record["status"] = "stop_failed"
                            record["error"] = str(error)
                            global_error = str(error)

                if execution_error is not None or global_error is not None:
                    break
    except VtuneCaptureError as error:
        global_error = str(error)
    except Exception as error:
        global_error = f"Deep-profile replay failed: {type(error).__name__}: {error}"

    summary = _summarize(capture_records)
    has_cell_failure = failed_cell is not None
    has_cell_errors = any(
        record["status"] in {"capture_failed", "stop_failed", "mapping_failed"}
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
