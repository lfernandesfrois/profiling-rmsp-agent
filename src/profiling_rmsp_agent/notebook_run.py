"""Execute an instrumented notebook end to end and collect profiler timings."""

import ast
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from profiling_rmsp_agent.notebook_copy import TO_DICTS_LINE
from profiling_rmsp_agent.report import FailedCell


@dataclass(frozen=True, slots=True)
class NotebookRunResult:
    cells: list[dict[str, Any]]
    failed_cell: FailedCell | None
    source_cell_indices: dict[int, int]


def _kernel_setup_options(rmsp_path: Path | None) -> dict[str, Any]:
    if rmsp_path is None:
        return {}
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    paths = existing.split(os.pathsep) if existing else []
    directory = str(rmsp_path)
    if not any(os.path.normcase(os.path.normpath(path)) == os.path.normcase(os.path.normpath(directory)) for path in paths):
        environment["PYTHONPATH"] = os.pathsep.join([directory, *paths])
    return {"env": environment}


def _extract_to_dicts_result(executed_cell: Any) -> list[dict[str, Any]]:
    for output in executed_cell.get("outputs", []):
        if output.get("output_type") == "execute_result":
            text = output.get("data", {}).get("text/plain")
            if text is not None:
                return ast.literal_eval(text)
    return []


def _fetch_partial_results(
    client: NotebookClient, failed_index: int
) -> list[dict[str, Any]]:
    # Best effort: the extension may not even be loaded yet if an early cell failed.
    try:
        fetch_cell = nbformat.v4.new_code_cell(source=TO_DICTS_LINE)
        executed = client.execute_cell(fetch_cell, failed_index)
        return _extract_to_dicts_result(executed)
    except Exception:
        return []


def run_notebook(
    notebook_path: str | Path, rmsp_path: Path | None = None
) -> NotebookRunResult:
    """Execute every cell of *notebook_path* and return the collected timings."""
    path = Path(notebook_path)
    notebook = nbformat.read(path, as_version=4)

    client = NotebookClient(notebook, timeout=None)
    source_cell_indices: dict[int, int] = {}
    code_cell_number = 0
    with client.setup_kernel(**_kernel_setup_options(rmsp_path)):
        executed = None
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            code_cell_number += 1
            try:
                executed = client.execute_cell(cell, index)
            except CellExecutionError as error:
                source_cell_indices[code_cell_number] = index
                failed_cell = FailedCell(
                    cell_number=code_cell_number,
                    ename=error.ename,
                    evalue=error.evalue,
                )
                cells = _fetch_partial_results(client, index)
                return NotebookRunResult(
                    cells=cells,
                    failed_cell=failed_cell,
                    source_cell_indices=source_cell_indices,
                )
            execution_count = getattr(executed, "execution_count", None)
            if isinstance(execution_count, int):
                source_cell_indices[execution_count] = index

        cells = _extract_to_dicts_result(executed) if executed is not None else []
        return NotebookRunResult(
            cells=cells,
            failed_cell=None,
            source_cell_indices=source_cell_indices,
        )
