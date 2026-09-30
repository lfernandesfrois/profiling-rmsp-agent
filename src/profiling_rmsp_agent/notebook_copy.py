"""Create a sibling notebook copy instrumented with the live cell profiler."""

from dataclasses import dataclass
from pathlib import Path

import nbformat
from nbformat import NotebookNode

LOAD_EXT_LINE = "%load_ext profiling_rmsp_agent.live_cell_profiler"
TO_DICTS_LINE = "live_cell_profiler.to_dicts()"


class NotebookCopyError(ValueError):
    """Raised when a notebook copy cannot be safely prepared."""


@dataclass(frozen=True, slots=True)
class InstrumentedNotebook:
    output_path: Path
    load_ext_cell_inserted: bool
    to_dicts_cell_inserted: bool


def _cell_source(cell: NotebookNode) -> str:
    source = cell.source
    return "".join(source) if isinstance(source, list) else source


def _is_exact_code_cell(cell: NotebookNode, expected_source: str) -> bool:
    return cell.cell_type == "code" and _cell_source(cell).strip() == expected_source


def _read_notebook(source_path: Path) -> NotebookNode:
    try:
        notebook = nbformat.read(source_path, as_version=4)
        nbformat.validate(notebook)
    except Exception as error:
        raise NotebookCopyError(
            f"Could not read a valid Jupyter notebook from {source_path}: {error}"
        ) from error
    return notebook


def _write_without_overwriting(notebook: NotebookNode, output_path: Path) -> None:
    try:
        output_file = output_path.open("x", encoding="utf-8", newline="\n")
    except FileExistsError as error:
        raise NotebookCopyError(
            f"Refusing to overwrite existing file: {output_path}"
        ) from error
    except OSError as error:
        raise NotebookCopyError(
            f"Could not create output file {output_path}: {error}"
        ) from error

    try:
        with output_file:
            nbformat.write(notebook, output_file, version=4)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise


def create_instrumented_notebook(notebook_path: str | Path) -> InstrumentedNotebook:
    """Write a sibling ``_profile.ipynb`` copy with the profiler load/report lines."""
    source_path = Path(notebook_path)
    if source_path.suffix.lower() != ".ipynb":
        raise NotebookCopyError(f"Input must have an .ipynb extension: {source_path}")
    if not source_path.is_file():
        raise NotebookCopyError(f"Notebook file does not exist: {source_path}")

    output_path = source_path.with_name(f"{source_path.stem}_profile.ipynb")
    if output_path.exists():
        raise NotebookCopyError(f"Refusing to overwrite existing file: {output_path}")

    notebook = _read_notebook(source_path)
    cells = notebook.cells

    needs_load_ext = not (cells and _is_exact_code_cell(cells[0], LOAD_EXT_LINE))
    needs_to_dicts = not (cells and _is_exact_code_cell(cells[-1], TO_DICTS_LINE))

    if needs_load_ext:
        notebook.cells.insert(0, nbformat.v4.new_code_cell(source=LOAD_EXT_LINE))
    if needs_to_dicts:
        notebook.cells.append(nbformat.v4.new_code_cell(source=TO_DICTS_LINE))

    try:
        nbformat.validate(notebook)
    except Exception as error:
        raise NotebookCopyError(f"The transformed notebook is invalid: {error}") from error

    _write_without_overwriting(notebook, output_path)
    return InstrumentedNotebook(
        output_path=output_path,
        load_ext_cell_inserted=needs_load_ext,
        to_dicts_cell_inserted=needs_to_dicts,
    )
