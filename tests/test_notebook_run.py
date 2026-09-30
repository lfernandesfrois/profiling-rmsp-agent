import contextlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import nbformat

from profiling_rmsp_agent.notebook_copy import create_instrumented_notebook
from profiling_rmsp_agent.notebook_run import run_notebook

_HAS_KERNEL_DEPS = (
    importlib.util.find_spec("nbclient") is not None
    and importlib.util.find_spec("ipykernel") is not None
)

_KERNEL_NAME = "profiling-rmsp-agent-tests-kernel"


def _register_test_kernel() -> None:
    from ipykernel.kernelspec import install

    install(user=True, kernel_name=_KERNEL_NAME, display_name=_KERNEL_NAME)


def _remove_test_kernel() -> None:
    from jupyter_client.kernelspec import KernelSpecManager

    with contextlib.suppress(Exception):
        KernelSpecManager().remove_kernel_spec(_KERNEL_NAME)


@unittest.skipUnless(_HAS_KERNEL_DEPS, "nbclient/ipykernel not installed")
class NotebookRunIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _register_test_kernel()

    @classmethod
    def tearDownClass(cls) -> None:
        _remove_test_kernel()

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_notebook(self, name: str, sources: list) -> Path:
        cells = [nbformat.v4.new_code_cell(source) for source in sources]
        notebook = nbformat.v4.new_notebook(cells=cells)
        notebook.metadata["kernelspec"] = {
            "name": _KERNEL_NAME,
            "display_name": _KERNEL_NAME,
            "language": "python",
        }
        path = self.directory / name
        nbformat.write(notebook, path)
        return path

    def test_successful_run_collects_timings_for_every_cell(self) -> None:
        input_path = self._write_notebook(
            "demo.ipynb",
            ["x = 1\nfor i in range(1000):\n    x += i", "y = sum(range(10000))"],
        )
        instrumented = create_instrumented_notebook(input_path)

        result = run_notebook(instrumented.output_path)

        self.assertIsNone(result.failed_cell)
        self.assertEqual(len(result.cells), 2)
        self.assertEqual(result.cells[0]["cell_number"], 2)
        self.assertEqual(result.cells[1]["cell_number"], 3)
        self.assertEqual(result.source_cell_indices[2], 1)
        self.assertEqual(result.source_cell_indices[3], 2)

    def test_failed_cell_is_reported_with_partial_results(self) -> None:
        input_path = self._write_notebook(
            "error_demo.ipynb",
            ["x = 1", "raise ValueError('boom')", "y = 2"],
        )
        instrumented = create_instrumented_notebook(input_path)

        result = run_notebook(instrumented.output_path)

        self.assertIsNotNone(result.failed_cell)
        self.assertEqual(result.failed_cell.cell_number, 3)
        self.assertEqual(result.failed_cell.ename, "ValueError")
        self.assertEqual(result.failed_cell.evalue, "boom")
        self.assertEqual(len(result.cells), 2)
        self.assertEqual(result.source_cell_indices[3], 2)

    def test_execution_count_mapping_accounts_for_markdown_cells(self) -> None:
        notebook = nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell("before"),
                nbformat.v4.new_code_cell("first_value = 1"),
                nbformat.v4.new_markdown_cell("between"),
                nbformat.v4.new_code_cell("second_value = 2"),
            ]
        )
        notebook.metadata["kernelspec"] = {
            "name": _KERNEL_NAME,
            "display_name": _KERNEL_NAME,
            "language": "python",
        }
        input_path = self.directory / "markdown_demo.ipynb"
        nbformat.write(notebook, input_path)
        instrumented = create_instrumented_notebook(input_path)

        result = run_notebook(instrumented.output_path)

        self.assertIsNone(result.failed_cell)
        self.assertEqual(result.source_cell_indices[2], 2)
        self.assertEqual(result.source_cell_indices[3], 4)


if __name__ == "__main__":
    unittest.main()
