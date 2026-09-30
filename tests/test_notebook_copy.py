from pathlib import Path
import tempfile
import unittest

import nbformat

from profiling_rmsp_agent.notebook_copy import (
    LOAD_EXT_LINE,
    TO_DICTS_LINE,
    NotebookCopyError,
    create_instrumented_notebook,
)


class CreateInstrumentedNotebookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_notebook(self, name: str, cells: list) -> Path:
        path = self.directory / name
        notebook = nbformat.v4.new_notebook(cells=cells)
        nbformat.write(notebook, path)
        return path

    def test_inserts_both_lines_when_missing(self) -> None:
        code = nbformat.v4.new_code_cell("value = 1")
        input_path = self.write_notebook("sample.ipynb", [code])
        original_bytes = input_path.read_bytes()

        result = create_instrumented_notebook(input_path)

        output = nbformat.read(result.output_path, as_version=4)
        self.assertEqual(result.output_path.name, "sample_profile.ipynb")
        self.assertTrue(result.load_ext_cell_inserted)
        self.assertTrue(result.to_dicts_cell_inserted)
        self.assertEqual(len(output.cells), 3)
        self.assertEqual(output.cells[0].source, LOAD_EXT_LINE)
        self.assertEqual(output.cells[1].source, "value = 1")
        self.assertEqual(output.cells[2].source, TO_DICTS_LINE)
        self.assertEqual(input_path.read_bytes(), original_bytes)
        nbformat.validate(output)

    def test_reuses_existing_load_ext_and_to_dicts_cells(self) -> None:
        load_ext = nbformat.v4.new_code_cell(LOAD_EXT_LINE)
        code = nbformat.v4.new_code_cell("value = 1")
        to_dicts = nbformat.v4.new_code_cell(TO_DICTS_LINE)
        input_path = self.write_notebook("existing.ipynb", [load_ext, code, to_dicts])

        result = create_instrumented_notebook(input_path)

        output = nbformat.read(result.output_path, as_version=4)
        self.assertFalse(result.load_ext_cell_inserted)
        self.assertFalse(result.to_dicts_cell_inserted)
        self.assertEqual(len(output.cells), 3)

    def test_inserts_load_ext_only_when_to_dicts_already_present(self) -> None:
        code = nbformat.v4.new_code_cell("value = 1")
        to_dicts = nbformat.v4.new_code_cell(TO_DICTS_LINE)
        input_path = self.write_notebook("half.ipynb", [code, to_dicts])

        result = create_instrumented_notebook(input_path)

        output = nbformat.read(result.output_path, as_version=4)
        self.assertTrue(result.load_ext_cell_inserted)
        self.assertFalse(result.to_dicts_cell_inserted)
        self.assertEqual(len(output.cells), 3)
        self.assertEqual(output.cells[0].source, LOAD_EXT_LINE)

    def test_refuses_to_overwrite_existing_output(self) -> None:
        code = nbformat.v4.new_code_cell("value = 1")
        input_path = self.write_notebook("collision.ipynb", [code])
        output_path = self.directory / "collision_profile.ipynb"
        output_path.write_text("keep existing file", encoding="utf-8")

        with self.assertRaisesRegex(NotebookCopyError, "Refusing to overwrite"):
            create_instrumented_notebook(input_path)

        self.assertEqual(output_path.read_text(encoding="utf-8"), "keep existing file")

    def test_rejects_non_notebook_extension(self) -> None:
        source_path = self.directory / "notes.py"
        source_path.write_text("value = 1", encoding="utf-8")

        with self.assertRaisesRegex(NotebookCopyError, r"\.ipynb"):
            create_instrumented_notebook(source_path)

    def test_rejects_missing_file(self) -> None:
        missing_path = self.directory / "missing.ipynb"

        with self.assertRaisesRegex(NotebookCopyError, "does not exist"):
            create_instrumented_notebook(missing_path)


if __name__ == "__main__":
    unittest.main()
