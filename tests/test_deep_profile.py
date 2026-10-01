import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from profiling_rmsp_agent.__main__ import main
from profiling_rmsp_agent.deep_profile import (
    DEFAULT_CONFIG_NAME,
    DeepProfileConfigError,
    DeepProfilePolicy,
    resolve_deep_profile_policy,
    resolve_rmsp_path,
)


class DeepProfilePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = DeepProfilePolicy(1.0, 1.0)

    def test_candidate_requires_both_strict_thresholds(self) -> None:
        self.assertTrue(
            self.policy.is_candidate(
                {
                    "total_time_seconds": 1.01,
                    "native_time_seconds": 2.0,
                    "python_time_seconds": 1.0,
                }
            )
        )

    def test_threshold_boundaries_are_strict(self) -> None:
        self.assertFalse(
            self.policy.is_candidate(
                {
                    "total_time_seconds": 1.0,
                    "native_time_seconds": 2.0,
                    "python_time_seconds": 1.0,
                }
            )
        )
        self.assertFalse(
            self.policy.is_candidate(
                {
                    "total_time_seconds": 2.0,
                    "native_time_seconds": 1.0,
                    "python_time_seconds": 1.0,
                }
            )
        )

    def test_native_python_ratio_above_and_below_threshold(self) -> None:
        above = {
            "total_time_seconds": 2.0,
            "native_time_seconds": 2.1,
            "python_time_seconds": 2.0,
        }
        below = {
            "total_time_seconds": 2.0,
            "native_time_seconds": 1.9,
            "python_time_seconds": 2.0,
        }

        self.assertTrue(self.policy.is_candidate(above))
        self.assertFalse(self.policy.is_candidate(below))

    def test_zero_python_time_does_not_divide_by_zero(self) -> None:
        self.assertTrue(
            self.policy.is_candidate(
                {
                    "total_time_seconds": 2.0,
                    "native_time_seconds": 0.1,
                    "python_time_seconds": 0.0,
                }
            )
        )
        self.assertFalse(
            self.policy.is_candidate(
                {
                    "total_time_seconds": 2.0,
                    "native_time_seconds": 0.0,
                    "python_time_seconds": 0.0,
                }
            )
        )

    def test_missing_or_invalid_cell_times_are_not_candidates(self) -> None:
        self.assertFalse(self.policy.is_candidate({}))
        self.assertFalse(
            self.policy.is_candidate(
                {
                    "total_time_seconds": float("nan"),
                    "native_time_seconds": 2.0,
                    "python_time_seconds": 1.0,
                }
            )
        )

    def test_invalid_thresholds_are_rejected(self) -> None:
        for invalid in (-1.0, float("inf"), float("nan"), True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(DeepProfileConfigError):
                    DeepProfilePolicy(invalid, 1.0)


class ResolveDeepProfilePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.notebook_path = self.directory / "notebook.ipynb"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_config(self, values: object, name: str = DEFAULT_CONFIG_NAME) -> Path:
        path = self.directory / name
        path.write_text(json.dumps(values), encoding="utf-8")
        return path

    def resolve(
        self,
        config_path: Path | None = None,
        min_total_time_seconds: float | None = None,
        min_native_python_ratio: float | None = None,
    ) -> DeepProfilePolicy:
        return resolve_deep_profile_policy(
            notebook_path=self.notebook_path,
            config_path=config_path,
            cli_min_total_time_seconds=min_total_time_seconds,
            cli_min_native_python_ratio=min_native_python_ratio,
        )

    def test_loads_config_from_notebook_directory(self) -> None:
        self.write_config(
            {"min_total_time_seconds": 2.0, "min_native_python_ratio": 0.75}
        )

        policy = self.resolve()

        self.assertEqual(policy.min_total_time_seconds, 2.0)
        self.assertEqual(policy.min_native_python_ratio, 0.75)

    def test_cli_values_override_config_individually(self) -> None:
        self.write_config(
            {"min_total_time_seconds": 2.0, "min_native_python_ratio": 0.75}
        )

        policy = self.resolve(min_total_time_seconds=3.0)

        self.assertEqual(policy.min_total_time_seconds, 3.0)
        self.assertEqual(policy.min_native_python_ratio, 0.75)

    def test_cli_can_supply_missing_config_value(self) -> None:
        self.write_config({"min_total_time_seconds": 2.0})

        policy = self.resolve(min_native_python_ratio=0.5)

        self.assertEqual(policy.min_total_time_seconds, 2.0)
        self.assertEqual(policy.min_native_python_ratio, 0.5)

    def test_cli_values_work_without_config(self) -> None:
        policy = self.resolve(
            min_total_time_seconds=1.0,
            min_native_python_ratio=1.25,
        )

        self.assertEqual(policy.min_total_time_seconds, 1.0)
        self.assertEqual(policy.min_native_python_ratio, 1.25)

    def test_requires_thresholds_when_config_is_missing(self) -> None:
        with self.assertRaisesRegex(
            DeepProfileConfigError, "thresholds are required"
        ):
            self.resolve()

    def test_explicit_missing_config_is_an_error(self) -> None:
        with self.assertRaisesRegex(DeepProfileConfigError, "Could not read config"):
            self.resolve(config_path=self.directory / "missing.json")

    def test_malformed_config_is_an_error(self) -> None:
        path = self.directory / DEFAULT_CONFIG_NAME
        path.write_text("{", encoding="utf-8")

        with self.assertRaisesRegex(DeepProfileConfigError, "not valid JSON"):
            self.resolve(min_total_time_seconds=1.0, min_native_python_ratio=1.0)

    def test_requires_configured_values_to_be_numbers(self) -> None:
        self.write_config(
            {"min_total_time_seconds": "1", "min_native_python_ratio": 1.0}
        )

        with self.assertRaisesRegex(
            DeepProfileConfigError, "min_total_time_seconds"
        ):
            self.resolve()

    def test_cli_fails_before_creating_or_running_notebook(self) -> None:
        stderr = io.StringIO()
        with (
            patch("profiling_rmsp_agent.__main__.create_instrumented_notebook") as create,
            patch("profiling_rmsp_agent.__main__.run_notebook") as run,
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as error,
        ):
            main(
                [
                    str(self.notebook_path),
                    "--min-total-time-seconds",
                    "1",
                ]
            )

        self.assertEqual(error.exception.code, 2)
        create.assert_not_called()
        run.assert_not_called()
        self.assertIn("thresholds are required", stderr.getvalue())


class ResolveRmspPathTests(unittest.TestCase):
    def test_missing_path_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            notebook = Path(directory) / "analysis.ipynb"
            self.assertIsNone(resolve_rmsp_path(notebook, None, None))

    def test_config_path_is_relative_to_config_and_cli_overrides_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "config"
            config_dir.mkdir()
            package_parent = config_dir / "libs"
            package_parent.mkdir()
            override = root / "override"
            override.mkdir()
            config = config_dir / "profile.json"
            config.write_text(json.dumps({"rmsp_path": "libs"}), encoding="utf-8")
            notebook = root / "analysis.ipynb"

            self.assertEqual(resolve_rmsp_path(notebook, config, None), package_parent)
            self.assertEqual(resolve_rmsp_path(notebook, config, override), override)

    def test_invalid_path_and_explicit_missing_config_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notebook = root / "analysis.ipynb"
            config = root / DEFAULT_CONFIG_NAME
            for value in ("", None, 42, "missing"):
                with self.subTest(value=value):
                    config.write_text(json.dumps({"rmsp_path": value}), encoding="utf-8")
                    with self.assertRaisesRegex(DeepProfileConfigError, "rmsp_path"):
                        resolve_rmsp_path(notebook, None, None)
            with self.assertRaisesRegex(DeepProfileConfigError, "Could not read config"):
                resolve_rmsp_path(notebook, root / "missing.json", root)

    def test_cli_passes_configured_path_to_first_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / DEFAULT_CONFIG_NAME
            config.write_text(json.dumps({
                "min_total_time_seconds": 1,
                "min_native_python_ratio": 1,
                "rmsp_path": ".",
            }), encoding="utf-8")
            instrumented = SimpleNamespace(output_path=root / "analysis_profile.ipynb")
            run_result = SimpleNamespace(cells=[], failed_cell=None, source_cell_indices={})
            with (
                patch("profiling_rmsp_agent.__main__.create_instrumented_notebook", return_value=instrumented),
                patch("profiling_rmsp_agent.__main__.run_notebook", return_value=run_result) as run,
                patch("profiling_rmsp_agent.__main__.write_report"),
            ):
                status = main([str(root / "analysis.ipynb")])

            self.assertEqual(status, 0)
            run.assert_called_once_with(instrumented.output_path, rmsp_path=root)

    def test_cli_override_is_passed_to_both_kernels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            override = root / "override"
            override.mkdir()
            config = root / DEFAULT_CONFIG_NAME
            config.write_text(json.dumps({
                "min_total_time_seconds": 1,
                "min_native_python_ratio": 1,
                "rmsp_path": ".",
            }), encoding="utf-8")
            instrumented = SimpleNamespace(
                output_path=root / "analysis_profile.ipynb", load_ext_cell_inserted=True
            )
            run_result = SimpleNamespace(
                cells=[{"cell_number": 2, "total_time_seconds": 2,
                        "native_time_seconds": 2, "python_time_seconds": 0}],
                failed_cell=None,
                source_cell_indices={2: 1},
            )
            deep_result = SimpleNamespace(status="completed", to_dict=lambda: {"status": "completed"})
            with (
                patch("profiling_rmsp_agent.__main__.create_instrumented_notebook", return_value=instrumented),
                patch("profiling_rmsp_agent.__main__.run_notebook", return_value=run_result) as run,
                patch("profiling_rmsp_agent.__main__.WindowsVtuneProfiler") as profiler,
                patch("profiling_rmsp_agent.__main__.run_candidate_replay", return_value=deep_result) as replay,
                patch("profiling_rmsp_agent.__main__.write_report"),
            ):
                status = main([
                    str(root / "analysis.ipynb"), "--rmsp-path", str(override),
                    "--capture-deep-profile",
                ])

            self.assertEqual(status, 0)
            run.assert_called_once_with(instrumented.output_path, rmsp_path=override)
            self.assertEqual(replay.call_args.kwargs["rmsp_path"], override)
            self.assertEqual(replay.call_args.args[2], {2: 0})
            profiler.return_value.preflight.assert_called_once()


if __name__ == "__main__":
    unittest.main()