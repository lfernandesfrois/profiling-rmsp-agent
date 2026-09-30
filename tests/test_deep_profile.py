import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from profiling_rmsp_agent.__main__ import main
from profiling_rmsp_agent.deep_profile import (
    DEFAULT_CONFIG_NAME,
    DeepProfileConfigError,
    DeepProfilePolicy,
    resolve_deep_profile_policy,
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


if __name__ == "__main__":
    unittest.main()