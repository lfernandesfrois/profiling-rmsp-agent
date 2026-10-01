"""Configuration and decision policy for deep-profile candidates."""

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_NAME = "profile_config.json"


class DeepProfileConfigError(ValueError):
    """Raised when deep-profile thresholds cannot be resolved safely."""


@dataclass(frozen=True, slots=True)
class DeepProfilePolicy:
    min_total_time_seconds: float
    min_native_python_ratio: float

    def __post_init__(self) -> None:
        _validate_threshold(
            "min_total_time_seconds", self.min_total_time_seconds
        )
        _validate_threshold(
            "min_native_python_ratio", self.min_native_python_ratio
        )

    def is_candidate(self, cell: Mapping[str, Any]) -> bool:
        total_time = _cell_time(cell, "total_time_seconds")
        native_time = _cell_time(cell, "native_time_seconds")
        python_time = _cell_time(cell, "python_time_seconds")
        if total_time is None or native_time is None or python_time is None:
            return False

        return (
            total_time > self.min_total_time_seconds
            and native_time > self.min_native_python_ratio * python_time
        )


def _validate_threshold(name: str, value: Any) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise DeepProfileConfigError(
            f"{name} must be a finite, non-negative number."
        )


def _cell_time(cell: Mapping[str, Any], name: str) -> float | None:
    value = cell.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        return None
    return float(value)


def _load_config(notebook_path: Path, config_path: Path | None) -> tuple[Path, dict[str, Any]]:
    selected_config_path = (
        config_path
        if config_path is not None
        else notebook_path.resolve().parent / DEFAULT_CONFIG_NAME
    )
    if config_path is None and not selected_config_path.exists():
        return selected_config_path, {}

    try:
        loaded = json.loads(selected_config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise DeepProfileConfigError(
            f"Could not read config file {selected_config_path}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise DeepProfileConfigError(
            f"Config file {selected_config_path} is not valid JSON: {error}"
        ) from error

    if not isinstance(loaded, dict):
        raise DeepProfileConfigError(
            f"Config file {selected_config_path} must contain a JSON object."
        )
    return selected_config_path, loaded


def resolve_rmsp_path(
    notebook_path: Path, config_path: Path | None, cli_rmsp_path: Path | None
) -> Path | None:
    """Resolve the parent directory of the rmsp package for kernel imports."""
    selected_config_path, config_values = _load_config(notebook_path, config_path)
    configured_path = config_values.get("rmsp_path")
    if cli_rmsp_path is None and "rmsp_path" not in config_values:
        return None
    if cli_rmsp_path is None and (
        not isinstance(configured_path, str) or not configured_path.strip()
    ):
        raise DeepProfileConfigError("rmsp_path must be a non-empty directory path.")

    directory = cli_rmsp_path if cli_rmsp_path is not None else Path(configured_path)
    directory = directory.expanduser()
    if not directory.is_absolute() and cli_rmsp_path is None:
        directory = selected_config_path.resolve().parent / directory
    directory = directory.resolve()
    if not directory.is_dir():
        raise DeepProfileConfigError(f"rmsp_path must be an existing directory: {directory}")
    return directory


def resolve_deep_profile_policy(
    notebook_path: Path,
    config_path: Path | None,
    cli_min_total_time_seconds: float | None,
    cli_min_native_python_ratio: float | None,
) -> DeepProfilePolicy:
    """Resolve thresholds from JSON config, then apply per-value CLI overrides."""
    _, config_values = _load_config(notebook_path, config_path)

    total_time = (
        cli_min_total_time_seconds
        if cli_min_total_time_seconds is not None
        else config_values.get("min_total_time_seconds")
    )
    native_python_ratio = (
        cli_min_native_python_ratio
        if cli_min_native_python_ratio is not None
        else config_values.get("min_native_python_ratio")
    )

    missing = []
    if total_time is None:
        missing.append("--min-total-time-seconds / min_total_time_seconds")
    if native_python_ratio is None:
        missing.append("--min-native-python-ratio / min_native_python_ratio")
    if missing:
        options = " and ".join(missing)
        raise DeepProfileConfigError(
            "Deep-profile thresholds are required; provide "
            f"{options} in the config file or command line."
        )

    return DeepProfilePolicy(
        min_total_time_seconds=total_time,
        min_native_python_ratio=native_python_ratio,
    )