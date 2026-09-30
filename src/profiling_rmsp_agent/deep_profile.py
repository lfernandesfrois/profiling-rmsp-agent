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


def resolve_deep_profile_policy(
    notebook_path: Path,
    config_path: Path | None,
    cli_min_total_time_seconds: float | None,
    cli_min_native_python_ratio: float | None,
) -> DeepProfilePolicy:
    """Resolve thresholds from JSON config, then apply per-value CLI overrides."""
    selected_config_path = (
        config_path
        if config_path is not None
        else notebook_path.resolve().parent / DEFAULT_CONFIG_NAME
    )
    config_values: dict[str, Any] = {}

    if config_path is not None or selected_config_path.exists():
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
        config_values = loaded

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