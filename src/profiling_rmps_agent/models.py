"""Normalized data models shared by profiling backends and reports."""

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class CellProfile:
    """Measurements collected for one executed notebook cell."""

    cell_number: int
    duration_seconds: float
    backend: str
    python_time_seconds: float | None = None
    native_time_seconds: float | None = None
    memory_delta_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.cell_number < 1:
            raise ValueError("cell_number must be positive")
        if not self.backend.strip():
            raise ValueError("backend must not be empty")
        for name in (
            "duration_seconds",
            "python_time_seconds",
            "native_time_seconds",
        ):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def native_fraction(self) -> float | None:
        """Return the native share only when both time components are available."""
        if self.python_time_seconds is None or self.native_time_seconds is None:
            return None
        component_time = self.python_time_seconds + self.native_time_seconds
        if component_time == 0:
            return 0.0
        return self.native_time_seconds / component_time

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this profile."""
        return asdict(self)