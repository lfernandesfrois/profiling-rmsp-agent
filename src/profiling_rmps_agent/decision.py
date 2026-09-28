"""Deterministic rules for escalating a cell to deep profiling."""

from dataclasses import dataclass

from profiling_rmps_agent.models import CellProfile


@dataclass(frozen=True, slots=True)
class DeepProfileDecision:
    should_profile: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DeepProfilePolicy:
    min_duration_seconds: float = 2.0
    min_memory_delta_bytes: int = 200 * 1024 * 1024
    min_native_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.min_duration_seconds < 0:
            raise ValueError("min_duration_seconds must be non-negative")
        if self.min_memory_delta_bytes < 0:
            raise ValueError("min_memory_delta_bytes must be non-negative")
        if not 0 <= self.min_native_fraction <= 1:
            raise ValueError("min_native_fraction must be between 0 and 1")

    def evaluate(self, profile: CellProfile) -> DeepProfileDecision:
        reasons: list[str] = []
        if profile.duration_seconds >= self.min_duration_seconds:
            reasons.append("duration_threshold")
        if (
            profile.memory_delta_bytes is not None
            and profile.memory_delta_bytes >= self.min_memory_delta_bytes
        ):
            reasons.append("memory_threshold")
        native_fraction = profile.native_fraction
        if (
            native_fraction is not None
            and native_fraction >= self.min_native_fraction
        ):
            reasons.append("native_time_threshold")
        return DeepProfileDecision(
            should_profile=bool(reasons),
            reasons=tuple(reasons),
        )