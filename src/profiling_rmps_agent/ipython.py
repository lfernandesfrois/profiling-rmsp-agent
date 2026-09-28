"""IPython extension for collecting lightweight per-cell measurements."""

import tracemalloc
from time import perf_counter
from typing import Any

from profiling_rmps_agent.decision import DeepProfileDecision, DeepProfilePolicy
from profiling_rmps_agent.models import CellProfile


class IPythonCellProfiler:
    """Record wall time and traced Python memory for executed cells."""

    def __init__(self, policy: DeepProfilePolicy | None = None) -> None:
        self._ipython: Any | None = None
        self._owns_tracemalloc = False
        self._active_cell: tuple[float, int] | None = None
        self._fallback_cell_number = 0
        self._profiles: list[CellProfile] = []
        self._decisions: list[DeepProfileDecision] = []
        self._policy = policy if policy is not None else DeepProfilePolicy()

    @property
    def profiles(self) -> tuple[CellProfile, ...]:
        return tuple(self._profiles)

    @property
    def deep_profile_candidates(self) -> tuple[tuple[CellProfile, DeepProfileDecision], ...]:
        return tuple(
            (profile, decision)
            for profile, decision in zip(self._profiles, self._decisions)
            if decision.should_profile
        )

    def to_dicts(self) -> list[dict[str, Any]]:
        records = []
        for profile, decision in zip(self._profiles, self._decisions):
            record = profile.to_dict()
            record["deep_profile"] = {
                "should_profile": decision.should_profile,
                "reasons": list(decision.reasons),
            }
            records.append(record)
        return records

    def attach(self, ipython: Any) -> None:
        if self._ipython is not None:
            raise RuntimeError("profiler is already attached")
        self._owns_tracemalloc = not tracemalloc.is_tracing()
        if self._owns_tracemalloc:
            tracemalloc.start()
        self._ipython = ipython
        ipython.events.register("pre_run_cell", self._before_cell)
        ipython.events.register("post_run_cell", self._after_cell)
        ipython.user_ns["notebook_profiler"] = self

    def detach(self) -> None:
        if self._ipython is None:
            return
        self._ipython.events.unregister("pre_run_cell", self._before_cell)
        self._ipython.events.unregister("post_run_cell", self._after_cell)
        if self._ipython.user_ns.get("notebook_profiler") is self:
            del self._ipython.user_ns["notebook_profiler"]
        self._ipython = None
        self._active_cell = None
        if self._owns_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()
        self._owns_tracemalloc = False

    def _before_cell(self, _info: object) -> None:
        current_memory, _ = tracemalloc.get_traced_memory()
        self._active_cell = (perf_counter(), current_memory)

    def _after_cell(self, result: object) -> None:
        if self._active_cell is None:
            return
        started_at, starting_memory = self._active_cell
        self._active_cell = None
        current_memory, _ = tracemalloc.get_traced_memory()
        execution_count = getattr(result, "execution_count", None)
        if not isinstance(execution_count, int) or execution_count < 1:
            self._fallback_cell_number += 1
            execution_count = self._fallback_cell_number
        profile = CellProfile(
            cell_number=execution_count,
            duration_seconds=perf_counter() - started_at,
            backend="ipython-tracemalloc",
            memory_delta_bytes=current_memory - starting_memory,
        )
        self._profiles.append(profile)
        self._decisions.append(self._policy.evaluate(profile))


_profiler: IPythonCellProfiler | None = None


def load_ipython_extension(ipython: Any) -> None:
    global _profiler
    if _profiler is not None:
        _profiler.detach()
    _profiler = IPythonCellProfiler()
    _profiler.attach(ipython)


def unload_ipython_extension(_ipython: Any) -> None:
    global _profiler
    if _profiler is not None:
        _profiler.detach()
        _profiler = None