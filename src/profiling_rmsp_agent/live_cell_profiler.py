"""Live IPython extension measuring per-cell Python vs native execution time."""

from dataclasses import dataclass
from time import perf_counter
import sys
from typing import Any


@dataclass(frozen=True, slots=True)
class CellTiming:
    """Timing measured for one executed notebook cell."""

    cell_number: int
    total_time_seconds: float
    python_time_seconds: float
    native_time_seconds: float
    python_percent: float | None
    native_percent: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_number": self.cell_number,
            "total_time_seconds": self.total_time_seconds,
            "python_time_seconds": self.python_time_seconds,
            "native_time_seconds": self.native_time_seconds,
            "python_percent": self.python_percent,
            "native_percent": self.native_percent,
        }


class _CallTimeSplitter:
    """Split exclusive time between Python-level and native/C-level calls.

    Installed via sys.setprofile. Each event's elapsed time since the last
    event is attributed to whichever call kind is currently innermost, then
    the internal stack is updated for call/return and
    c_call/c_return/c_exception events.
    """

    def __init__(self) -> None:
        self._stack: list[str] = []
        self._last_time = perf_counter()
        self.python_time_seconds = 0.0
        self.native_time_seconds = 0.0

    def __call__(self, frame: Any, event: str, arg: Any) -> None:
        now = perf_counter()
        if self._stack:
            elapsed = now - self._last_time
            if self._stack[-1] == "python":
                self.python_time_seconds += elapsed
            else:
                self.native_time_seconds += elapsed
        self._last_time = now

        if event == "call":
            self._stack.append("python")
        elif event == "return":
            if self._stack:
                self._stack.pop()
        elif event == "c_call":
            self._stack.append("native")
        elif event in ("c_return", "c_exception"):
            if self._stack:
                self._stack.pop()


class LiveCellProfiler:
    """Measure wall time and the Python/native split for each executed cell."""

    def __init__(self) -> None:
        self._ipython: Any | None = None
        self._active_cell: tuple[float, _CallTimeSplitter] | None = None
        self._fallback_cell_number = 0
        self._timings: list[CellTiming] = []

    @property
    def timings(self) -> tuple[CellTiming, ...]:
        return tuple(self._timings)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [timing.to_dict() for timing in self._timings]

    def attach(self, ipython: Any) -> None:
        if self._ipython is not None:
            raise RuntimeError("profiler is already attached")
        self._ipython = ipython
        ipython.events.register("pre_run_cell", self._before_cell)
        ipython.events.register("post_run_cell", self._after_cell)
        ipython.user_ns["live_cell_profiler"] = self

    def detach(self) -> None:
        if self._ipython is None:
            return
        self._ipython.events.unregister("pre_run_cell", self._before_cell)
        self._ipython.events.unregister("post_run_cell", self._after_cell)
        if self._ipython.user_ns.get("live_cell_profiler") is self:
            del self._ipython.user_ns["live_cell_profiler"]
        self._ipython = None
        if self._active_cell is not None:
            sys.setprofile(None)
            self._active_cell = None

    def _before_cell(self, _info: object) -> None:
        splitter = _CallTimeSplitter()
        sys.setprofile(splitter)
        self._active_cell = (perf_counter(), splitter)

    def _after_cell(self, result: object) -> None:
        if self._active_cell is None:
            return
        started_at, splitter = self._active_cell
        sys.setprofile(None)
        self._active_cell = None

        total_time_seconds = perf_counter() - started_at
        python_time = splitter.python_time_seconds
        native_time = splitter.native_time_seconds
        sampled_total = python_time + native_time
        if sampled_total > 0:
            python_percent = python_time / sampled_total * 100
            native_percent = native_time / sampled_total * 100
        else:
            python_percent = None
            native_percent = None

        execution_count = getattr(result, "execution_count", None)
        if not isinstance(execution_count, int) or execution_count < 1:
            self._fallback_cell_number += 1
            execution_count = self._fallback_cell_number

        self._timings.append(
            CellTiming(
                cell_number=execution_count,
                total_time_seconds=total_time_seconds,
                python_time_seconds=python_time,
                native_time_seconds=native_time,
                python_percent=python_percent,
                native_percent=native_percent,
            )
        )


_profiler: LiveCellProfiler | None = None


def load_ipython_extension(ipython: Any) -> None:
    global _profiler
    if _profiler is not None:
        _profiler.detach()
    _profiler = LiveCellProfiler()
    _profiler.attach(ipython)


def unload_ipython_extension(_ipython: Any) -> None:
    global _profiler
    if _profiler is not None:
        _profiler.detach()
        _profiler = None
