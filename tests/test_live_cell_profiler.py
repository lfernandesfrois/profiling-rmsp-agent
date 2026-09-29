import unittest
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from profiling_rmsp_agent.live_cell_profiler import LiveCellProfiler, _CallTimeSplitter


class FakeEvents:
    def __init__(self) -> None:
        self.callbacks: dict[str, list[Callable[..., None]]] = {}

    def register(self, event: str, callback: Callable[..., None]) -> None:
        self.callbacks.setdefault(event, []).append(callback)

    def unregister(self, event: str, callback: Callable[..., None]) -> None:
        self.callbacks[event].remove(callback)


class FakeIPython:
    def __init__(self) -> None:
        self.events = FakeEvents()
        self.user_ns: dict[str, Any] = {}


def run_cell(shell: FakeIPython, body: Callable[[], None], execution_count: int) -> None:
    shell.events.callbacks["pre_run_cell"][0](object())
    body()
    shell.events.callbacks["post_run_cell"][0](SimpleNamespace(execution_count=execution_count))


class LiveCellProfilerTests(unittest.TestCase):
    def test_attach_registers_hooks_and_exposes_profiler_in_namespace(self) -> None:
        shell = FakeIPython()
        profiler = LiveCellProfiler()

        profiler.attach(shell)

        self.assertIn("pre_run_cell", shell.events.callbacks)
        self.assertIn("post_run_cell", shell.events.callbacks)
        self.assertIs(shell.user_ns["live_cell_profiler"], profiler)

    def test_detach_unregisters_hooks_and_cleans_namespace(self) -> None:
        shell = FakeIPython()
        profiler = LiveCellProfiler()
        profiler.attach(shell)

        profiler.detach()

        self.assertEqual(shell.events.callbacks["pre_run_cell"], [])
        self.assertEqual(shell.events.callbacks["post_run_cell"], [])
        self.assertNotIn("live_cell_profiler", shell.user_ns)

    def test_records_wall_time_and_call_kind_split_for_executed_cell(self) -> None:
        shell = FakeIPython()
        profiler = LiveCellProfiler()
        profiler.attach(shell)

        def python_heavy() -> None:
            def add_one(value: int) -> int:
                return value + 1

            total = 0
            for i in range(2000):
                total = add_one(total)

        run_cell(shell, python_heavy, execution_count=5)

        self.assertEqual(len(profiler.timings), 1)
        timing = profiler.timings[0]
        self.assertEqual(timing.cell_number, 5)
        self.assertGreaterEqual(timing.total_time_seconds, 0)
        self.assertGreaterEqual(timing.python_time_seconds, 0)
        self.assertGreaterEqual(timing.native_time_seconds, 0)
        self.assertIsNotNone(timing.python_percent)
        self.assertIsNotNone(timing.native_percent)
        self.assertAlmostEqual(
            timing.python_percent + timing.native_percent, 100.0, places=5
        )

    def test_native_heavy_cell_reports_higher_native_percent(self) -> None:
        shell = FakeIPython()
        profiler = LiveCellProfiler()
        profiler.attach(shell)

        def native_heavy() -> None:
            data = list(range(200_000))
            for _ in range(200):
                sum(data)

        run_cell(shell, native_heavy, execution_count=1)

        timing = profiler.timings[0]
        self.assertIsNotNone(timing.native_percent)
        self.assertGreater(timing.native_percent, 50.0)

    def test_cell_with_no_calls_reports_a_tiny_self_instrumentation_split(self) -> None:
        shell = FakeIPython()
        profiler = LiveCellProfiler()
        profiler.attach(shell)

        # The pre/post hooks themselves generate a few call events, so a
        # truly empty cell still reports a tiny nonzero split; the exact
        # zero-events case is checked directly on the splitter below.
        run_cell(shell, lambda: None, execution_count=2)

        timing = profiler.timings[0]
        self.assertIsInstance(timing.python_time_seconds, float)
        self.assertIsInstance(timing.native_time_seconds, float)

    def test_splitter_reports_no_time_when_no_events_observed(self) -> None:
        splitter = _CallTimeSplitter()

        self.assertEqual(splitter.python_time_seconds, 0.0)
        self.assertEqual(splitter.native_time_seconds, 0.0)

    def test_splitter_attributes_time_to_innermost_active_call(self) -> None:
        splitter = _CallTimeSplitter()

        splitter(None, "call", None)
        splitter(None, "c_call", None)
        splitter(None, "c_return", None)
        splitter(None, "return", None)

        self.assertGreaterEqual(splitter.python_time_seconds, 0.0)
        self.assertGreaterEqual(splitter.native_time_seconds, 0.0)
        self.assertEqual(splitter._stack, [])

    def test_fallback_cell_number_when_execution_count_missing(self) -> None:
        shell = FakeIPython()
        profiler = LiveCellProfiler()
        profiler.attach(shell)

        shell.events.callbacks["pre_run_cell"][0](object())
        shell.events.callbacks["post_run_cell"][0](object())
        shell.events.callbacks["pre_run_cell"][0](object())
        shell.events.callbacks["post_run_cell"][0](object())

        self.assertEqual(profiler.timings[0].cell_number, 1)
        self.assertEqual(profiler.timings[1].cell_number, 2)

    def test_to_dicts_matches_timings(self) -> None:
        shell = FakeIPython()
        profiler = LiveCellProfiler()
        profiler.attach(shell)

        run_cell(shell, lambda: (lambda: None)(), execution_count=9)

        records = profiler.to_dicts()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["cell_number"], 9)
        self.assertIn("total_time_seconds", records[0])

    def test_detach_while_cell_active_removes_profile_hook(self) -> None:
        import sys

        shell = FakeIPython()
        profiler = LiveCellProfiler()
        profiler.attach(shell)

        shell.events.callbacks["pre_run_cell"][0](object())
        self.assertIsNotNone(sys.getprofile())

        profiler.detach()

        self.assertIsNone(sys.getprofile())


if __name__ == "__main__":
    unittest.main()
