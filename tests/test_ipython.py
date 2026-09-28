import unittest
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from profiling_rmps_agent.decision import DeepProfilePolicy
from profiling_rmps_agent.ipython import IPythonCellProfiler


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


class IPythonCellProfilerTests(unittest.TestCase):
    def test_records_cell_and_unregisters_on_detach(self) -> None:
        shell = FakeIPython()
        profiler = IPythonCellProfiler(
            policy=DeepProfilePolicy(min_duration_seconds=0)
        )

        profiler.attach(shell)
        shell.events.callbacks["pre_run_cell"][0](object())
        shell.events.callbacks["post_run_cell"][0](
            SimpleNamespace(execution_count=12)
        )

        profile = profiler.profiles[0]
        self.assertEqual(profile.cell_number, 12)
        self.assertEqual(profile.backend, "ipython-tracemalloc")
        self.assertIsNone(profile.native_time_seconds)
        self.assertGreaterEqual(profile.duration_seconds, 0)
        record = profiler.to_dicts()[0]
        self.assertTrue(record["deep_profile"]["should_profile"])
        self.assertEqual(
            record["deep_profile"]["reasons"], ["duration_threshold"]
        )
        self.assertEqual(len(profiler.deep_profile_candidates), 1)
        self.assertIs(shell.user_ns["notebook_profiler"], profiler)

        profiler.detach()

        self.assertEqual(shell.events.callbacks["pre_run_cell"], [])
        self.assertEqual(shell.events.callbacks["post_run_cell"], [])
        self.assertNotIn("notebook_profiler", shell.user_ns)


if __name__ == "__main__":
    unittest.main()