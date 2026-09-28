import unittest

from profiling_rmps_agent import CellProfile, DeepProfilePolicy


class DeepProfilePolicyTests(unittest.TestCase):
    def test_escalates_slow_cell(self) -> None:
        profile = CellProfile(cell_number=1, duration_seconds=3.0, backend="test")

        decision = DeepProfilePolicy().evaluate(profile)

        self.assertTrue(decision.should_profile)
        self.assertEqual(decision.reasons, ("duration_threshold",))

    def test_escalates_native_heavy_cell(self) -> None:
        profile = CellProfile(
            cell_number=2,
            duration_seconds=1.0,
            backend="test",
            python_time_seconds=1.0,
            native_time_seconds=2.0,
        )

        decision = DeepProfilePolicy().evaluate(profile)

        self.assertTrue(decision.should_profile)
        self.assertEqual(decision.reasons, ("native_time_threshold",))

    def test_does_not_infer_native_share_when_unavailable(self) -> None:
        profile = CellProfile(
            cell_number=3,
            duration_seconds=1.0,
            backend="test",
            native_time_seconds=1.0,
        )

        decision = DeepProfilePolicy().evaluate(profile)

        self.assertFalse(decision.should_profile)
        self.assertEqual(decision.reasons, ())


if __name__ == "__main__":
    unittest.main()