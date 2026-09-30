import csv
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from profiling_rmsp_agent.etw_capture import (
    EtwCaptureError,
    EtwConfig,
    WindowsEtwProfiler,
    resolve_etw_config,
)
from profiling_rmsp_agent.wpa_export import summarize_kernel_cpu_csvs


class FakeProcessRunner:
    def __init__(self, fail_at: int | None = None) -> None:
        self.commands: list[list[str]] = []
        self.fail_at = fail_at

    def __call__(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if self.fail_at == len(self.commands):
            raise OSError("simulated process failure")
        if "-stop" in command:
            Path(command[command.index("-stop") + 1]).touch()
        stdout = "CPUProfile" if "-profiles" in command else "There are no trace profiles running."
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


class EtwProfilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.wpr_profile = self.directory / "capture.wprp"
        self.wpa_profile = self.directory / "cpu.wpaProfile"
        self.wpr_profile.write_text("<WPRProfile />", encoding="utf-8")
        self.wpa_profile.write_text("<WpaProfileContainer />", encoding="utf-8")
        self.config = EtwConfig(
            wpr_executable="wpr.exe",
            wpr_profile=f"{self.wpr_profile}!CPUProfile.Verbose",
            wpaexporter_executable="wpaexporter.exe",
            wpa_export_profile=self.wpa_profile,
            artifact_directory=self.directory / "artifacts",
            cpu_csv_pattern="*sampled.csv",
            process_id_column="Process ID",
            thread_id_column="Thread ID",
            cpu_time_column="Weight (ms)",
            cpu_time_unit="ms",
            timeout_seconds=10,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def make_profiler(
        self, runner: FakeProcessRunner, platform_name: str = "win32"
    ) -> WindowsEtwProfiler:
        return WindowsEtwProfiler(
            self.config,
            runner=runner,
            executable_lookup=lambda name: f"C:/WPT/{name}",
            platform_name=platform_name,
        )

    def test_wpr_profiles_are_checked_then_unique_capture_starts_and_stops(self) -> None:
        runner = FakeProcessRunner()
        profiler = self.make_profiler(runner)
        profiler.preflight()

        capture = profiler.start_capture(
            cell_number=3, notebook_cell_index=4, kernel_pid=123
        )
        profiler.stop_capture(capture)

        self.assertEqual(
            [command[1] for command in runner.commands],
            ["-profiles", "-status", "-start", "-stop"],
        )
        start, stop = runner.commands[2:]
        self.assertEqual(start[1:4], ["-start", self.config.wpr_profile, "-filemode"])
        self.assertEqual(start[-2:], ["-instancename", capture.instance_name])
        self.assertEqual(stop[-2:], ["-instancename", capture.instance_name])
        self.assertEqual(capture.status, "captured")
        self.assertTrue(capture.etl_path.is_file())

    def test_does_not_stop_a_capture_when_our_start_failed(self) -> None:
        runner = FakeProcessRunner(fail_at=3)
        profiler = self.make_profiler(runner)
        profiler.preflight()

        with self.assertRaisesRegex(EtwCaptureError, "Could not start ETW command"):
            profiler.start_capture(cell_number=1, notebook_cell_index=0, kernel_pid=7)

        self.assertEqual(
            [command[1] for command in runner.commands],
            ["-profiles", "-status", "-start", "-cancel"],
        )
        self.assertEqual(
            runner.commands[-1][1:],
            ["-cancel", "-instancename", runner.commands[-2][-1]],
        )

    def test_stop_failure_is_reported_and_capture_is_not_reused(self) -> None:
        runner = FakeProcessRunner(fail_at=4)
        profiler = self.make_profiler(runner)
        profiler.preflight()
        capture = profiler.start_capture(1, 0, 9)

        with self.assertRaisesRegex(EtwCaptureError, "Could not start ETW command"):
            profiler.stop_capture(capture)

        self.assertEqual(capture.status, "stop_failed")
        self.assertEqual(runner.commands[-1][1:], ["-cancel", "-instancename", capture.instance_name])
        with self.assertRaisesRegex(EtwCaptureError, "not owned by this run"):
            profiler.stop_capture(capture)

    def test_refuses_preexisting_recording_without_stopping_it(self) -> None:
        class RecordingRunner(FakeProcessRunner):
            def __call__(
                self, command: list[str], **kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                self.commands.append(command)
                stdout = (
                    "CPUProfile"
                    if "-profiles" in command
                    else "WPR recording is in progress"
                )
                return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

        runner = RecordingRunner()
        profiler = self.make_profiler(runner)

        with self.assertRaisesRegex(EtwCaptureError, "already recording"):
            profiler.preflight()

        self.assertEqual(
            [command[1] for command in runner.commands], ["-profiles", "-status"]
        )

    def test_exporter_generates_csv_and_csv_summary_filters_pid_and_thread(self) -> None:
        runner = FakeProcessRunner()
        profiler = self.make_profiler(runner)
        profiler.preflight()
        capture = profiler.start_capture(2, 4, 17)
        profiler.stop_capture(capture)

        def exporter_runner(
            command: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            runner.commands.append(command)
            output_dir = Path(command[command.index("-outputfolder") + 1])
            with (output_dir / "cpu-sampled.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["Process ID", "Thread ID", "Weight (ms)"],
                )
                writer.writeheader()
                writer.writerow(
                    {"Process ID": "17", "Thread ID": "3", "Weight (ms)": "2.5"}
                )
                writer.writerow(
                    {"Process ID": "17", "Thread ID": "4", "Weight (ms)": "1.5"}
                )
                writer.writerow(
                    {"Process ID": "99", "Thread ID": "6", "Weight (ms)": "100"}
                )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        profiler._runner = exporter_runner
        csv_directory = profiler.run_directory / "cell-2-csv"
        csv_paths = profiler.export_capture(capture.etl_path, csv_directory)
        summary = summarize_kernel_cpu_csvs(csv_paths, profiler.config, kernel_pid=17)

        self.assertIn("-profile", runner.commands[-1])
        self.assertEqual(summary.sampled_cpu_time_seconds, 0.004)
        self.assertEqual(summary.thread_rows, 2)
        self.assertEqual(summary.distinct_threads, 2)
        self.assertEqual(len(csv_paths), 1)

    def test_non_windows_preflight_is_rejected(self) -> None:
        profiler = self.make_profiler(FakeProcessRunner(), platform_name="linux")

        with self.assertRaisesRegex(EtwCaptureError, "only on Windows"):
            profiler.preflight()


class EtwConfigTests(unittest.TestCase):
    def test_config_resolves_paths_and_csv_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wpr_profile = root / "custom.wprp"
            wpa_profile = root / "cpu.wpaProfile"
            wpr_profile.write_text("profile", encoding="utf-8")
            wpa_profile.write_text("layout", encoding="utf-8")
            config_path = root / "profile_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "etw": {
                            "wpr_executable": "tools/wpr.exe",
                            "wpr_profile": "custom.wprp!CPUProfile",
                            "wpa_export_profile": "cpu.wpaProfile",
                            "artifact_directory": "artifacts",
                            "cpu_csv_pattern": "*cpu.csv",
                            "process_id_column": "PID",
                            "thread_id_column": "TID",
                            "cpu_time_column": "CPU (ms)",
                            "cpu_time_unit": "ms",
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = resolve_etw_config(root / "analysis.ipynb", config_path)

            self.assertEqual(config.wpr_profile, f"{wpr_profile}!CPUProfile")
            self.assertEqual(config.wpa_export_profile, wpa_profile)
            self.assertEqual(config.artifact_directory, root / "artifacts")
            self.assertEqual(config.wpr_executable, str((root / "tools/wpr.exe").resolve()))

    def test_requires_existing_wpr_profile_before_any_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "profile_config.json"
            wpa_profile = root / "cpu.wpaProfile"
            wpa_profile.write_text("layout", encoding="utf-8")
            config_path.write_text(
                json.dumps(
                    {"etw": {"wpa_export_profile": "cpu.wpaProfile"}}
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(EtwCaptureError, "wpr_profile is required"):
                resolve_etw_config(root / "analysis.ipynb", config_path)


if __name__ == "__main__":
    unittest.main()