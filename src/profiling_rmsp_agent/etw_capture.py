"""Windows Performance Recorder and WPA Exporter integration."""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Sequence
import uuid


DEFAULT_CONFIG_NAME = "profile_config.json"
_TIME_UNIT_TO_SECONDS = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "100ns": 1e-7,
}


class EtwCaptureError(RuntimeError):
    """Raised when WPR/WPA capture or export cannot safely proceed."""


@dataclass(frozen=True, slots=True)
class EtwConfig:
    wpr_executable: str
    wpr_profile: str
    wpaexporter_executable: str
    wpa_export_profile: Path
    artifact_directory: Path
    cpu_csv_pattern: str
    process_id_column: str
    thread_id_column: str
    cpu_time_column: str
    cpu_time_unit: str
    csv_delimiter: str = ","
    timeout_seconds: float = 120.0


@dataclass(slots=True)
class EtlCapture:
    cell_number: int
    notebook_cell_index: int
    kernel_pid: int
    etl_path: Path
    instance_name: str
    status: str = "pending"
    error: str | None = None
    status: str = "pending"
    error: str | None = None


ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]


def _read_config(config_path: Path) -> dict[str, object]:
    try:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise EtwCaptureError(f"Could not read config file {config_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise EtwCaptureError(
            f"Config file {config_path} is not valid JSON: {error}"
        ) from error
    if not isinstance(loaded, dict):
        raise EtwCaptureError(f"Config file {config_path} must contain a JSON object.")
    return loaded


def resolve_etw_config(notebook_path: Path, config_path: Path | None) -> EtwConfig:
    """Load and validate ETW settings from the notebook's profiling config."""
    selected_config = config_path or notebook_path.resolve().parent / DEFAULT_CONFIG_NAME
    document = _read_config(selected_config)
    etw_values = document.get("etw")
    if not isinstance(etw_values, dict):
        raise EtwCaptureError(
            f"Config {selected_config} must contain an 'etw' object to enable deep capture."
        )

    wpa_profile_value = etw_values.get("wpa_export_profile")
    if not isinstance(wpa_profile_value, str) or not wpa_profile_value.strip():
        raise EtwCaptureError(
            "etw.wpa_export_profile must point to a WPA .wpaProfile file. "
            "Create/export a CPU Usage (Sampled) by Process, Thread profile in WPA first."
        )
    wpa_export_profile = Path(wpa_profile_value).expanduser()
    if not wpa_export_profile.is_absolute():
        wpa_export_profile = selected_config.resolve().parent / wpa_export_profile
    if wpa_export_profile.suffix.lower() != ".wpaprofile":
        raise EtwCaptureError("etw.wpa_export_profile must have a .wpaProfile extension.")
    if not wpa_export_profile.is_file():
        raise EtwCaptureError(f"WPA export profile not found: {wpa_export_profile}")

    wpr_profile = etw_values.get("wpr_profile")
    if not isinstance(wpr_profile, str) or not wpr_profile.strip():
        raise EtwCaptureError(
            "etw.wpr_profile is required and must reference an existing .wprp profile."
        )
    if "!" not in wpr_profile:
        raise EtwCaptureError(
            "etw.wpr_profile must include a .wprp path and profile name, for example "
            "'profiles/cpu.wprp!CpuProfile.Verbose'."
        )
    wprp_path_text, profile_name = wpr_profile.rsplit("!", 1)
    wprp_path = Path(os.path.expandvars(wprp_path_text)).expanduser()
    if not wprp_path.is_absolute():
        wprp_path = selected_config.resolve().parent / wprp_path
    if wprp_path.suffix.lower() != ".wprp" or not wprp_path.is_file():
        raise EtwCaptureError(f"WPR profile file not found or invalid: {wprp_path}")
    if not profile_name.strip():
        raise EtwCaptureError("etw.wpr_profile must name a profile after the '!'.")
    wpr_profile = f"{wprp_path.resolve()}!{profile_name}"

    required_columns = {
        "cpu_csv_pattern": etw_values.get("cpu_csv_pattern"),
        "process_id_column": etw_values.get("process_id_column"),
        "thread_id_column": etw_values.get("thread_id_column"),
        "cpu_time_column": etw_values.get("cpu_time_column"),
    }
    for key, value in required_columns.items():
        if not isinstance(value, str) or not value.strip():
            raise EtwCaptureError(f"etw.{key} must be a non-empty string.")

    cpu_time_unit = etw_values.get("cpu_time_unit", "ms")
    if not isinstance(cpu_time_unit, str) or cpu_time_unit.lower() not in _TIME_UNIT_TO_SECONDS:
        valid_units = ", ".join(_TIME_UNIT_TO_SECONDS)
        raise EtwCaptureError(f"etw.cpu_time_unit must be one of: {valid_units}.")

    csv_delimiter = etw_values.get("csv_delimiter", ",")
    if not isinstance(csv_delimiter, str) or len(csv_delimiter) != 1:
        raise EtwCaptureError("etw.csv_delimiter must be exactly one character.")

    timeout_seconds = etw_values.get("timeout_seconds", 120.0)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise EtwCaptureError("etw.timeout_seconds must be a finite positive number.")

    artifact_directory_value = etw_values.get("artifact_directory")
    if artifact_directory_value is None:
        artifact_directory = notebook_path.resolve().parent / (
            f"{notebook_path.stem}_deep_profile_artifacts"
        )
    elif isinstance(artifact_directory_value, str) and artifact_directory_value.strip():
        artifact_directory = Path(artifact_directory_value).expanduser()
        if not artifact_directory.is_absolute():
            artifact_directory = selected_config.resolve().parent / artifact_directory
    else:
        raise EtwCaptureError("etw.artifact_directory must be a non-empty path.")

    return EtwConfig(
        wpr_executable=_configured_executable(
            etw_values.get("wpr_executable"),
            "wpr.exe",
            selected_config.resolve().parent,
        ),
        wpr_profile=wpr_profile,
        wpaexporter_executable=_configured_executable(
            etw_values.get("wpaexporter_executable"),
            "wpaexporter.exe",
            selected_config.resolve().parent,
        ),
        wpa_export_profile=wpa_export_profile.resolve(),
        artifact_directory=artifact_directory.resolve(),
        cpu_csv_pattern=required_columns["cpu_csv_pattern"],
        process_id_column=required_columns["process_id_column"],
        thread_id_column=required_columns["thread_id_column"],
        cpu_time_column=required_columns["cpu_time_column"],
        cpu_time_unit=cpu_time_unit.lower(),
        csv_delimiter=csv_delimiter,
        timeout_seconds=float(timeout_seconds),
    )


def _configured_executable(value: object, default: str, base_directory: Path) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise EtwCaptureError("Configured ETW executable paths must be non-empty strings.")
    expanded = Path(os.path.expandvars(value)).expanduser()
    if not expanded.is_absolute() and expanded.parent != Path("."):
        return str((base_directory / expanded).resolve())
    return str(expanded)


class WindowsEtwProfiler:
    """Own sequential WPR captures and WPA CSV exports for one profiling run."""

    def __init__(
        self,
        config: EtwConfig,
        runner: ProcessRunner = subprocess.run,
        executable_lookup: Callable[[str], str | None] = shutil.which,
        platform_name: str = sys.platform,
    ) -> None:
        self.config = config
        self._runner = runner
        self._executable_lookup = executable_lookup
        self._platform_name = platform_name
        self._wpr_path: str | None = None
        self._wpaexporter_path: str | None = None
        self._active_capture: EtlCapture | None = None
        self.run_directory: Path | None = None

    def preflight(self) -> Path:
        if self._platform_name != "win32":
            raise EtwCaptureError("WPR/WPA deep capture is supported only on Windows.")

        self._wpr_path = self._resolve_executable(self.config.wpr_executable)
        self._wpaexporter_path = self._resolve_executable(
            self.config.wpaexporter_executable
        )
        self._validate_wpr_profile()
        self._ensure_wpr_idle()

        try:
            self.config.artifact_directory.mkdir(parents=True, exist_ok=True)
            self.run_directory = Path(
                tempfile.mkdtemp(
                    prefix=f"run-{uuid.uuid4().hex[:8]}-",
                    dir=self.config.artifact_directory,
                )
            )
        except OSError as error:
            raise EtwCaptureError(
                f"Could not create ETW artifact directory under "
                f"{self.config.artifact_directory}: {error}"
            ) from error
        return self.run_directory

    def _resolve_executable(self, configured: str) -> str:
        candidate = Path(configured).expanduser()
        if candidate.is_absolute() or candidate.parent != Path("."):
            if candidate.is_file():
                return str(candidate.resolve())
            raise EtwCaptureError(f"ETW executable not found: {configured}")
        resolved = self._executable_lookup(configured)
        if resolved is None:
            raise EtwCaptureError(
                f"{configured} was not found on PATH. Install Windows Performance Toolkit "
                "or set its path in profile_config.json."
            )
        return resolved

    def _invoke(
        self, arguments: Sequence[str], *, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = self._runner(
                list(arguments),
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise EtwCaptureError(
                f"ETW command timed out after {self.config.timeout_seconds:g}s: "
                f"{arguments[0]}"
            ) from error
        except OSError as error:
            raise EtwCaptureError(f"Could not start ETW command {arguments[0]}: {error}") from error

        if check and result.returncode != 0:
            output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
            raise EtwCaptureError(
                f"ETW command failed ({result.returncode}): {' '.join(arguments)}"
                + (f"\n{output}" if output else "")
            )
        return result

    def _validate_wpr_profile(self) -> None:
        assert self._wpr_path is not None
        custom_path = None
        profile_name = self.config.wpr_profile
        if "!" in profile_name:
            custom_path, profile_name = profile_name.rsplit("!", 1)
        command = [self._wpr_path, "-profiles"]
        if custom_path is not None:
            command.append(custom_path)
        result = self._invoke(command)
        available_profiles = f"{result.stdout}\n{result.stderr}".casefold()
        profile_names = {profile_name.casefold()}
        base_name, separator, variant = profile_name.rpartition(".")
        if separator and variant.casefold() in {"light", "verbose"}:
            profile_names.add(base_name.casefold())
        if not any(name in available_profiles for name in profile_names):
            raise EtwCaptureError(
                f"WPR profile {profile_name!r} was not listed by {self._wpr_path} -profiles."
            )

    def _ensure_wpr_idle(self) -> None:
        assert self._wpr_path is not None
        result = self._invoke([self._wpr_path, "-status"], check=False)
        output = f"{result.stdout}\n{result.stderr}".casefold()
        if "recording is in progress" in output or "wpr recording is in progress" in output:
            raise EtwCaptureError(
                "WPR is already recording. The existing recording was left untouched; "
                "stop it manually before requesting deep capture."
            )
        if any(
            marker in output
            for marker in (
                "there are no trace profiles running",
                "not recording",
                "no recording is in progress",
            )
        ):
            return
        raise EtwCaptureError(
            "Could not confirm that WPR is idle. Refusing to start a system-wide capture. "
            f"WPR status output: {output.strip() or f'exit code {result.returncode}'}"
        )

    def start_capture(
        self, cell_number: int, notebook_cell_index: int, kernel_pid: int
    ) -> EtlCapture:
        if self.run_directory is None or self._wpr_path is None:
            raise EtwCaptureError("ETW profiler must pass preflight before capture starts.")
        if self._active_capture is not None:
            raise EtwCaptureError("A WPR capture is already active; captures must be sequential.")
        capture_id = uuid.uuid4().hex[:8]
        etl_path = self.run_directory / f"cell_{cell_number}_{capture_id}.etl"
        instance_name = f"rmsp-{self.run_directory.name}-{capture_id}"
        capture = EtlCapture(
            cell_number,
            notebook_cell_index,
            kernel_pid,
            etl_path,
            instance_name,
        )
        try:
            self._invoke(
                [
                    self._wpr_path,
                    "-start",
                    self.config.wpr_profile,
                    "-filemode",
                    "-instancename",
                    instance_name,
                ]
            )
        except EtwCaptureError as start_error:
            try:
                self._invoke(
                    [self._wpr_path, "-cancel", "-instancename", instance_name]
                )
            except EtwCaptureError as cancel_error:
                raise EtwCaptureError(
                    f"{start_error}; targeted WPR cancel also failed: {cancel_error}"
                ) from start_error
            raise
        self._active_capture = capture
        return capture

    def stop_capture(self, capture: EtlCapture) -> Path:
        if self._active_capture != capture:
            raise EtwCaptureError("Refusing to stop a WPR capture not owned by this run.")
        assert self._wpr_path is not None
        try:
            try:
                self._invoke(
                    [
                        self._wpr_path,
                        "-stop",
                        str(capture.etl_path),
                        f"Notebook cell {capture.cell_number} deep profile",
                        "-instancename",
                        capture.instance_name,
                    ]
                )
            except EtwCaptureError as error:
                capture.status = "stop_failed"
                cleanup_error = None
                try:
                    self._invoke(
                        [
                            self._wpr_path,
                            "-cancel",
                            "-instancename",
                            capture.instance_name,
                        ]
                    )
                except EtwCaptureError as cancel_error:
                    cleanup_error = str(cancel_error)
                capture.error = "; ".join(
                    message for message in (str(error), cleanup_error) if message
                )
                raise EtwCaptureError(capture.error) from error
        finally:
            self._active_capture = None
        if not capture.etl_path.is_file():
            raise EtwCaptureError(
                f"WPR reported success but did not create ETL: {capture.etl_path}"
            )
        capture.status = "captured"
        return capture.etl_path

    def export_capture(self, etl_path: Path, output_directory: Path) -> list[Path]:
        if self._wpaexporter_path is None:
            raise EtwCaptureError("ETW profiler must pass preflight before export.")
        output_directory.mkdir(parents=True, exist_ok=False)
        self._invoke(
            [
                self._wpaexporter_path,
                "-i",
                str(etl_path),
                "-profile",
                str(self.config.wpa_export_profile),
                "-outputfolder",
                str(output_directory),
            ]
        )
        csv_paths = sorted(path for path in output_directory.glob("*.csv") if path.is_file())
        if not csv_paths:
            raise EtwCaptureError(
                f"WPAExporter created no CSV files for ETL {etl_path}."
            )
        return csv_paths
