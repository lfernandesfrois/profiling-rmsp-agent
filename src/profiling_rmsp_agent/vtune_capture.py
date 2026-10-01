"""Intel VTune Profiler CLI integration: attach-by-PID sampling per candidate cell."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Protocol, Sequence

from profiling_rmsp_agent.deep_profile import DEFAULT_CONFIG_NAME


_VALID_SAMPLING_MODES = {"hw", "sw"}
_DEFAULT_EXECUTABLE = "vtune"
_DEFAULT_SAMPLING_MODE = "hw"
_DEFAULT_TIMEOUT_SECONDS = 120.0
_DEFAULT_START_GRACE_SECONDS = 1.0


class VtuneCaptureError(RuntimeError):
    """Raised when a VTune attach/collect/stop step cannot safely proceed."""


class _Popen(Protocol):
    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def kill(self) -> None: ...
    stdout: Any
    stderr: Any


PopenFactory = Callable[..., _Popen]
ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class VtuneConfig:
    executable: str
    sampling_mode: str
    search_dirs: tuple[str, ...]
    source_search_dirs: tuple[str, ...]
    artifact_directory: Path
    timeout_seconds: float
    start_grace_seconds: float


@dataclass(slots=True)
class VtuneCapture:
    cell_number: int
    notebook_cell_index: int
    kernel_pid: int
    result_dir: Path
    status: str = "pending"
    error: str | None = None


def _read_config(config_path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise VtuneCaptureError(
            f"Could not read config file {config_path}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise VtuneCaptureError(
            f"Config file {config_path} is not valid JSON: {error}"
        ) from error
    if not isinstance(loaded, dict):
        raise VtuneCaptureError(f"Config file {config_path} must contain a JSON object.")
    return loaded


def resolve_vtune_config(notebook_path: Path, config_path: Path | None) -> VtuneConfig:
    """Load optional VTune settings from the notebook's profiling config.

    The entire ``vtune`` section (and the config file itself) is optional; every
    field has a documented default so ``--capture-deep-profile`` works out of the
    box when the DLL under test already carries its own debug symbols.
    """
    selected_config = config_path or notebook_path.resolve().parent / DEFAULT_CONFIG_NAME

    vtune_values: dict[str, Any] = {}
    if config_path is not None or selected_config.exists():
        document = _read_config(selected_config)
        configured = document.get("vtune", {})
        if configured and not isinstance(configured, dict):
            raise VtuneCaptureError(f"Config {selected_config} key 'vtune' must be a JSON object.")
        vtune_values = configured or {}

    executable = vtune_values.get("executable", _DEFAULT_EXECUTABLE)
    if not isinstance(executable, str) or not executable.strip():
        raise VtuneCaptureError("vtune.executable must be a non-empty string.")

    sampling_mode = vtune_values.get("sampling_mode", _DEFAULT_SAMPLING_MODE)
    if not isinstance(sampling_mode, str) or sampling_mode.lower() not in _VALID_SAMPLING_MODES:
        valid = ", ".join(sorted(_VALID_SAMPLING_MODES))
        raise VtuneCaptureError(f"vtune.sampling_mode must be one of: {valid}.")

    search_dirs = _resolve_dir_list(vtune_values.get("search_dirs"), "search_dirs", selected_config)
    source_search_dirs = _resolve_dir_list(
        vtune_values.get("source_search_dirs"), "source_search_dirs", selected_config
    )

    artifact_directory_value = vtune_values.get("artifact_directory")
    if artifact_directory_value is None:
        artifact_directory = notebook_path.resolve().parent / (
            f"{notebook_path.stem}_deep_profile_artifacts"
        )
    else:
        if not isinstance(artifact_directory_value, str) or not artifact_directory_value.strip():
            raise VtuneCaptureError("vtune.artifact_directory must be a non-empty path.")
        artifact_directory = Path(artifact_directory_value).expanduser()
        if not artifact_directory.is_absolute():
            artifact_directory = selected_config.resolve().parent / artifact_directory

    timeout_seconds = vtune_values.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS)
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise VtuneCaptureError("vtune.timeout_seconds must be a positive number.")

    start_grace_seconds = vtune_values.get("start_grace_seconds", _DEFAULT_START_GRACE_SECONDS)
    if (
        isinstance(start_grace_seconds, bool)
        or not isinstance(start_grace_seconds, (int, float))
        or start_grace_seconds < 0
    ):
        raise VtuneCaptureError("vtune.start_grace_seconds must be a non-negative number.")

    return VtuneConfig(
        executable=executable,
        sampling_mode=sampling_mode.lower(),
        search_dirs=tuple(search_dirs),
        source_search_dirs=tuple(source_search_dirs),
        artifact_directory=artifact_directory,
        timeout_seconds=float(timeout_seconds),
        start_grace_seconds=float(start_grace_seconds),
    )


def _resolve_dir_list(value: Any, key: str, config_path: Path) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise VtuneCaptureError(f"vtune.{key} must be a list of non-empty path strings.")
    resolved: list[str] = []
    for item in value:
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = config_path.resolve().parent / path
        if not path.is_dir():
            raise VtuneCaptureError(f"vtune.{key} entry not found or not a directory: {path}")
        resolved.append(str(path.resolve()))
    return resolved


class WindowsVtuneProfiler:
    """Own sequential VTune attach-by-PID captures for one deep-profile run."""

    def __init__(
        self,
        config: VtuneConfig,
        popen: PopenFactory = subprocess.Popen,
        runner: ProcessRunner = subprocess.run,
        executable_lookup: Callable[[str], str | None] = shutil.which,
        platform_name: str = sys.platform,
    ) -> None:
        self.config = config
        self._popen = popen
        self._runner = runner
        self._executable_lookup = executable_lookup
        self._platform_name = platform_name
        self._vtune_path: str | None = None
        self._active_capture: VtuneCapture | None = None
        self._active_popen: _Popen | None = None
        self.run_directory: Path | None = None

    def preflight(self) -> Path:
        """Resolve the VTune executable, sanity-check it, and create a run directory."""
        if self._platform_name != "win32":
            raise VtuneCaptureError("VTune deep capture is supported only on Windows.")

        self._vtune_path = self._resolve_executable(self.config.executable)
        self._invoke([self._vtune_path, "-help", "collect", "hotspots"])

        try:
            self.config.artifact_directory.mkdir(parents=True, exist_ok=True)
            self.run_directory = Path(
                tempfile.mkdtemp(prefix="run-", dir=self.config.artifact_directory)
            )
        except OSError as error:
            raise VtuneCaptureError(
                f"Could not create VTune artifact directory under {self.config.artifact_directory}: {error}"
            ) from error
        return self.run_directory

    def _resolve_executable(self, configured: str) -> str:
        candidate = Path(configured).expanduser()
        if candidate.is_absolute() or candidate.parent != Path("."):
            if candidate.is_file():
                return str(candidate.resolve())
            raise VtuneCaptureError(f"VTune executable not found: {configured}")
        resolved = self._executable_lookup(configured)
        if resolved is None:
            raise VtuneCaptureError(
                f"{configured} was not found on PATH. Install Intel VTune Profiler "
                "or set its path in profile_config.json."
            )
        return resolved

    def _invoke(self, arguments: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        try:
            result = self._runner(
                list(arguments),
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise VtuneCaptureError(
                f"VTune command timed out after {self.config.timeout_seconds:g}s: {arguments[0]}"
            ) from error
        except OSError as error:
            raise VtuneCaptureError(f"Could not start VTune command {arguments[0]}: {error}") from error

        if check and result.returncode != 0:
            output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
            raise VtuneCaptureError(
                f"VTune command failed ({result.returncode}): {' '.join(arguments)}"
                + (f"\n{output}" if output else "")
            )
        return result

    def start_capture(self, cell_number: int, notebook_cell_index: int, kernel_pid: int) -> VtuneCapture:
        if self.run_directory is None or self._vtune_path is None:
            raise VtuneCaptureError("VTune profiler must pass preflight before capture starts.")
        if self._active_capture is not None:
            raise VtuneCaptureError("A VTune capture is already active; captures must be sequential.")

        result_dir = self.run_directory / f"cell-{cell_number}"
        if result_dir.exists():
            raise VtuneCaptureError(f"VTune result directory already exists: {result_dir}")

        command = [
            self._vtune_path,
            "-collect",
            "hotspots",
            "-knob",
            f"sampling-mode={self.config.sampling_mode}",
            "-target-pid",
            str(kernel_pid),
            "-result-dir",
            str(result_dir),
            "-duration",
            "unlimited",
        ]
        for directory in self.config.search_dirs:
            command += ["-search-dir", directory]
        for directory in self.config.source_search_dirs:
            command += ["-source-search-dir", directory]

        try:
            process = self._popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except OSError as error:
            raise VtuneCaptureError(f"Could not start VTune command {command[0]}: {error}") from error

        if self.config.start_grace_seconds > 0:
            time.sleep(self.config.start_grace_seconds)
        if process.poll() is not None:
            stdout = process.stdout.read() if process.stdout is not None else ""
            stderr = process.stderr.read() if process.stderr is not None else ""
            output = "\n".join(part for part in (stdout, stderr) if part).strip()
            raise VtuneCaptureError(
                f"VTune exited immediately (code {process.poll()}) starting capture for cell "
                f"{cell_number}: {' '.join(command)}" + (f"\n{output}" if output else "")
            )

        capture = VtuneCapture(
            cell_number=cell_number,
            notebook_cell_index=notebook_cell_index,
            kernel_pid=kernel_pid,
            result_dir=result_dir,
            status="recording",
        )
        self._active_capture = capture
        self._active_popen = process
        return capture

    def stop_capture(self, capture: VtuneCapture) -> Path:
        if self._active_capture != capture or self._active_popen is None:
            raise VtuneCaptureError("Refusing to stop a VTune capture not owned by this run.")
        assert self._vtune_path is not None
        process = self._active_popen

        try:
            errors: list[str] = []
            try:
                self._invoke(
                    [self._vtune_path, "-command", "stop", "-result-dir", str(capture.result_dir)]
                )
            except VtuneCaptureError as error:
                errors.append(str(error))

            try:
                process.wait(timeout=self.config.timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                errors.append(
                    "VTune process did not exit after -command stop; it was force-killed."
                )

            if errors:
                capture.status = "stop_failed"
                capture.error = "; ".join(errors)
                raise VtuneCaptureError(capture.error)
        finally:
            self._active_capture = None
            self._active_popen = None

        if not capture.result_dir.is_dir() or not any(capture.result_dir.iterdir()):
            raise VtuneCaptureError(
                f"VTune reported success but produced no result directory contents: {capture.result_dir}"
            )
        capture.status = "captured"
        return capture.result_dir
