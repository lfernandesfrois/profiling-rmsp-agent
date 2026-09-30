"""Parse configured WPA CSV output for one kernel process."""

import csv
from dataclasses import dataclass
from pathlib import Path

from profiling_rmsp_agent.etw_capture import EtwCaptureError, EtwConfig


@dataclass(frozen=True, slots=True)
class CpuCsvSummary:
    csv_path: Path
    kernel_pid: int
    sampled_cpu_time_seconds: float
    thread_rows: int
    distinct_threads: int

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "csv_path": str(self.csv_path),
            "kernel_pid": self.kernel_pid,
            "sampled_cpu_time_seconds": self.sampled_cpu_time_seconds,
            "thread_rows": self.thread_rows,
            "distinct_threads": self.distinct_threads,
        }


def summarize_kernel_cpu_csvs(
    csv_paths: list[Path], config: EtwConfig, kernel_pid: int
) -> CpuCsvSummary:
    """Sum sampled CPU time for rows belonging to the kernel PID and its threads."""
    selected_paths = [
        path for path in csv_paths if path.match(config.cpu_csv_pattern)
    ]
    if len(selected_paths) != 1:
        raise EtwCaptureError(
            f"Expected exactly one WPA CSV matching {config.cpu_csv_pattern!r}; "
            f"found {len(selected_paths)}. Check the exported WPA profile and CSV pattern."
        )

    csv_path = selected_paths[0]
    total_metric = 0.0
    thread_rows = 0
    thread_ids: set[str] = set()
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file, delimiter=config.csv_delimiter)
            fieldnames = set(reader.fieldnames or ())
            required = {
                config.process_id_column,
                config.thread_id_column,
                config.cpu_time_column,
            }
            missing = required - fieldnames
            if missing:
                raise EtwCaptureError(
                    f"WPA CSV {csv_path} is missing configured column(s): "
                    + ", ".join(sorted(missing))
                )

            for row_number, row in enumerate(reader, start=2):
                process_id = _parse_integer(row.get(config.process_id_column, ""))
                thread_text = (row.get(config.thread_id_column) or "").strip()
                if process_id != kernel_pid or not thread_text:
                    continue
                thread_id = _parse_integer(thread_text)
                if thread_id is None:
                    continue
                metric_text = (row.get(config.cpu_time_column) or "").strip()
                if not metric_text:
                    continue
                try:
                    metric_value = float(metric_text)
                except ValueError as error:
                    raise EtwCaptureError(
                        f"Non-numeric value in {config.cpu_time_column!r} at "
                        f"{csv_path}:{row_number}: {metric_text!r}"
                    ) from error
                total_metric += metric_value
                thread_rows += 1
                thread_ids.add(str(thread_id))
    except OSError as error:
        raise EtwCaptureError(f"Could not read WPA CSV {csv_path}: {error}") from error

    unit_to_seconds = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "100ns": 1e-7}
    return CpuCsvSummary(
        csv_path=csv_path,
        kernel_pid=kernel_pid,
        sampled_cpu_time_seconds=total_metric * unit_to_seconds[config.cpu_time_unit],
        thread_rows=thread_rows,
        distinct_threads=len(thread_ids),
    )


def _parse_integer(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return int(text, 0)
    except ValueError:
        try:
            return int(text)
        except ValueError:
            return None