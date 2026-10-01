"""Command-line interface: profile a notebook end to end."""

import argparse
from pathlib import Path
import sys
from typing import Sequence

from profiling_rmsp_agent.notebook_copy import NotebookCopyError, create_instrumented_notebook
from profiling_rmsp_agent.notebook_run import run_notebook
from profiling_rmsp_agent.deep_profile import (
    DeepProfileConfigError,
    resolve_deep_profile_policy,
    resolve_rmsp_path,
)
from profiling_rmsp_agent.deep_profile_run import run_candidate_replay
from profiling_rmsp_agent.vtune_capture import (
    VtuneCaptureError,
    WindowsVtuneProfiler,
    resolve_vtune_config,
)
from profiling_rmsp_agent.report import build_report, write_report


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a profiled notebook copy, run it end to end, and save "
            "per-cell Python/native timing to a JSON report."
        )
    )
    parser.add_argument("notebook", type=Path, help="input .ipynb file")
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON config file (default: profile_config.json beside the notebook)",
    )
    parser.add_argument(
        "--rmsp-path",
        type=Path,
        help="parent directory of the rmsp package, added only to notebook kernels",
    )
    parser.add_argument(
        "--min-total-time-seconds",
        type=float,
        help="candidate minimum total cell time; overrides config",
    )
    parser.add_argument(
        "--min-native-python-ratio",
        type=float,
        help="candidate minimum native/Python time ratio; overrides config",
    )
    parser.add_argument(
        "--capture-deep-profile",
        action="store_true",
        help=(
            "replay the notebook in a fresh kernel and attach Intel VTune Profiler by PID to "
            "candidate cells; cells execute a second time"
        ),
    )
    parsed = parser.parse_args(arguments)

    try:
        deep_profile_policy = resolve_deep_profile_policy(
            notebook_path=parsed.notebook,
            config_path=parsed.config,
            cli_min_total_time_seconds=parsed.min_total_time_seconds,
            cli_min_native_python_ratio=parsed.min_native_python_ratio,
        )
        rmsp_path = resolve_rmsp_path(parsed.notebook, parsed.config, parsed.rmsp_path)
    except DeepProfileConfigError as error:
        parser.error(str(error))

    vtune_profiler = None
    if parsed.capture_deep_profile:
        try:
            vtune_config = resolve_vtune_config(parsed.notebook, parsed.config)
            vtune_profiler = WindowsVtuneProfiler(vtune_config)
            vtune_profiler.preflight()
        except VtuneCaptureError as error:
            parser.error(str(error))

    try:
        instrumented = create_instrumented_notebook(parsed.notebook)
    except NotebookCopyError as error:
        parser.error(str(error))

    print(f"Created {instrumented.output_path}")
    result = run_notebook(instrumented.output_path, rmsp_path=rmsp_path)

    report_path = instrumented.output_path.with_suffix(".json")
    report = build_report(
        notebook_path=parsed.notebook,
        profiled_notebook_path=instrumented.output_path,
        cells=result.cells,
        failed_cell=result.failed_cell,
        deep_profile_policy=deep_profile_policy,
    )
    write_report(report, report_path)

    deep_profile_failed = False
    if vtune_profiler is not None:
        candidates = [
            cell for cell in report["cells"] if cell["deep_profile_candidate"]
        ]
        if candidates:
            index_offset = int(instrumented.load_ext_cell_inserted)
            source_cell_indices = {
                execution_count: cell_index - index_offset
                for execution_count, cell_index in result.source_cell_indices.items()
                if 0 <= cell_index - index_offset
            }
            deep_result = run_candidate_replay(
                parsed.notebook,
                candidates,
                source_cell_indices,
                vtune_profiler,
                rmsp_path=rmsp_path,
            )
            report["deep_profile"] = deep_result.to_dict()
            deep_profile_failed = deep_result.status in {
                "failed",
                "completed_with_errors",
            }
        else:
            report["deep_profile"] = {
                "status": "no_candidates",
                "summary": {
                    "candidate_count": 0,
                    "captured_count": 0,
                    "capture_failed_count": 0,
                    "not_reached_count": 0,
                },
                "failed_cell": None,
                "error": None,
                "cells": [],
            }
        write_report(report, report_path)
    print(f"Wrote {report_path}")

    if result.failed_cell is not None:
        print(
            f"Notebook execution stopped at cell {result.failed_cell.cell_number}: "
            f"{result.failed_cell.ename}: {result.failed_cell.evalue}",
            file=sys.stderr,
        )
        return 1

    if deep_profile_failed:
        print("Deep-profile capture completed with errors.", file=sys.stderr)
        return 1

    print(f"Profiled {len(result.cells)} cell(s) successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
