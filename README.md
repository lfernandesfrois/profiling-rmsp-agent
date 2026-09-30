# Profiling RMSP Agent

Measures, for each executed Jupyter notebook cell, wall time and the split
between time spent in Python-level code and time spent in native/C-level
calls (e.g. numpy/pandas internals). Timing is computed live in the kernel
via `sys.setprofile`: no subprocess, no external profiler dependency, and
full notebook state (variables, imports) is shared normally across cells.

Install the project in the notebook's Python environment:

```powershell
python -m pip install -e .
```

## Command-line: profile a notebook end to end

```powershell
python -m profiling_rmsp_agent path\to\notebook.ipynb
```

This:

1. Creates a copy named `notebook_profile.ipynb` next to the original
   (fails without overwriting if that file already exists).
2. Inserts `%load_ext profiling_rmsp_agent.live_cell_profiler` as the first
   cell and `live_cell_profiler.to_dicts()` as the last cell, unless those
   exact lines are already present.
3. Executes the copy from start to finish, using the notebook's own
   `kernelspec` (the kernel must have `profiling_rmsp_agent` installed).
4. Writes the collected timings to `notebook_profile.json` next to the copy.

If a cell raises an error, execution stops there. The JSON report still gets
written, with `status: "failed"`, a `failed_cell` entry (`cell_number`,
`ename`, `evalue`), and timings for every cell that ran before the failure
(including the failed cell's own partial timing). The command prints the
failing cell and error to the terminal and exits with a non-zero status.

## Deep-profile candidates

Every cell execution record in the JSON report includes a boolean
`deep_profile_candidate`. It is `true` only when both conditions hold:

- `total_time_seconds > min_total_time_seconds`
- `native_time_seconds > min_native_python_ratio * python_time_seconds`

The second condition is equivalent to a strict native/Python time ratio
comparison without dividing by zero. When Python time is zero, positive native
time satisfies the ratio condition; when both are zero, it does not.

The CLI looks for `profile_config.json` beside the input notebook. For example:

```json
{
   "min_total_time_seconds": 1.0,
   "min_native_python_ratio": 1.0
}
```

Use `--config PATH` to select another JSON file. Command-line values override
the matching file values:

```powershell
python -m profiling_rmsp_agent path\to\notebook.ipynb `
   --min-total-time-seconds 1.0 `
   --min-native-python-ratio 1.0
```

If either threshold is still missing after loading the file and applying CLI
overrides, the command stops before creating or executing the profiled notebook.

### ETW deep capture (Windows)

Deep capture is opt-in because it replays the notebook in a fresh kernel after
the timing pass. Earlier cells run again to reconstruct state; file writes,
database updates, network requests, prompts, and other side effects can repeat.
Only cells marked `deep_profile_candidate` receive an active WPR capture, but
all preceding code cells execute during replay.

Install Windows Performance Toolkit (WPT), then create a WPA export profile in
WPA with the **CPU Usage (Sampled) by Process, Thread** table and save it as a
`.wpaProfile`. WPAExporter requires this profile; there is no generic
export-everything mode. The exported CSV filename and column headers are
specific to that profile/WPT version, so inspect an export and copy its actual
filename and headers into the config.

Add an `etw` section to `profile_config.json` beside the notebook (paths can be
absolute or relative to the config file):

```json
{
   "min_total_time_seconds": 1.0,
   "min_native_python_ratio": 1.0,
   "etw": {
      "wpr_profile": "profiles/CPU.wprp!CPUProfile.Verbose",
      "wpa_export_profile": "C:/path/to/CPU-Sampled.wpaProfile",
      "artifact_directory": "notebook_deep_profile_artifacts",
      "cpu_csv_pattern": "replace-with-the-exported-cpu-csv-filename.csv",
      "process_id_column": "replace-with-the-CSV-process-id-header",
      "thread_id_column": "replace-with-the-CSV-thread-id-header",
      "cpu_time_column": "replace-with-the-CSV-sampled-cpu-time-header",
      "cpu_time_unit": "ms",
      "timeout_seconds": 120
   }
}
```

Run the normal timing/profile pass and request the separate ETW replay:

```powershell
python -m profiling_rmsp_agent path\to\notebook.ipynb --capture-deep-profile
```

The command checks WPT, the profiles, and WPR's idle state before executing the
notebook. It never stops a WPR recording started outside this run. Each
candidate gets its own ETL and CSV directory under the artifact folder; the
aggregate JSON adds a `deep_profile` section with per-cell status and file
references plus sampled CPU totals for the kernel process. Sampled CPU time is
an estimate from ETW samples, not cell wall-clock time or exact unsampled CPU
time. Run WPT from an appropriately elevated prompt if the selected profile
requires it.

## Using the extension directly in a notebook

In the notebook, load the extension as the first cell:

```python
%load_ext profiling_rmsp_agent.live_cell_profiler
```

Every cell executed afterward is measured automatically. Inspect results with:

```python
live_cell_profiler.to_dicts()
```

Each record has `cell_number`, `total_time_seconds` (wall time), and
`python_time_seconds`/`native_time_seconds` with `python_percent`/
`native_percent` (the Python/native split relative to each other, summing to
100% when at least one function call was observed). A cell that makes no
function calls reports a near-zero split (self-instrumentation overhead only).

## Development

```powershell
python -m pip install -e .
python -m unittest discover -s tests
```
