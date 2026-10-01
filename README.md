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

### Importing rmsp during profiling

If `rmsp` is not installed in the notebook kernel, set `rmsp_path` to the
**parent** directory containing the `rmsp` package (for example, use `C:/libs`
when the package is in `C:/libs/rmsp`). You can set it in the same JSON file:

```json
{
   "min_total_time_seconds": 1.0,
   "min_native_python_ratio": 1.0,
   "rmsp_path": "C:/libs"
}
```

Or override it for one run with `--rmsp-path`:

```powershell
python -m profiling_rmsp_agent path\to\notebook.ipynb `
   --rmsp-path C:\libs `
   --min-total-time-seconds 1.0 `
   --min-native-python-ratio 1.0
```

Relative JSON paths are resolved from the config file's directory; relative
CLI paths are resolved from the current working directory. The directory is
added to the kernel's Python import path for the timing pass and any optional
VTune replay. Each kernel shuts down after execution, so the path does not
persist in the CLI environment or the generated notebook. This does not change
the Windows DLL search path.

### VTune deep capture (Windows)

Deep capture is opt-in because it replays the notebook in a fresh kernel after
the timing pass. Earlier cells run again to reconstruct state; file writes,
database updates, network requests, prompts, and other side effects can repeat.
Only cells marked `deep_profile_candidate` get an active Intel VTune Profiler
sampling collection, attached directly to the replay kernel's PID; all
preceding code cells execute during replay without profiling.

Install Intel VTune Profiler and make sure the `vtune` command-line tool is on
`PATH` (or point `vtune.executable` at it in the config). Hardware event-based
sampling (the default) requires the Intel sampling driver to be installed and
may need an elevated/admin session; if that is not available, set
`"sampling_mode": "sw"` to use user-mode sampling instead (no driver required,
higher overhead).

The entire `vtune` config section is optional — `--capture-deep-profile` works
with no extra configuration when the profiled DLL already carries its own debug
symbols (it is loaded in-process by the same Python kernel being sampled). Add
a `vtune` section to `profile_config.json` only to override defaults or point
at extra symbol/source directories:

```json
{
   "min_total_time_seconds": 1.0,
   "min_native_python_ratio": 1.0,
   "vtune": {
      "executable": "vtune",
      "sampling_mode": "hw",
      "search_dirs": ["C:/path/to/dll/and/pdb"],
      "source_search_dirs": ["C:/path/to/cpp/sources"],
      "artifact_directory": "notebook_deep_profile_artifacts",
      "timeout_seconds": 120,
      "start_grace_seconds": 1.0
   }
}
```

Run the normal timing/profile pass and request the separate VTune replay:

```powershell
python -m profiling_rmsp_agent path\to\notebook.ipynb --capture-deep-profile
```

For each candidate cell, the command attaches `vtune -collect hotspots` to the
replay kernel's PID with `-duration unlimited`, executes the cell, then issues
`vtune -command stop` to finalize that cell's result directory (the only
documented way to end and finalize an unlimited-duration attach collection).
Each candidate gets its own result directory under
`<notebook>_deep_profile_artifacts/run-<id>/cell-<n>`; the aggregate JSON adds
a `deep_profile` section with per-cell status and result directory paths.
Inspect a result with `vtune-gui <result_dir>` or
`vtune -report hotspots -result-dir <result_dir>` — real DLL function names
resolve automatically as long as the DLL's symbols (PDB) are discoverable
(same directory as the DLL, or via `search_dirs`/`source_search_dirs`).

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
