# Profiling RMSP Agent

An IPython extension that measures, for each executed notebook cell, wall
time and the split between time spent in Python-level code and time spent in
native/C-level calls (e.g. numpy/pandas internals), computed live in the same
kernel via `sys.setprofile`. No subprocess, no external profiler dependency,
and full notebook state (variables, imports) is shared normally across cells.

Install the project in the notebook's Python environment:

```powershell
python -m pip install -e .
```

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
