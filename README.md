# Profiling RMPS Agent

An agent for profiling Jupyter notebook cells and escalating expensive cells
to deeper Python/native analysis.

## Current scope

The initial core defines a normalized cell profile and deterministic rules for
deciding whether a profile should be escalated. The optional IPython extension
records per-cell wall time and Python-traced memory. Native time, process RSS,
deep profiling, and LLM analysis are not implemented yet.

## Development

Create and activate a Python 3.10+ environment, then install the project in
editable mode and run the tests:

```powershell
python -m pip install -e .
python -m unittest discover -s tests
```

In an IPython or Jupyter kernel, load the extension with `%load_ext
profiling_rmps_agent.ipython`. The profiler is available as
`notebook_profiler`; inspect captured records with
`notebook_profiler.to_dicts()`. Each record includes a deterministic
`deep_profile` decision; candidates are also available as
`notebook_profiler.deep_profile_candidates`. Memory deltas come from
`tracemalloc` and do not include native allocations.