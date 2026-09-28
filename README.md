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

## Use it from a notebook

Install the package into the same Python environment used by the notebook
kernel. From the repository directory, run:

```powershell
python -m pip install -e .
```

In the notebook, select that same environment as the kernel, then run this as
the first code cell:

```python
%load_ext profiling_rmps_agent.ipython
```

The extension attaches to the active IPython kernel. Every cell executed after
that cell is measured automatically. For example, run this in the next cell:

```python
values = [number * number for number in range(1_000_000)]
len(values)
```

Then inspect the collected records in another cell:

```python
notebook_profiler.to_dicts()
```

The profiler object is placed in the notebook namespace by the extension. A
record contains the cell number, wall-clock duration, Python-traced memory
delta, and the deterministic `deep_profile` decision. Candidates are also
available with:

```python
notebook_profiler.deep_profile_candidates
```

If `%load_ext` reports that the module cannot be found, the notebook is using
a different Python environment from the one where `python -m pip install -e .`
was run. Select the correct kernel and run `%reload_ext
profiling_rmps_agent.ipython`.

Memory deltas come from `tracemalloc` and do not include native allocations.
Native profiling is not implemented yet.