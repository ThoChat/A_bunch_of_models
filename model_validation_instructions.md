# Instructions: implement and validate a crowd model in JuPedSim

Your task: implement the model in `<MODEL>/` as a Python
JuPedSim operational model, then build a validation notebook that reproduces
the source paper's main validation results. Every figure should compare
directly with the paper's own data, and every scenario should have a
trajectory animation. `SFM/` and `RVO/` are finished examples of exactly
this. Read them first and copy their structure and tone.

## 0. Read first

- `<MODEL>/model_description.md` and the paper PDF in `<MODEL>/`. List every
  validation experiment the paper reports, and what it measures for each
  (numbers, plots, or only qualitative claims).
- `RVO/pyrvo.py`, `RVO/validation/*.py`, `RVO/build_notebook.py`. This is
  the closest template for a velocity-based model.
- `SFM/pysocial_force.py`, `SFM/build_notebook.py`. This is the template for
  force-based models, and for hand-digitized paper curves.

## 1. Target layout

```
<MODEL>/
  py<model>.py            # model: frozen state dataclass + CustomOperationalModel subclass
  build_notebook.py       # generates <MODEL>_validation.ipynb with nbformat (md()/code() helpers)
  <MODEL>_validation.ipynb
  validation/
    analysis.py           # shared trajectory loading + metrics
    run_<scenario>.py     # one runner per paper experiment (argparse, --tag)
    results/              # .sqlite trajectories, .json summaries, plot_*.png, paper_fig*.png crops
```

Never edit the `.ipynb` by hand. Edit `build_notebook.py`, then regenerate
and execute the notebook.

## 2. Environment

- Use the shared venv: `source .venv-python-model/bin/activate` at the repo
  root. JuPedSim is already compiled, so do **not** run ninja or
  `compile_and_run.sh`, and do not modify JuPedSim's C++ code.
- Standalone scripts also need `source jupedsim/build/environment`, which
  sets `PYTHONPATH`.
- The notebook's **first code cell** must add those paths to `sys.path`
  itself. Copy it from `RVO/build_notebook.py`. It must end with
  `print("jupedsim import OK")`.
- Execute the notebook with
  `jupyter nbconvert --to notebook --execute --inplace <MODEL>_validation.ipynb`
  from inside `<MODEL>/`, then check that no cell has an error output.

## 3. The model (`py<model>.py`)

- Subclass `jupedsim.models.custom_model.CustomOperationalModel` and
  implement `compute_next_state(self, state, step) -> (new_state, (dx, dy))`.
- Make the state a `@dataclass(kw_only=True, frozen=True)`. Always return a
  new object via `dataclasses.replace`. Never return `state` itself, and
  never mutate it or a neighbour's state.
- Things the step gives you:
  - `step.dt`
  - `step.orientation_to_next_target` (a unit vector, zero once the goal
    is reached)
  - `step.other_agents_in_range(r)`, where each neighbour has
    `.relative_position` and `.state`
  - `step.walls_in_range(d)`, where each wall has `.closest_point`,
    `.distance` and `.normal`, all relative to the agent
- Implement the paper's equations faithfully. Any practical extension
  beyond the paper (an overlap fallback, extra candidate samples, etc.)
  needs a short comment saying why, and a mention in the notebook.
- If a scenario needs special agents (like RVO's non-reactive car), prefer
  a flag in the one state class over a separate model.
- Vectorize per-agent work with numpy. The callback runs once per agent
  per step in pure Python.

## 4. Scenarios (`validation/run_<scenario>.py`)

- Follow the paper's setup as closely as possible: geometry, agent count,
  speeds and radii. Where you must deviate (for compute time or
  stability), deviate as little as possible and document why in the
  notebook.
- **Goals:** `sim.add_waypoint_stage(pos, distance)` with a one-stage
  `jps.JourneyDescription`. Agents are not removed on arrival. Use exit
  stages only if the paper's agents leave the scene.
- **Obstacles:** build a `shapely.Polygon(outer, holes=[...])` and pass it
  as `geometry`. `excluded_areas=` is **silently ignored** by this build.
- **Trajectory writer:** `SqliteTrajectoryWriter`. You **must** call
  `writer.close()` after the loop, or short runs silently lose all their
  data.
- Wrap the simulation loop in `try/except`. Record
  `crashed: {step, time, error}` and all parameters in the JSON summary.
- IF the information is not given in the paper the **Pick `dt` empirically:** 
  try progressively smaller values and report the chosen one and why. 
  Stiff force models need very small `dt`
  (SFM: 1e-4); velocity-based models need much less (RVO: 0.025).
- **Timing matters as much as speed.** Place and time the agents so that
  the interaction the paper shows actually happens. Then check that it
  does happen: for example, count who gave way, and measure delays or
  minimum clearance. An agent that "crosses behind" a car that had
  already gone by did not avoid anything.
- For multi-point sweeps, run each point as its own OS process, since the
  Python callback holds the GIL.

## 5. Comparing with the paper (required in every figure)

- **Numeric plots in the paper:** hand-digitize them. Render the page with
  `pdftoppm -r 300`, crop the figure, read the points against the
  gridlines, and store them as `PAPER_FIG<n>` dicts in one reference cell.
  State the method and the approximate error in that cell's markdown.
  Plot them as dashed black lines or markers labelled "paper, Fig. n
  (digitized)".
- **Claims stated in words** (e.g. "no collisions", "all agents reach
  their goals"): turn each into a reference value, such as a line at
  clearance 0 m or at 100%, and draw it on the relevant plot.
- **Qualitative figures** (trajectory pictures, snapshots): crop them from
  the PDF into `validation/results/paper_fig<n>_*.png`. Show each next to
  the matching plot of ours, with the same layout and ordering.
- **Check figure numbers** against the PDF text. Don't guess them.
- Every scenario gets an animation via
  `from jupedsim.internal.notebook_utils import animate, read_sqlite_file`.
  Downsample with `every_nth_frame` to keep the notebook small.

## 6. Checking your metrics

Before you write a verdict, check each metric against the raw trajectories
and a snapshot plot. Two real bugs from past work:

- **Arrival counted against a shared target point.** A group sharing one
  target can't all be within 0.6 m of it. This reported 23% arrival when
  the true figure was 98%.
- **Two scenarios with the same label** were merged into one bar by
  matplotlib, which shifted every value label by one row.

Look at every generated PNG before reporting.

## 7. Notebook structure (`build_notebook.py`)

1. Title and citation, plus a summary of how the paper validated the model
   (from `model_description.md`).
2. The `sys.path` / jupedsim import cell.
3. The setup cell, with the same matplotlib rcParams as SFM/RVO.
4. The paper reference-data cell (digitized values, textual reference
   values, image paths).
5. Implementation notes: the real bugs found and the `dt` choice.
6. One section per paper experiment:
   1. Paper setup vs. our setup.
   2. Results table.
   3. Figures with the paper overlaid.
   4. Animation.
   5. A plain verdict.
7. A summary table: `| # | Phenomenon (paper) | Reproduced? | Notes |`,
   using ✅ / ⚠️ / ❌.

## 8. Reporting standard

- Report what actually happened. Don't tune parameters until the result
  looks good, and don't invent problems or results.
- Where the model does not reproduce the paper, say so, give numbers, and
  say what you checked in order to rule out causes.
- Separate "shape reproduced" from "absolute numbers reproduced". Old
  hardware or a different engine explains differences in absolute speed,
  but only claim that if you have actually compared.
- When done, report: the files created, the `dt`, the bugs found, a
  reproduced/not-reproduced verdict per experiment with key numbers,
  confirmation that the notebook ran cleanly, and any open issues.
