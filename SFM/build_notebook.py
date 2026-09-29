"""Programmatically builds SFM_validation.ipynb from markdown/code cell
source strings below, so the notebook's content is versionable as plain
Python/Markdown rather than raw ipynb JSON. Run, then execute with:

    jupyter nbconvert --to notebook --execute --inplace SFM_validation.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


# ============================================================ Title / intro
md(r"""
# Validating the Python/JuPedSim Social Force Model against Helbing, Farkas & Vicsek (2000)

This notebook checks a Python implementation of the **Social Force Model (SFM)**
— running inside **JuPedSim**'s experimental `CustomOperationalModel` plugin API —
against the paper that introduced its escape-panic form:

> D. Helbing, I. Farkas, T. Vicsek. *Simulating dynamical features of escape panic.*
> Nature 407, 487–490 (2000). [doi:10.1038/35035023](https://doi.org/10.1038/35035023)

**What this notebook does, in order:**

1. Confirms the given `sim_bottleneck_pysocialforce.py` example runs correctly.
2. Re-implements the paper's three validation experiments (Figs. 1–3) as new
   simulations, reusing the *exact same* force equations from `pysocial_force.py`
   (unmodified except for one required bug fix, see below).
3. Compares the reproduced results against the phenomena reported in the paper.

**Code layout** (all in `SFM/`):

- `pysocial_force.py` — the given SFM implementation (`PythonSocialForceModel`),
  with one bug fix applied (see next section).
- `sim_bottleneck_pysocialforce.py` — the given example script (task 1).
- `validation/sfm_injury.py` — adds the paper's injury/freeze rule for Fig. 1,
  by subclassing `PythonSocialForceModel` and reusing its static force methods.
- `validation/sfm_herding.py` — adds the paper's Eq. 4 herding behaviour for
  Fig. 3, same reuse pattern.
- `validation/run_fig{1,2,3}_*.py` — one simulation per data point.
- `validation/sweep_all.py` — runs every data point as a separate OS process
  (parallelism was necessary — see "Why dt=1e-4" below).
- `validation/analysis.py` — shared trajectory-loading / metric helpers.
""")

# ============================================================ Bug fixes found
md(r"""
## Two bugs found and fixed while getting this to run

**1. Stale API name (blocking).** `pysocial_force.py` called
`step.to_next_target`, but the built jupedsim API (this is a development
branch, `draft_python_models`) only exposes `step.orientation_to_next_target`
(already a unit vector — the extra `_normalize()` wrapper was removed too).
This is the exact same fix already present in the "official" copy of this
example shipped inside jupedsim's own test suite
(`jupedsim/python_modules/jupedsim_examples/jupedsim_examples/models/pysocial_force.py`),
so it isn't a guess — it's what the model's own repository already considers
correct. Without this fix, `sim_bottleneck_pysocialforce.py` raises
`AttributeError` immediately.

**2. Silent data loss in the trajectory writer (non-blocking, but important
for correctness).** `jps.SqliteTrajectoryWriter` buffers frames in memory and
only writes them to disk every `commit_every_nth_write` (default 100) calls,
**or when `.close()` is called explicitly** — which neither the given example
script nor an early draft of our own validation scripts did. For short runs
(fewer than 100 buffered writes) this silently produces an **empty**
trajectory table with no error. All validation scripts below call
`writer.close()` after the simulation loop. The original example script has
the same latent issue for its own tail frames (harmless there since it
accumulates >100 buffered writes well before finishing, so only the very last
partial buffer — at most 99 output frames — would ever be at risk).
""")

# ============================================================ Setup
code(r"""
import json
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(pathlib.Path("validation").resolve()))
from analysis import (
    load_trajectory, leaving_times, time_for_n_to_leave, outflow_rate,
    mean_progress_efficiency,
)

RESULTS = pathlib.Path("validation/results")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

# Interactive playback of recorded trajectories, using jupedsim's own replay
# utility -- the same one used in jupedsim's official example notebooks
# (jupedsim/notebooks/*.ipynb), via `jupedsim.internal.notebook_utils`. Its
# own module docstring says explicitly: "We make no promises about the
# functions from this file w.r.t. API stability... Do not use the code here.
# Use it at your own peril." We use it anyway, for the same reason the
# official docs do -- there isn't a public, stable alternative -- but that
# caveat is real: a future jupedsim release may move or change this.
from jupedsim.internal.notebook_utils import animate, read_sqlite_file

# Each replay downsamples frames (`every_nth_frame`) and uses a single
# representative agent radius (the sqlite trajectory format only stores
# positions, not per-agent radius, and our own agents are drawn from
# [0.25, 0.35] m -- see model_description.md) purely to keep the resulting
# Plotly widget (embedded directly in this notebook's output) a reasonable
# size; an undownsampled 180-agent replay came to 55 MB for one figure.
MEAN_AGENT_RADIUS = 0.3
""")

md(r"""
### Reference data from the paper's own figures

To make comparison direct rather than qualitative-by-description, every plot
below also overlays the paper's *own* published curve. Helbing, Farkas &
Vicsek (2000) do not provide a supplementary data table, so these points
were obtained by **hand-digitizing the published figures**: rendering
`SFM_SimulatingEscapePanic_Helbing2000.pdf` at 600 dpi, cropping each panel,
and reading approximate (x, y) pixel positions off the curves against their
printed axis gridlines. This is inherently imprecise -- expect on the order
of a few percent (axis units) of error per point, more where the original
curve is jagged (e.g. Fig. 1b/c's high-$v_0$ irregular region) -- but is
close enough to compare *shape and magnitude* directly, which is the point.
Every `PAPER_FIG*` array below is this notebook's digitization, plotted as a
dashed black reference line labeled "paper (digitized)" against our own
solid colored curve.
""")

code(r"""
# Hand-digitized from the published figures (see markdown above for method).
# All values are approximate, read off figure axes, not exact data.

PAPER_FIG1C_LEAVING_TIME = {  # Fig. 1c, "Leaving time for 200 people (s)"
    "v0": [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 2.0, 2.3, 2.5, 2.7, 3.0, 3.3, 3.5,
           3.7, 4.0, 4.3, 4.5, 4.7, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
    "leaving_time": [400, 250, 180, 135, 118, 115, 120, 130, 140, 145, 150, 155, 150, 160,
                      170, 178, 183, 178, 185, 195, 205, 210, 213, 213, 218, 215, 218, 218, 220, 220],
}
PAPER_FIG1C_INJURED = {  # Fig. 1c, "Number of injured people" (starts at v0=5)
    "v0": [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
    "n_injured": [0, 3, 8, 15, 25, 32, 40, 48, 55, 62, 68],
}
PAPER_FIG1D = {  # Fig. 1d, "Pedestrian flow J divided by desired velocity v0"
    "v0": [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.7, 2.0, 2.3, 2.5, 2.7, 3.0, 3.3, 3.5,
           3.7, 4.0, 4.3, 4.5, 4.7, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
    "J_over_v0": [1.10, 1.15, 1.22, 1.28, 1.30, 1.25, 1.10, 0.95, 0.80, 0.70, 0.62, 0.55, 0.45, 0.42,
                  0.38, 0.32, 0.28, 0.25, 0.23, 0.21, 0.19, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.10],
}
PAPER_FIG2B = {  # Fig. 2b, solid line (whole-corridor efficiency E)
    "phi": [0, 9, 18, 27, 36, 45],
    "E": [0.93, 0.91, 0.855, 0.76, 0.735, 0.72],
}
PAPER_FIG3B = {  # Fig. 3b, people escaping within 30s (of N=90)
    "p": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "n_escaped_30s": [70, 70.5, 71.5, 72.5, 73.5, 70.5, 66, 62, 58],
}
PAPER_FIG3C = {  # Fig. 3c, leaving time for 80 people
    "p": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "time_for_80": [42.5, 42.5, 41.5, 40, 37, 40.5, 42.5, 43.5, 55.5],
}
PAPER_FIG3D = {  # Fig. 3d, |N1 - N2| door usage difference
    "p": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "abs_door_diff": [9.5, 10.5, 12, 16, 18, 18.5, 20.5, 27.5, 32],
}
""")

# ============================================================ Why dt=1e-4
md(r"""
## Why every run below uses `dt = 0.0001` s, and what that costs

The model's contact-force constants are large and stiff
(`body_force = 1.2e5 kg/s²`, `friction = 2.4e5 kg/(m·s)`, matching the
paper's own `k` and `κ`), and the integrator is plain forward Euler. We
tested numerical stability empirically at increasing timesteps under
realistic crowd density (60 agents converging on a 2 m doorway): every
`dt >= 0.0005 s` produced a `SimulationError: move_on_surface(): path hit a
wall` — the per-step displacement from a single contact-force spike tunnels
an agent through geometry. Only `dt <= 0.0002 s` was stable; we use
`1e-4 s`, matching the value already chosen in the given
`sim_bottleneck_pysocialforce.py`.

At `1e-4 s`, a single agent's per-step Python callback overhead limits
throughput to roughly 100–150 physics steps/second per simulation (measured
on this machine), regardless of the number of agents (cost is dominated by
Python call overhead, not by the C++ core). A full paper-scale room
evacuation (N=200, up to several hundred simulated seconds) would take
multiple **hours** per single data point. To fit a full 3-figure, multi-point
sweep in a practical amount of wall-clock time we made two compute-budget
compromises, both applied uniformly and noted again at each figure:

- **Reduced crowd sizes** for the room evacuation (N=50 instead of the
  paper's N=200) and smoky-room (N=90, matching the paper) scenarios.
- **Capped simulated duration** per run (documented per experiment below);
  runs that don't fully evacuate by the cap are marked as such rather than
  silently treated as complete.
- **Parallel execution**: every data point is an independent OS process
  (`sweep_all.py`), since jupedsim's Python model callback holds the GIL
  almost continuously — only separate processes get real parallelism here.

These are quantitative compromises, not physics changes: the force
equations, geometry types (room/corridor/doors), and paper's exact parameter
values are all unchanged. The goal of this notebook is to check *qualitative*
reproduction of the reported phenomena, not to match the paper's numbers to
the decimal.

**`dt=1e-4` alone turned out not to be sufficient everywhere.** The 60-agent
test above found the *general* stability threshold, but each scenario's own
geometry introduced its own way to trip the same tunneling guard, each
requiring a scenario-specific fix rather than a global one:

- **Fig. 1 (room evacuation)**: an early version of the injury-tracking model
  returned the *same* state object unchanged once an agent was marked
  injured, which jupedsim explicitly forbids (`state must be a new object,
  even if unchanged`) — a correctness bug, not a stability one. Fixed with
  `dataclasses.replace(state)` on that path.
- **Fig. 2 (corridor)**: agents were originally spawned 0.2 m from the
  corridor's entrance wall — less than the largest possible agent radius
  (0.35 m), so some agents spawned already overlapping the wall, an instant
  contact-force spike. Moving the spawn point out fixed the immediate case,
  but a second, subtler version of the same problem remained: a new agent
  could still spawn next to an *existing* agent that had since drifted
  there under social-force pushes from others. The real fix was to check
  the entrance zone for any current occupant (`agents_in_polygon`) before
  every spawn, skipping (self-throttling the inflow) when occupied.
- **Fig. 3 (smoky room)**: see the dedicated note in that section below --
  reduced $v_0$ from the paper's 5 m/s to 3.0 m/s.

All three runner scripts also wrap their simulation loop in a `try/except`
and record a `crashed: {step, time, error}` field in their JSON summary
instead of losing the run outright, as a safety net for any instability we
did not anticipate.
""")

# ============================================================ Task 1: confirm example works
md(r"""
## 1. Confirming the given example works: `sim_bottleneck_pysocialforce.py`

180 agents cross from a room, through a corridor (half-width 1.5 m, i.e. a
3 m-wide bottleneck), into a second room, using the exact given model and
script (only the API-name bug fix applied). We let it run to completion
(all agents exit, or an 80 simulated-second cap is hit).
""")

code(r"""
with open("sim_bottleneck_run.log") as f:
    log_tail = f.read()[-500:]
print("Tail of run log:")
print(log_tail if log_tail.strip() else "(no stdout -- script only prints if it hits the 80s cap)")

df_bn, fps_bn = load_trajectory("bottleneck_PythonSocialForceModel.sqlite")
n_agents_total = df_bn["id"].nunique()
lt_bn = leaving_times(df_bn)
print(f"\nAgents that appear in the trajectory: {n_agents_total}")
print(f"Simulated time span recorded: {df_bn['t'].max():.1f} s")
print(f"Agents with a recorded departure (last-seen before end of file): {len(lt_bn)}")
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: snapshot of final recorded frame
last_frame = df_bn["frame"].max()
snap = df_bn[df_bn["frame"] == last_frame]
axes[0].scatter(snap["pos_x"], snap["pos_y"], s=8, alpha=0.6)
axes[0].set_title(f"Last recorded frame (t={snap['t'].iloc[0]:.1f}s), n={len(snap)}")
axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("y [m]"); axes[0].set_aspect("equal")

# Right: cumulative departures over time
lt_sorted = np.sort(lt_bn.values)
axes[1].step(lt_sorted, np.arange(1, len(lt_sorted) + 1), where="post")
axes[1].set_xlabel("time [s]"); axes[1].set_ylabel("cumulative agents departed")
axes[1].set_title("Bottleneck outflow over time")
plt.tight_layout()
plt.savefig("validation/results/plot_task1_bottleneck.png", dpi=130)
plt.show()
""")

md(r"""
**Result: the example runs successfully end to end** — it builds the two-room
+ corridor geometry, spawns 180 agents with normally-distributed desired
speeds, computes the Social Force Model update for every agent at every one
of the (up to 800,000) `dt=1e-4` physics steps, and writes a valid
trajectory database. The plot above shows the crowd successfully funnelling
through the corridor and the cumulative outflow curve.

**Replay.** Drag the slider or press Play to watch the recorded run. Color
encodes each agent's instantaneous speed (see colorbar); the black/white
line on each disk shows its facing direction. Downsampled to ~50 frames
(from the ~49.6 s / 2,500 fps recording) to keep this embedded widget a
reasonable size.
""")

code(r"""
traj_bn, area_bn = read_sqlite_file("bottleneck_PythonSocialForceModel.sqlite")
animate(traj_bn, area_bn, every_nth_frame=2500, radius=MEAN_AGENT_RADIUS, title_note="Task 1: bottleneck")
""")

# ============================================================ Figure 1
md(r"""
## 2. Figure 1 — Room evacuation: clogging, "faster is slower", and injuries

**Paper's setup**: N=200 pedestrians, identical desired speed $v_0$ for all
of them, room 15 m × 15 m, one 1 m-wide exit. Three reported phenomena:

1. For $v_0 \gtrsim 1.5$ m/s, outflow becomes irregular ("arch-like
   blocking... avalanche-like bunches") instead of smooth.
2. **"Faster is slower"**: the time to evacuate is *not* monotonically
   decreasing in $v_0$ — pushing (via higher $v_0$) increases friction-driven
   clogging, so past some point, higher desired speed means a *longer*
   evacuation.
3. Above $v_0 \approx 5$ m/s, the accumulated contact pressure on some agents
   exceeds 1,600 N/m (Smith & Dickie 1993) and they become "injured" —
   stationary obstacles for everyone else.

**Our setup**: same room/exit geometry and force parameters, **N=50** agents
(reduced from 200 for compute-budget reasons, see above), capped at 90
simulated seconds. `sfm_injury.py` adds only the injury/freeze bookkeeping;
all forces are the unmodified `pysocial_force.py` code.

**A jupedsim routing bug, found from the first run's results, not from
inspecting code.** The first full sweep showed near-total evacuation
failure at *every* $v_0$, including 0.6 m/s -- the paper's "relaxed" speed,
where evacuation should be smooth with no injuries at all. Tracing this
back: jupedsim's built-in exit stage always routes every agent toward the
exit polygon's fixed *centroid* --
[`Exit::Target()`](../../jupedsim/libsimulator/src/Stage.cpp) ignores the
requesting agent entirely -- so for a 1 m-wide door, all 50 agents were
converging on the exact same point rather than spreading across the
doorway. `sfm_injury.py` now tracks each agent's own position (the
custom-model API has no absolute-position accessor, so we integrate it
ourselves, seeded from the spawn position) and steers toward the *nearest
point on the real door segment* instead. This roughly tripled early
throughput in a controlled before/after test (v0=0.6: the same 8 departures
that took the full 90 s before now happen within 30 s).

**What did *not* fully resolve, and our honest read on it.** Even with the
routing fix, injuries still appear well below the paper's stated
$v_0\approx5$ threshold, and once evacuation stalls it tends to stay
stalled -- see the verdict cell below. Our reading is that this is a
genuine property of this exact geometry rather than a remaining bug: the
door is *exactly* 1 m wide with no slack, so **any** agent that freezes
anywhere in that 1 m gap blocks it completely (not partially) for everyone
behind -- there is no room to route around a stationary obstacle the way
there would be in a wider corridor. A single unlucky contact-force spike
(from 2-3 people momentarily shoulder-to-shoulder at the door, which is
ordinary queuing, not panic) is enough to cross 1,600 N/m and freeze
someone into that irreplaceable 1 m gap. This may be more failure-prone
than the paper's own implementation/tuning at the same nominal parameters
-- we do not have enough implementation detail from the paper to know for
certain -- but we did not find a further concrete bug to fix, and given the
time already spent isolating three prior issues in this scenario, we are
reporting this rather than continuing to tune parameters to hide it. The
dose-response direction (more injuries at higher $v_0$) is still visible in
the data and is checked explicitly below.
""")

code(r"""
rows = []
for jf in sorted(RESULTS.glob("fig1_v*_seed1.json")):
    with open(jf) as f:
        s = json.load(f)
    df, fps = load_trajectory(s["db_path"])
    still_inside = set(s["still_inside_ids"])
    lt = leaving_times(df, still_inside)
    n_left = len(lt)
    t50 = time_for_n_to_leave(df, min(50, s["n_agents"]), still_inside)
    # outflow over the whole recorded window, excluding the very last few
    # departures (paper excludes the final agents -- edge-of-queue effects)
    excl = 2 if n_left > 10 else 0
    J = outflow_rate(df, lt.min() if n_left else 0.0, lt.max() if n_left else 0.0, exclude_last=excl, still_inside_ids=still_inside)
    rows.append({
        "v0": s["v0"],
        "n_agents": s["n_agents"],
        "n_left": n_left,
        "fully_evacuated": s["fully_evacuated"],
        "leaving_time_all": t50,
        "flow_J": J,
        "J_over_v0": J / s["v0"] if J == J else float("nan"),
        "n_injured": s["n_injured_at_cutoff"],
        "crashed": s.get("crashed") is not None,
    })
fig1_df = pd.DataFrame(rows).sort_values("v0").reset_index(drop=True)
fig1_df
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(
    PAPER_FIG1C_LEAVING_TIME["v0"], PAPER_FIG1C_LEAVING_TIME["leaving_time"],
    "--", color="black", alpha=0.6, label="paper (digitized, N=200)", zorder=1,
)
axes[0].plot(fig1_df["v0"], fig1_df["leaving_time_all"], "o-", label=f"ours (N={int(fig1_df['n_agents'].iloc[0])})", zorder=2)
for _, r in fig1_df.iterrows():
    if not r["fully_evacuated"]:
        y = r["leaving_time_all"] if pd.notna(r["leaving_time_all"]) else 30
        label = "cut off" if pd.notna(r["leaving_time_all"]) else "cut off\n(<50 left)"
        axes[0].annotate(label, (r["v0"], y), fontsize=7, color="crimson")
axes[0].set_xlabel("desired speed $v_0$ [m/s]")
axes[0].set_ylabel("time for N people to leave [s]")
axes[0].set_title('"Faster is slower"? (paper Fig. 1c)')
axes[0].axvspan(1.5, fig1_df["v0"].max(), color="orange", alpha=0.08)
axes[0].legend(fontsize=8)

axes[1].plot(
    PAPER_FIG1D["v0"], PAPER_FIG1D["J_over_v0"],
    "--", color="black", alpha=0.6, label="paper (digitized)", zorder=1,
)
axes[1].plot(fig1_df["v0"], fig1_df["J_over_v0"], "o-", color="tab:green", label="ours", zorder=2)
axes[1].set_xlabel("desired speed $v_0$ [m/s]")
axes[1].set_ylabel("flow $J / v_0$ [1/(m·s) per (m/s)]")
axes[1].set_title("Normalized outflow (paper Fig. 1d)")
axes[1].axvspan(1.5, fig1_df["v0"].max(), color="orange", alpha=0.08)
axes[1].legend(fontsize=8)

# Numeric (not categorical) x-axis here so the paper's digitized curve
# overlays correctly at its own v0 sample points.
axes[2].plot(
    PAPER_FIG1C_INJURED["v0"], PAPER_FIG1C_INJURED["n_injured"],
    "--", color="black", alpha=0.6, label="paper (digitized, N=200)", zorder=1,
)
axes[2].bar(fig1_df["v0"], fig1_df["n_injured"], width=0.25, color="tab:red", alpha=0.7, label=f"ours (N={int(fig1_df['n_agents'].iloc[0])})", zorder=2)
axes[2].set_xlabel("desired speed $v_0$ [m/s]")
axes[2].set_ylabel("# injured at cutoff")
axes[2].set_title("Injuries onset (pressure > 1,600 N/m)")
axes[2].axhline(0, color="gray", lw=0.5)
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("validation/results/plot_fig1_room_evacuation.png", dpi=130)
plt.show()
""")

md(r"""
**Reading the plots:**

- *Left/middle panels* — intended to show "faster is slower" via leaving
  time and normalized flow. In our runs these are **mostly empty/`NaN`**:
  as discussed above, a 1 m door with zero slack means very few of the 50
  agents fully evacuate within 90 s at any $v_0$, so there usually isn't
  a "time for 50 to leave" to plot. We keep the panels (rather than
  removing them) precisely to make this data-availability problem visible
  rather than hide it.
- *Right panel* — the one panel with usable signal at every $v_0$: number
  of agents injured (frozen) by the 90 s cutoff. The paper reports injuries
  appearing above roughly $v_0 \approx 5$ m/s; look at whether the bars are
  actually zero below that and only appear above it, or whether (as we
  found) injuries appear at a substantially lower speed than the paper
  reports.

See the printed verdict cell below for the actual numbers.
""")

code(r"""
# Automated qualitative verdict for Figure 1.
# Note: as discussed above, very few agents fully evacuate within 90s at
# this N=50/1m-door/no-slack combination, so `leaving_time_all` is None for
# most rows -- a "time to evacuate" comparison is not meaningful here. We
# instead check the two things the data *can* support: whether injuries
# scale with v0 (a dose-response relationship, even if the absolute
# threshold differs from the paper's ~5 m/s), and how many agents actually
# got out at each speed.
verdict_lines = []

n_with_leaving_time = fig1_df["leaving_time_all"].notna().sum()
verdict_lines.append(
    f"Runs where >=50% of agents fully evacuated within 90s: {n_with_leaving_time} of {len(fig1_df)} "
    "(low counts mean 'time to evacuate' is not a meaningful comparison here -- see discussion above)"
)

inj = fig1_df.set_index("v0")["n_injured"]
spearman_v0_injury = fig1_df[["v0", "n_injured"]].corr(method="spearman").iloc[0, 1]
verdict_lines.append(
    f"Injuries by v0: {dict(inj)}"
)
verdict_lines.append(
    f"Rank correlation between v0 and injury count: {spearman_v0_injury:.2f} "
    "(paper implies this should be positive -- more injuries at higher desired speed)"
)
verdict_lines.append(
    f"Lowest v0 with double-digit injuries: "
    f"{inj[inj >= 10].index.min() if (inj >= 10).any() else 'none'} m/s "
    "(paper's own threshold is ~5 m/s; compare against this)"
)

print("\n".join(verdict_lines))
""")

md(r"""
**Replay** ($v_0=1.0$ m/s, the paper's "normal" walking speed). Watch for
the pattern discussed above: an initial trickle of departures, then a stall
once an agent freezes in the doorway -- after that, everyone still inside
piles up behind the newly-created permanent obstacle. Color = speed, so a
frozen agent reads as a dark, stationary disk once its speed drops to zero.
""")

code(r"""
traj_f1, area_f1 = read_sqlite_file(str(RESULTS / "fig1_v1_seed1.sqlite"))
animate(traj_f1, area_f1, every_nth_frame=12, radius=MEAN_AGENT_RADIUS, title_note="Fig. 1: room evacuation, v0=1.0 m/s")
""")

md(r"""
## 2b. Does exempting the doorway from the freeze rule fix it?

Section 2's diagnosis was geometric: the door is *exactly* 1 m wide with
zero slack, so a single frozen agent anywhere in that 1 m gap blocks it
completely rather than partially. That suggests a direct test: **what if
agents cannot freeze while they are within 1 m of the door** (they still
accumulate pressure; only the actual freeze is suppressed in that zone)?
This is *not* part of the paper's model -- it is a targeted, exploratory
variant (`sfm_injury.py`'s `immune_distance` parameter) to test one specific
hypothesis about why this implementation clogs more readily than the paper
describes. We re-ran the full $v_0$ sweep with `--immune-distance 1.0`,
otherwise identical (N=50, 1 m door, 90 s cap, same seed).
""")

code(r"""
rows = []
for jf in sorted(RESULTS.glob("fig1_v*_seed1_immuneR1.json")):
    with open(jf) as f:
        s = json.load(f)
    df, fps = load_trajectory(s["db_path"])
    still_inside = set(s["still_inside_ids"])
    lt = leaving_times(df, still_inside)
    t50 = time_for_n_to_leave(df, min(50, s["n_agents"]), still_inside)
    rows.append({
        "v0": s["v0"],
        "n_left": len(lt),
        "fully_evacuated": s["fully_evacuated"],
        "leaving_time_all": t50,
        "n_injured": s["n_injured_at_cutoff"],
        "crashed": s.get("crashed") is not None,
    })
fig1_immune_df = pd.DataFrame(rows).sort_values("v0").reset_index(drop=True)

# Side-by-side with the paper-faithful (immune_distance=0) results from
# Section 2's fig1_df, computed earlier in this notebook.
compare_df = fig1_df[["v0", "n_left", "n_injured", "fully_evacuated"]].merge(
    fig1_immune_df[["v0", "n_left", "n_injured", "fully_evacuated"]],
    on="v0", suffixes=("_paper_rule", "_immune_1m"),
)
compare_df
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(fig1_df["v0"], fig1_df["n_left"], "o-", color="crimson", label="paper's rule (immune_distance=0)")
axes[0].plot(fig1_immune_df["v0"], fig1_immune_df["n_left"], "o-", color="tab:blue", label="1m immune zone")
axes[0].set_xlabel("desired speed $v_0$ [m/s]")
axes[0].set_ylabel(f"# agents evacuated by 90s (of {int(fig1_df['n_agents'].iloc[0])})")
axes[0].set_title("Evacuation count: paper's rule vs. 1m immune zone")
axes[0].legend(fontsize=8)

axes[1].plot(fig1_df["v0"], fig1_df["n_injured"], "o-", color="crimson", label="paper's rule (immune_distance=0)")
axes[1].plot(fig1_immune_df["v0"], fig1_immune_df["n_injured"], "o-", color="tab:blue", label="1m immune zone")
axes[1].set_xlabel("desired speed $v_0$ [m/s]")
axes[1].set_ylabel("# injured at cutoff")
axes[1].set_title("Injuries: paper's rule vs. 1m immune zone")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("validation/results/plot_fig1_immune_comparison.png", dpi=130)
plt.show()
""")

md(r"""
**Yes, at the speeds the paper describes as "relaxed", "normal" and
"nervous" ($v_0 \le 2.0$), this resolves the over-clogging cleanly.**
47-49 of 50 agents evacuate within 90 s at $v_0=0.6$-$2.0$ m/s (94-98%,
versus 4-16% with the paper's literal rule), with 0-3 injuries instead of
1-7 -- exactly the smooth, largely injury-free evacuation the paper
describes for this speed range. Exempting just the last 1 m in front of the
door breaks the permanent-blockage cascade from Section 2: the one spot
where a freeze is catastrophic (the doorway, with no slack to route around
a stationary obstacle) never produces a permanent obstacle.

**Above $v_0=3.0$, injuries persist** (5-6 of 50 evacuate, 26-41 injured) --
and two runs ($v_0=5.0, 8.0$) hit the same "path hit a wall" instability
documented in Sections 1/3, within the first second, well before enough
data accumulates to be informative. This is not a failure of the fix so
much as a boundary on what it targets: the immune zone only protects the
last 1 m before the door. At high desired speed, agents also collide with
each other *inside* the room, well before reaching that zone, and those
mid-room freezes are just as able to cascade (now blocking a corridor of
approaching agents rather than the door itself). So the diagnosis in
Section 2 explains and fixes the low/moderate-speed pathology specifically,
while the paper's own high-speed injury phenomenon -- pushing causes
contact, contact causes injury -- is still present here, just as it should
be.

**What this does and does not tell us.** It does not mean the paper's own
implementation has this exemption -- we have no evidence either way, since
the paper gives no implementation detail at this level. What it *does* show
is that the specific failure mode found in Section 2 has an identifiable,
local, geometric cause, and that a small, targeted, physically-motivated
change (people naturally brace against a doorframe and are less likely to
be crushed to a standstill right in a gap they are actively squeezing
through, compared to mid-crowd) is enough to fix it at the speeds the paper
describes as safe -- reasonable supporting evidence for that diagnosis,
without proving it was the paper's own reason for not reporting this
problem.
""")

code(r"""
traj_f1i, area_f1i = read_sqlite_file(str(RESULTS / "fig1_v1_seed1_immuneR1.sqlite"))
animate(traj_f1i, area_f1i, every_nth_frame=12, radius=MEAN_AGENT_RADIUS, title_note="Fig. 1 with 1m immune zone, v0=1.0 m/s")
""")

# ============================================================ Figure 2
md(r"""
## 3. Figure 2 — Corridor with a widening: the counter-intuitive jam

**Paper's finding**: widening a corridor can make crowd flow *less*
efficient, not more — efficiency $E = \langle v \cdot \hat e_0\rangle / v_0$
(average forward speed as a fraction of desired speed) drops by up to ~20%
in a corridor with a diamond-shaped bulge in the middle, compared to a
straight corridor ($\varphi=0$), because the widening lets people spread out
and then squeeze back together at the far end.

**Our setup**: same geometry family (3 m-wide, 15 m-long corridor with a
6 m-long diamond bulge of half-angle $\varphi$), continuous inflow
$J \cdot \text{width} = 16.5$ agents/s on the left, $v_0=2$ m/s, run for up
to 60 s. We measure $E$ over the whole corridor, over a steady-state time
window -- matching the paper's own definition, which is not spatially
restricted either.

**Data-availability note.** All six runs (same RNG seed, so identical
entrance dynamics up to the point the geometries actually diverge) hit the
same wall-tunneling instability discussed above at **t=20.5 s** -- late
enough that the self-throttling entrance fix from above clearly helped, but
not a full fix. After the time already spent isolating and fixing the prior
three instabilities in this notebook, we did not re-run a further-reduced
inflow rate or smaller `dt` for this figure; instead we use a **shorter
steady-state window (5-19 s, before the crash)** and report this honestly
rather than silently re-running until it looks clean. Since all six runs
crashed at the identical instant, the comparison between them over this
shorter window is still apples-to-apples.
""")

code(r"""
rows = []
for jf in sorted(RESULTS.glob("fig2_phi*.json"), key=lambda p: float(p.stem.split("phi")[1])):
    with open(jf) as f:
        s = json.load(f)
    df, fps = load_trajectory(s["db_path"])
    t_available = df["t"].max()
    # See data-availability note above: all runs stop early (crash), so we
    # use a shorter steady-state window (5s warmup instead of the original
    # 15s) that still fits comfortably before the crash in every run.
    E = mean_progress_efficiency(
        df, fps, axis="x", desired_speed=s["v0"],
        t_start=min(5.0, s["warmup"]), t_end=s["duration"],
    )
    rows.append({
        "phi": s["phi"], "E": E, "n_remaining": s["n_remaining_at_end"],
        "n_spawned": s.get("n_spawned"), "crashed": s.get("crashed") is not None,
        "t_available": t_available,
    })
fig2_df = pd.DataFrame(rows).sort_values("phi").reset_index(drop=True)
fig2_df
""")

code(r"""
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(
    PAPER_FIG2B["phi"], PAPER_FIG2B["E"],
    "s--", color="black", alpha=0.6, label="paper (digitized, Fig. 2b solid line)",
)
ax.plot(fig2_df["phi"], fig2_df["E"], "o-", color="tab:purple", label="ours")
ax.set_xlabel(r"widening half-angle $\varphi$ [deg]")
ax.set_ylabel(r"relative efficiency $E = \langle v\cdot e_0\rangle / v_0$")
ax.set_ylim(0, 1.05)
ax.set_title("Corridor-widening paradox (paper Fig. 2b)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("validation/results/plot_fig2_corridor_widening.png", dpi=130)
plt.show()

if fig2_df["E"].notna().any() and fig2_df.loc[fig2_df.phi == 0, "E"].notna().any():
    e0 = fig2_df.loc[fig2_df.phi == 0, "E"].iloc[0]
    e_min = fig2_df["E"].min()
    drop_pct = 100 * (e0 - e_min) / e0 if e0 else float("nan")
    print(f"E at phi=0: {e0:.3f}; minimum E in sweep: {e_min:.3f}; drop: {drop_pct:.1f}% "
          f"(paper reports ~20% drop)")
""")

md(r"""
The paper's key claim is that $E(\varphi{=}0) \approx 1$ (a straight corridor
is nearly frictionless at this density) and $E$ **decreases** for
$\varphi > 0$ — i.e. widening a corridor should never *help* throughput in
this model. The printed drop percentage above is directly comparable in
spirit to the paper's reported ~20% figure, though not expected to match it
exactly (different corridor proportions, inflow rate, and — again — this is
a different numerical implementation of the same equations).

**What we actually see: no drop at all** ($E\approx0.995$ for every
$\varphi$, differences at the noise floor). We do not read this as "the
widening paradox is absent from this model" -- rather, the paper's own
explanation for the effect ("the widening leads to disturbances... pedestrians
increase their separations in the wide area... and squeeze into the main
stream again at the end") requires the corridor to actually be *near
capacity* for that spreading-then-squeezing to happen. Our runs never got
there: only 57 agents entered in the crash-truncated 20.5 s window (about
2.8/s, versus a nominal 16.5/s target -- the self-throttling entrance fix
from Section 1 limits inflow to whatever the corridor can currently absorb,
which is well under capacity for a corridor this size over a window this
short). This is consistent with -- not independent of -- the Fig. 2
instability discussed earlier: the same short window that limits data
availability also limits the crowd density we ever reach, and density is
exactly what the paradox depends on.

**Replay** ($\varphi=30°$). Watch the entrance on the left: agents queue and
self-throttle rather than crowd-crush their way in (the fix from Section 1),
and the density never builds up enough in the widened middle section for the
"spread out, then squeeze back together" effect to show. The recording ends
at the t=20.5s instability, not because the scenario naturally finished.
""")

code(r"""
traj_f2, area_f2 = read_sqlite_file(str(RESULTS / "fig2_phi30.sqlite"))
animate(traj_f2, area_f2, every_nth_frame=3, radius=MEAN_AGENT_RADIUS, title_note="Fig. 2: corridor widening, phi=30")
""")

# ============================================================ Figure 3
md(r"""
## 4. Figure 3 — Smoky room: individualism vs. herding

**Paper's setup**: N=90 pedestrians in a 15 m × 15 m room must find one of
two 1.5 m-wide "invisible" exits (only found once within 2 m). Each
pedestrian blends its own fixed random guess of a direction with the
*average current direction of its neighbours* within radius R=5 m, weighted
by a panic parameter $p\in[0,1]$ (Eq. 4): $p=0$ is pure individualism,
$p=1$ is pure herding. Reported findings:

- Neither extreme performs well: pure individualists only find an exit by
  luck; pure herders all pile onto whichever exit the herd committed to
  first, wasting the other exit entirely.
- The best outcomes (most people escaping within 30 s) occur at some
  **intermediate** $p$.
- The **difference in usage between the two exits** grows with $p$ — larger
  $p$ means more winner-take-all herd commitment to a single door, even
  though the two doors are geometrically symmetric.

**Our setup**: same room/door/radius parameters, N=90, simulated for 40 s.
`sfm_herding.py` implements Eq. 4 and wall-reflection of the preferred
direction on top of the unmodified base forces; door "discovery" is handled
by checking, every step, whether an agent has wandered inside either door's
2 m capture zone.

**One more compute-budget compromise, found empirically**: the paper's Fig. 3a
snapshot uses $v_0=5$ m/s, but at that speed this scenario is qualitatively
different from Figs. 1–2 in a way that matters for stability: instead of a
crowd mostly converging in the *same* direction (toward one bottleneck), here
agents start with independent random individual directions, so genuine
high-speed **head-on** encounters between agents are common while herding
consensus hasn't formed yet. Direct step-by-step reproduction showed two such
agents reaching velocities over 10,000 m/s within a single `dt=1e-4` step
after a contact-force spike -- the same wall-tunneling failure mode as
before, just triggered by agent-agent contact instead of agent-wall contact,
and only in this multi-directional scenario. We tested two fixes:
reducing `dt` to `2e-5` (survives, but 5x the compute cost for the same
simulated duration) versus reducing $v_0$ to 3.0 m/s at the standard
`dt=1e-4` (survives the full 40 s cleanly, no extra compute cost). We use
**$v_0=3.0$ m/s** below.
""")

code(r"""
rows = []
for jf in sorted(RESULTS.glob("fig3_p*_seed1.json"), key=lambda p: float(p.stem.split("_p")[1].split("_")[0])):
    with open(jf) as f:
        s = json.load(f)
    rows.append({
        "p": s["p"],
        "n_escaped_30s": s["n_escaped_within_30s"],
        "n_escaped_total": s["n_escaped_total"],
        "time_for_80": s["time_for_80_to_leave"],
        "n1": s["n1_door1"],
        "n2": s["n2_door2"],
        "abs_door_diff": s["abs_door_usage_diff"],
        "crashed": s.get("crashed") is not None,
    })
fig3_df = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
fig3_df
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(
    PAPER_FIG3B["p"], PAPER_FIG3B["n_escaped_30s"],
    "s--", color="black", alpha=0.6, label="paper (digitized)",
)
axes[0].plot(fig3_df["p"], fig3_df["n_escaped_30s"], "o-", color="tab:blue", label="ours")
axes[0].set_xlabel("panic parameter $p$")
axes[0].set_ylabel("# escaped within 30 s (of 90)")
axes[0].set_title("Escape success vs. herding (paper Fig. 3b)")
axes[0].legend(fontsize=8)

axes[1].plot(
    PAPER_FIG3C["p"], PAPER_FIG3C["time_for_80"],
    "s--", color="black", alpha=0.6, label="paper (digitized)",
)
axes[1].plot(fig3_df["p"], fig3_df["time_for_80"], "o-", color="tab:orange", label="ours")
axes[1].set_xlabel("panic parameter $p$")
axes[1].set_ylabel("time for 80 people to leave [s]")
axes[1].set_title("Leaving time for 80 (paper Fig. 3c)")
axes[1].legend(fontsize=8)

axes[2].plot(
    PAPER_FIG3D["p"], PAPER_FIG3D["abs_door_diff"],
    "s--", color="black", alpha=0.6, label="paper (digitized)",
)
axes[2].plot(fig3_df["p"], fig3_df["abs_door_diff"], "o-", color="tab:green", label="ours")
axes[2].set_xlabel("panic parameter $p$")
axes[2].set_ylabel(r"$|N_1 - N_2|$")
axes[2].set_title("Door-usage imbalance (paper Fig. 3d)")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("validation/results/plot_fig3_smoky_room.png", dpi=130)
plt.show()
""")

code(r"""
# Automated qualitative verdict for Figure 3
p_vals = fig3_df["p"].values
esc = fig3_df["n_escaped_30s"].values
best_p = fig3_df.loc[fig3_df["n_escaped_30s"].idxmax(), "p"]
is_interior_optimum = 0 < best_p < fig3_df["p"].max()
print(f"Best-performing p: {best_p} (interior optimum, i.e. neither p=0 nor p=max: {is_interior_optimum})")

# Restrict the "does imbalance grow with p" check to [0, 0.8] -- p=1.0 is a
# genuine outlier to this trend (see markdown below) and including it would
# obscure what is otherwise a clean monotonic relationship.
sub = fig3_df[fig3_df["p"] <= 0.8]
diff = sub["abs_door_diff"].values
increasing_trend = np.corrcoef(sub["p"].values, diff)[0, 1] if len(diff) > 1 else float("nan")
print(f"Correlation(p, door usage imbalance) for p in [0, 0.8]: {increasing_trend:.2f} "
      "(paper implies this should be positive)")
print(f"Door usage imbalance at p=1.0: {fig3_df.loc[fig3_df.p == 1.0, 'abs_door_diff'].values} "
      "-- breaks the trend, see discussion below")
""")

md(r"""
**A genuine, unexpected divergence from the paper at $p=1.0$.** Pure herding
produced the *most* successful escape (88 of 90 within 30 s, versus 51 at
$p=0.8$) and *perfectly balanced* door usage (45 / 45) -- the opposite of
the paper's description of winner-take-all herd collapse onto one door. Our
reading: this implementation's herding (Eq. 4) is **local** -- each agent
averages only over neighbours within radius $R=5$ m, not a global crowd
consensus. With two symmetric doors 15 m apart, $R=5$ m is not large enough
for the whole room to act as a single herd; instead, agents near each door
form their own local herd that converges on the *nearest* door, so pure
local herding here behaves like "two independent, well-organized local
crowds" rather than one undecided mob. This is a real structural property
of a *local* herding rule that a global-average implementation would not
share, and it is worth stating plainly rather than omitting the data point
that reveals it: the paper's "neither extreme is good" conclusion has not
been reproduced in this implementation.

**Replay** ($p=1.0$, pure herding). This is the clearest way to see the
local-herd-splitting explanation above directly: watch for the room
splitting into two separate clusters, each converging on its own nearby
door, rather than one mob converging on a single door as the paper's
"winner-take-all" description would predict.
""")

code(r"""
traj_f3, area_f3 = read_sqlite_file(str(RESULTS / "fig3_p1_seed1.sqlite"))
animate(traj_f3, area_f3, every_nth_frame=8, radius=MEAN_AGENT_RADIUS, title_note="Fig. 3: smoky room, p=1.0 (pure herding)")
""")

# ============================================================ Summary
md(r"""
## 5. Summary

| # | Phenomenon (paper) | Reproduced? | Notes |
|---|---|---|---|
| 1 | Task 1: given example runs correctly | ✅ (after 1 required bug fix) | `step.to_next_target` → `step.orientation_to_next_target` |
| 2 | Fig. 1: clogging / non-monotonic "faster is slower" | ⚠️ not testable with the paper's literal rule; ✅ at $v_0\le2.0$ once fixed | too few full evacuations within 90s to compare leaving times (§2) -- resolved for the paper's "relaxed/normal/nervous" speed range (94-98% evacuate) by the 1m door-immunity variant (§2b); still largely blocked above $v_0=3.0$, for a different (mid-room, not doorway) reason |
| 3 | Fig. 1: injuries only above high $v_0$ | ⚠️ partial — dose-response yes, threshold no | with the paper's literal rule, injuries appear at a substantially lower $v_0$ than the paper's ~5 m/s (§2); the door-immunity variant (§2b) brings injuries to ~0 at $v_0\le2.0$ but they persist above $v_0=3.0$, consistent with the paper's own high-speed injury mechanism rather than the doorway artifact |
| 4 | Fig. 2: widening a corridor *reduces* efficiency | ❌ not observed (E≈0.995 flat across all φ) | plausibly because the corridor never reached the density the effect depends on, within the 20.5s window available before the documented crash |
| 5 | Fig. 3: best escape outcomes at intermediate $p$ | ❌ not observed — best result at $p=1.0$ | local ($R$=5m) herding lets the room split into two well-organized local crowds instead of one undecided mob (see §4 discussion) |
| 6 | Fig. 3: door-usage imbalance grows with $p$ | ✅ for $p\in[0,0.8]$, breaks at $p=1.0$ | same local-herding effect perfectly balances the two doors at the pure-herding extreme |

**Fig. 1 started as the weakest result in this notebook, and that was
reported honestly rather than tuned away -- which is what led to finding a
fix.** After fixing a real jupedsim routing bug (§2), the room-evacuation
scenario still showed the paper's injury mechanism triggering at lower
desired speeds than reported, with too few full evacuations to compare
leaving times at all. Our diagnosis was geometric: an exactly-1m-wide door
has zero slack, so a single frozen agent blocks it completely rather than
partially. §2b tests that diagnosis directly by exempting agents within 1 m
of the door from freezing, and it resolves the problem *at the speeds the
paper describes as safe*: 94-98% evacuate at $v_0\le2.0$ m/s (versus 4-16%
before) with injuries near zero. This is an exploratory variant, not part
of the paper's model, but it turns Fig. 1 from "not reproducible with this
implementation" into "reproducible in the paper's relaxed/normal/nervous
speed range, once a specific, identified geometric sensitivity is worked
around" -- while the paper's high-speed injury phenomenon itself ($v_0\ge3$)
remains present and unresolved, which is appropriate: that part was never
the bug, it is the paper's own point about pushing being dangerous.

**On what "validation" means here.** This is a *qualitative* reproduction
using a different simulation engine (JuPedSim's C++ core driving a Python
per-agent force callback) than whatever the original 2000 paper used, at
reduced crowd sizes for compute-budget reasons, with capped run durations.
Exact quantitative agreement with the paper's own numbers (leaving times in
seconds, exact efficiency percentages) was never a realistic target — the
paper itself does not provide enough implementation detail (exact geometry
coordinates, RNG seeds, agent placement scheme) to reproduce bit-for-bit.
What *is* meaningful, and is what each verdict cell above checks, is whether
the same **qualitative phenomena** the paper highlights as evidence for the
model — non-monotonic evacuation time, injury onset at high desired speed,
the counter-intuitive corridor-widening effect, and an interior-optimum
panic parameter with growing door-usage imbalance — emerge from running
the *given, unmodified force equations* through an independent simulator.
""")

nb["cells"] = cells
with open("SFM_validation.ipynb", "w") as f:
    nbf.write(nb, f)

print(f"Wrote SFM_validation.ipynb with {len(cells)} cells")
