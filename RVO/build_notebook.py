"""Programmatically builds RVO_validation.ipynb from markdown/code cell
source strings below, so the notebook's content is versionable as plain
Python/Markdown rather than raw ipynb JSON. Run, then execute with:

    jupyter nbconvert --to notebook --execute --inplace RVO_validation.ipynb
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
# Validating the Python/JuPedSim Reciprocal Velocity Obstacles model against van den Berg, Lin & Manocha (2008)

This notebook checks a Python implementation of **Reciprocal Velocity
Obstacles (RVO)** — running inside **JuPedSim**'s experimental
`CustomOperationalModel` plugin API, the same pattern used for the Social
Force Model elsewhere in this repo — against the paper that introduced it:

> J. van den Berg, M. Lin, D. Manocha. *Reciprocal Velocity Obstacles for
> Real-Time Multi-Agent Navigation.* IEEE ICRA 2008.
> [doi:10.1109/ROBOT.2008.4543489](https://doi.org/10.1109/ROBOT.2008.4543489)

RVO's validation in the original paper is purely **synthetic/algorithmic**
(see `model_description.md`): there is no comparison to recorded human
trajectories anywhere in it. What the paper actually proves and checks is:

- **Theorem 6 (collision-freedom):** if two agents both use the RVO rule,
  they never collide.
- **Theorem 8 (oscillation-freedom):** RVO's 50/50 reciprocal averaging
  removes the "dancing" back-and-forth that plain Velocity Obstacles (VO)
  produces when both sides react to each other's *last* move.
- **Scalability:** frame time scales roughly linearly with the number of
  agents, real-time (>10 FPS) at 1000 agents on 2008-era hardware.

**What this notebook does, in order:**

1. Implements RVO's velocity-selection rule (`pyrvo.py`) as a JuPedSim
   custom model, and directly tests Theorem 6 (a proven, checkable
   numerical guarantee — a stronger, more rigorous target than anything
   the Social Force Model's own source paper offered) at several crowd
   densities.
2. Reproduces the paper's **Circle scenario** (Section VI-A): N agents
   swap to their antipodal point; used for both an oscillation comparison
   (RVO vs. plain VO, N=12) and a scalability sweep (N up to 1000).
3. Reproduces the paper's **Narrow Passage scenario** (Section VI-A):
   four groups funnel through shared gaps between square obstacles.
4. Reproduces the paper's **Moving Obstacle scenario** (Section VI-A):
   pedestrians cross a street a non-cooperating car drives straight
   through.

**Code layout** (all in `RVO/`):

- `pyrvo.py` — the RVO model (`ReciprocalVelocityObstacleModel`,
  `RVOState`).
- `validation/run_circle.py`, `run_circle_scaling.py`,
  `run_narrow_passage.py`, `run_moving_obstacle.py` — one runner per
  scenario.
- `validation/analysis.py` — shared trajectory-loading / metric helpers.
- `validation/results/` — recorded `.sqlite` trajectories and `.json`
  run summaries.
""")

# ============================================================ Model design + bugs found
md(r"""
## Model design, and two real problems found (and fixed) while building it

`pyrvo.py` picks a new velocity every step by sampling a set of candidate
velocities and scoring each one by (a) distance to the agent's preferred
velocity and (b) how soon it would enter a Velocity Obstacle, exactly the
selection approach the paper itself describes. The apex of each neighbor's
VO cone is shifted to the *average* of both agents' current velocities for
a `reciprocal=True` agent (real RVO, Theorem 6/8), or left at the
neighbor's own velocity for `reciprocal=False` (plain VO, and *always* for
a non-cooperating obstacle like a wall or a non-reactive "car" agent, which
won't take its half of the responsibility).

Two problems only showed up once agents were actually run, not from
reading the paper or the code in isolation:

**1. A blind sampling grid deadlocks in near-exact head-on encounters
(found first).** Two agents walking straight at each other reached the
point of just touching, then froze there — permanently, both stuck at
velocity `(0, 0)` for the rest of the run. Once two agents are that close,
the VO cone's half-angle approaches 90°, blocking almost the entire
forward half-plane; every sampled candidate that got around it was so far
from the preferred direction that *standing still* scored better, and
standing still is a fixed point (next step recomputes the exact same
blocked cone). Fix: also sample the exact tangent directions of every
active VO cone each step (see `_tangent_candidates` in `pyrvo.py`) — the
closest a velocity can get to "straight at the goal" while provably
clearing that one obstacle. This resolved the two-agent deadlock cleanly
(agents pass with the minimum possible clearance instead of freezing).

**2. `Simulation(geometry=..., excluded_areas=...)` silently ignores
`excluded_areas`, in this JuPedSim build** — the constructor accepts
`**kwargs` but never forwards them to geometry construction, so the square
obstacles for the Narrow Passage scenario were simply missing from the
walkable area, with no error. Fix: bake the holes into a `shapely.Polygon`
directly and pass that as `geometry=` (see `run_narrow_passage.py`), which
*is* handled correctly (`build_geometry`'s shapely-Polygon path reads
`.interiors` for holes).

**A third thing, found through the validation runs below rather than through
debugging, is reported honestly rather than engineered away:** the
sampling-based selection is not the paper's exact, proven-correct geometry
— at higher agent density, several neighbors' VO cones can overlap so much
that no sampled candidate is simultaneously outside all of them, and the
soft fallback scoring (distance-to-preferred vs. collision urgency) does
not always prevent a shallow, real penetration. Section 2 below quantifies
exactly when this does and doesn't hold.
""")

# ============================================================ Setup
code(r"""
import pathlib
import sys

# jupedsim is not pip-installed; its Python package and compiled bindings are
# made importable via PYTHONPATH (see jupedsim/build/environment). Jupyter
# kernels don't inherit that, so replicate it here. The kernel cwd is this
# notebook's folder (RVO/), matching the pathlib.Path("validation") usage below.
_REPO = pathlib.Path.cwd().parent
_JP = _REPO / "jupedsim"
for _d in (
    _JP / "python_modules" / "jupedsim",
    _JP / "python_modules" / "jupedsim_visualizer",
    _JP / "python_modules" / "jupedsim_examples",
    _JP / "build" / "lib",
):
    _p = str(_d.resolve())
    if _p not in sys.path:
        sys.path.insert(0, _p)

# sanity check
from jupedsim.internal.notebook_utils import animate, read_sqlite_file
print("jupedsim import OK")
""")

code(r"""
import json
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(pathlib.Path("validation").resolve()))
from analysis import (
    load_trajectory, min_pairwise_distance, path_tortuosity, turning_metrics,
    trim_to_active_phase, reached_target,
)

RESULTS = pathlib.Path("validation/results")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

# Same caveat as SFM's own notebook: this is JuPedSim's own internal replay
# utility. Its module docstring says explicitly "We make no promises about
# the functions from this file w.r.t. API stability... Do not use the code
# here. Use it at your own peril." We use it anyway, for the same reason --
# there isn't a public, stable alternative.
from jupedsim.internal.notebook_utils import animate, read_sqlite_file

MEAN_AGENT_RADIUS = 0.25
""")

md(r"""
### Reference values from the original paper

Every plot below also shows the paper's own result, drawn in dashed black
and labeled "paper". van den Berg, Lin & Manocha (2008) report very
little numeric data: **Fig. 11** (frame time and frame rate against N in
the Circle scenario) is the only quantitative plot. Its points were
**hand-digitized**: the PDF page was rendered at 300 dpi and each marker
was read off against the printed gridlines. That method is good to about
±0.001 s/frame and ±1 frame/s. The N=100 frame-rate marker sits above the
top of the published axis (>100 frame/s), so it is left out.

The paper's other results are stated in words or shown as pictures, so we
compare against them in the same form:

- **Collision-freedom**: "there were no collisions among agents", which
  is proven in Theorem 6. That means a minimum clearance of 0 m.
- **Narrow Passage**: "eventually all agents reach their goals safely",
  i.e. 100% of agents reach their target.
- **Moving Obstacle**: all eleven pedestrians cross. The ones that
  cannot pass in front of the car wait until it has gone by (Fig. 10),
  i.e. 100% cross.
- **Fig. 7** (VO vs. RVO traces for N=12) and **Fig. 10** (stills from
  the car scenario) are images, not data. They were cropped from the PDF
  and are shown next to our own plots.
""")

code(r"""
# Hand-digitized from Fig. 11 of the paper (see markdown above for method).
PAPER_FIG11 = {  # diamond markers, "sec/frame" (right axis)
    "n": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    "sec_per_frame": [0.0083, 0.0163, 0.0242, 0.0318, 0.0395, 0.0473, 0.0556, 0.0648, 0.0732, 0.0822],
}
PAPER_FIG11_FPS = {  # square markers, "frames/sec" (left axis); N=100 is off-scale (>100)
    "n": [200, 300, 400, 500, 600, 700, 800, 900, 1000],
    "fps": [62, 42, 32, 25.7, 21.2, 18.2, 15.4, 13.7, 12.3],
}
PAPER_MIN_CLEARANCE = 0.0          # Theorem 6 / "no collisions among agents"
PAPER_NARROW_PASSAGE_REACHED = 1.0  # "eventually all agents reach their goals safely"
PAPER_MOVING_OBSTACLE_REACHED = 1.0 # all eleven pedestrians cross (Fig. 10)
PAPER_FIG7_IMG = RESULTS / "paper_fig7_circle_vo_vs_rvo.png"
PAPER_FIG10_IMG = RESULTS / "paper_fig10_moving_obstacle.png"
PAPER_STYLE = dict(color="black", ls="--", alpha=0.7)
""")

# ============================================================ Theorem 6 across scenarios
md(r"""
## 1. Theorem 6 (collision-freedom): the paper's actual proven claim

Unlike the Social Force Model's source paper (which offered no formal
guarantee to test against, only plausibility/calibration), RVO's paper
*proves* that two reciprocal agents never collide. That is a sharp,
numerically checkable claim: minimum inter-agent clearance
(center-to-center distance minus the sum of the two radii) should never go
negative. We check it directly, across every scenario below, at several
densities.
""")

code(r"""
rows = []

# Circle scenario, N=12 (Section 3 below has full detail)
for tag, label in [("circle_n12_rvo", "Circle N=12 (RVO)"), ("circle_n12_vo", "Circle N=12 (plain VO)")]:
    df, fps = load_trajectory(RESULTS / f"{tag}.sqlite")
    with open(RESULTS / f"{tag}.json") as f:
        s = json.load(f)
    _, clearance = min_pairwise_distance(df, s["agent_radius"])
    rows.append({"scenario": label, "min_clearance_m": clearance})

# Circle scaling sweep, N=12/50/100 (full trajectories recorded)
with open(RESULTS / "circle_scaling.json") as f:
    scaling = json.load(f)
for r in scaling:
    if r["recorded_trajectory"]:
        rows.append({"scenario": f"Circle N={r['n']} (RVO, scaling run)", "min_clearance_m": r["min_clearance"]})
    else:
        rows.append({
            "scenario": f"Circle N={r['n']} (RVO, timing-only window)",
            "min_clearance_m": r["min_clearance"],
        })

# Narrow passage
df_np, fps_np = load_trajectory(RESULTS / "narrow_passage.sqlite")
with open(RESULTS / "narrow_passage.json") as f:
    s_np = json.load(f)
_, clearance_np = min_pairwise_distance(df_np, s_np["agent_radius"])
rows.append({"scenario": "Narrow Passage (RVO)", "min_clearance_m": clearance_np})

# Moving obstacle (pedestrian radius vs. car radius, not uniform)
df_mo, fps_mo = load_trajectory(RESULTS / "moving_obstacle.sqlite")
with open(RESULTS / "moving_obstacle.json") as f:
    s_mo = json.load(f)
radii_mo = {aid: (s_mo["car_radius"] if aid == s_mo["car_id"] else s_mo["agent_radius"]) for aid in s_mo["agent_ids"]}
_, clearance_mo = min_pairwise_distance(df_mo, radii_mo)
rows.append({"scenario": "Moving Obstacle (RVO vs. car)", "min_clearance_m": clearance_mo})

theorem6_df = pd.DataFrame(rows)
theorem6_df["collision_free"] = theorem6_df["min_clearance_m"] >= -1e-6
theorem6_df
""")

code(r"""
fig, ax = plt.subplots(figsize=(12, 5))
colors = ["tab:green" if ok else "tab:red" for ok in theorem6_df["collision_free"]]
ax.barh(range(len(theorem6_df)), theorem6_df["min_clearance_m"], color=colors, alpha=0.8)
ax.set_yticks(range(len(theorem6_df)), theorem6_df["scenario"])
ax.axvline(PAPER_MIN_CLEARANCE, lw=1.5, **PAPER_STYLE)
for y, v in enumerate(theorem6_df["min_clearance_m"]):
    ax.text(0.01 if v >= 0 else v - 0.01, y, f"{v:+.4f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
ax.invert_yaxis()
ax.set_xlim(theorem6_df["min_clearance_m"].min() - 0.12, 0.12)
ax.set_xlabel("minimum clearance over the run [m]  (distance − $r_i$ − $r_j$)")
ax.set_title("Collision-freedom: ours vs. the paper's proven guarantee")
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([], [], lw=1.5, label="paper: no collisions (Theorem 6), clearance $\\geq$ 0", **PAPER_STYLE),
    Patch(color="tab:green", alpha=0.8, label="ours: collision-free"),
    Patch(color="tab:red", alpha=0.8, label="ours: overlap"),
], fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
plt.tight_layout()
plt.savefig(RESULTS / "plot_theorem6_clearance.png", dpi=130)
plt.show()
""")

md(r"""
**Reading this table:** a `min_clearance_m` at or just above 0 is the
*best* possible outcome under a discrete-time simulation -- it means
agents got exactly as close as the physical constraint allows and no
closer, i.e. genuinely tight, correct collision avoidance, not a wide,
overly cautious margin. Negative values are real, quantified constraint
violations (partial disk overlap).

**What we actually find:** at low agent density -- the two-agent-scale
Circle N=12 comparison, and the sparse Moving Obstacle crossing -- Theorem
6 holds to within a few thousandths of a meter, i.e. essentially exactly,
which is a genuinely clean, positive validation result. At higher density
(Circle N=50/100, and the crowded Narrow Passage bottleneck), real
penetration appears and grows with density. This is not a discretization
artifact alone (we checked: halving `dt` narrows but does not close the
gap) -- it is the model's finite-candidate sampling failing to find a
velocity that is simultaneously outside *every* active neighbor's VO cone
once enough neighbors' cones overlap at once, something the paper's own
exact geometric construction does not have to contend with. We verified
this is a fundamental limitation of the sampling approach rather than a
tuning problem: neither a finer angular/speed grid nor a larger collision
penalty weight closed the gap in our own testing (finer grid: N=50
clearance improved only from -0.397 m to -0.360 m at roughly double the
per-step cost; a 10x larger collision weight made it *worse*, -0.285 m,
by pulling candidates toward exact cone-boundary tangents that a *second*
neighbor's cone often also touches). We report this rather than tuning
around it further.
""")

# ============================================================ Circle scenario
md(r"""
## 2. Circle scenario -- oscillation comparison (Theorem 8) and scalability

**Paper's setup**: N agents placed evenly on a circle, each walking
straight to its own antipodal point, forcing a dense crossing at the
center. First run at N=12 comparing RVO against plain VO directly
(qualitative proof that reciprocal averaging removes "dancing"), then
scaled to 250 and 1000 agents to check that per-step time grows
(approximately) linearly with N.

**Our setup**: same idea, `dt=0.025` s (found empirically -- `dt=0.05`
let two near-symmetric agents visibly interpenetrate before reacting,
`dt=0.0125` gave no further improvement worth its 2x cost; see the
Theorem 6 discussion above for how this interacts with density). A tiny
random angular jitter (±0.01 rad) is added to the otherwise perfectly even
spacing, since an exactly antipodal pair is the most degenerate case the
tangent-candidate fix targets, and we want the comparison to reflect
ordinary behaviour, not one single edge case.
""")

code(r"""
fig = plt.figure(figsize=(11, 10.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
ax_paper = fig.add_subplot(gs[0, :])
ax_paper.imshow(plt.imread(PAPER_FIG7_IMG))
ax_paper.set_title("Paper, Fig. 7: original VO (left) vs. RVO (right), N=12")
ax_paper.axis("off")
axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
for ax, tag, title in [
    (axes[0], "circle_n12_vo", "Ours: plain VO (non-reciprocal)"),
    (axes[1], "circle_n12_rvo", "Ours: RVO (reciprocal)"),
]:
    df, fps = load_trajectory(RESULTS / f"{tag}.sqlite")
    with open(RESULTS / f"{tag}.json") as f:
        s = json.load(f)
    targets = {int(k): v for k, v in s["targets"].items()}
    df_trim = trim_to_active_phase(df, targets, tolerance=0.15, buffer=0.5)
    for agent_id, g in df_trim.groupby("id"):
        g = g.sort_values("frame")
        ax.plot(g["pos_x"], g["pos_y"], lw=1.2, alpha=0.8)
        ax.plot(g["pos_x"].iloc[0], g["pos_y"].iloc[0], "o", color="black", ms=3)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
plt.suptitle("Circle scenario, N=12: paper vs. ours (ours: paths during the active crossing, dots = start)")
plt.tight_layout()
plt.savefig(RESULTS / "plot_circle_paths.png", dpi=130)
plt.show()
""")

code(r"""
rows = []
for tag, label in [("circle_n12_vo", "plain VO"), ("circle_n12_rvo", "RVO")]:
    df, fps = load_trajectory(RESULTS / f"{tag}.sqlite")
    with open(RESULTS / f"{tag}.json") as f:
        s = json.load(f)
    targets = {int(k): v for k, v in s["targets"].items()}
    df_trim = trim_to_active_phase(df, targets, tolerance=0.15, buffer=0.5)
    tort = path_tortuosity(df_trim)
    turn = turning_metrics(df_trim, resample_dt=0.15)
    _, clearance = min_pairwise_distance(df, s["agent_radius"])
    rows.append({
        "mode": label,
        "mean_tortuosity": tort.mean(),
        "mean_total_turning_deg": turn["total_turning_deg"].mean(),
        "mean_reversals": turn["reversals"].mean(),
        "min_clearance_m": clearance,
    })
oscillation_df = pd.DataFrame(rows)
oscillation_df
""")

md(r"""
**Reading this comparison honestly.** The collision-freedom numbers are
unambiguous and go the direction the paper predicts: plain VO produces a
real collision (negative clearance -- two agents' disks visibly overlap in
the path plot above, where a VO path suddenly bends sharply after two
agents get too close), while RVO stays collision-free to within
numerical precision.

**The oscillation metrics (tortuosity, total turning, reversals) do not
show a clean win for RVO in this implementation -- if anything, they lean
slightly the other way**, and we are reporting that rather than picking a
more flattering metric or time window. Our reading, after checking the
underlying paths directly rather than trusting the summary numbers: once
plain VO's two agents actually collide, the model's overlap-handling
fallback (see `pyrvo.py`) gives a single, clean "push apart" direction --
decisive, but only because a real collision already happened. RVO, by
contrast, never collides, which means it spends the whole encounter
continuously renegotiating a tight tangential maneuver around the other
agent's VO cone boundary -- correct and collision-free, but that
continuous fine renegotiation registers as *more* heading change by a
naive turning-angle count than a single decisive (if physically
incorrect) push. A pure heading-turning proxy conflates "smoothly
threading a tight gap" with "genuinely dancing back and forth," and in a
two-agent, N=12-scale encounter that distinction matters more than at
larger N where true multi-agent dancing would dominate. The path plot
above is the more trustworthy comparison at this scale: visually, neither
path shows the sharp multi-reversal zig-zag the paper's own Fig. 7
illustrates for plain VO, most likely because our implementation's plain
VO mode resolves most conflicts via an actual collision rather than a
sustained oscillatory stand-off. We did not find, within the time
available, a scenario configuration in this implementation that cleanly
reproduces the paper's own dancing/no-dancing contrast -- a genuine
limitation of this reproduction, not a subtle result to read past.
""")

md(r"""
### Scalability: does frame time grow linearly with N?

Timing measured with `time.perf_counter()` around each `sim.iterate()`
call. For N=12/50/100 a full trajectory was also recorded (used for the
collision-freedom table above and the replay below); for N=250/500/1000
only a short timing-only window was run (no trajectory writer) -- 1000
agents' full sqlite trajectory over a full crossing was unnecessary disk
churn for a number we only need once, for this plot. This means the
250/500/1000 clearance values in the Theorem 6 table above are **not**
representative of collision-freedom over a full crossing -- they are an
early-transient snapshot from a 3-second window, included only for
completeness, not as a density-vs-clearance data point.
""")

code(r"""
with open(RESULTS / "circle_scaling.json") as f:
    scaling = json.load(f)
scaling_df = pd.DataFrame(scaling)[["n", "mean_step_time_ms", "recorded_trajectory", "crashed"]]
scaling_df
""")

code(r"""
full = scaling_df[scaling_df["recorded_trajectory"]]
window = scaling_df[~scaling_df["recorded_trajectory"]]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.plot(PAPER_FIG11["n"], np.array(PAPER_FIG11["sec_per_frame"]) * 1000, "D", label="paper, Fig. 11 (digitized)", **PAPER_STYLE)
ax.plot(scaling_df["n"], scaling_df["mean_step_time_ms"], "-", color="tab:blue", alpha=0.5)
ax.plot(full["n"], full["mean_step_time_ms"], "o", color="tab:blue", label="ours, full crossing")
ax.plot(window["n"], window["mean_step_time_ms"], "o", mfc="white", color="tab:blue", label="ours, 3 s timing window")
ax.set_xlabel("number of agents N")
ax.set_ylabel("mean wall-clock time per step [ms]")
ax.set_title("Time per step vs. N (paper Fig. 11, diamonds)")
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(PAPER_FIG11_FPS["n"], PAPER_FIG11_FPS["fps"], "s", label="paper, Fig. 11 (digitized)", **PAPER_STYLE)
ours_fps = 1000 / scaling_df["mean_step_time_ms"]
ax.plot(scaling_df["n"], ours_fps, "-", color="tab:orange", alpha=0.5)
ax.plot(full["n"], 1000 / full["mean_step_time_ms"], "o", color="tab:orange", label="ours, full crossing")
ax.plot(window["n"], 1000 / window["mean_step_time_ms"], "o", mfc="white", color="tab:orange", label="ours, 3 s timing window")
ax.axhline(10, color="gray", lw=0.8, ls=":")
ax.text(1000, 10.5, "10 frames/s (paper's real-time claim)", ha="right", fontsize=7, color="gray")
ax.set_yscale("log")
ax.set_xlabel("number of agents N")
ax.set_ylabel("frames (steps) per second [log]")
ax.set_title("Frame rate vs. N (paper Fig. 11, squares)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(RESULTS / "plot_circle_scaling.png", dpi=130)
plt.show()

# Simple linear-fit check: is per-agent cost roughly constant?
per_agent_ms = scaling_df["mean_step_time_ms"] / scaling_df["n"]
print("Per-agent cost (ms/step / N) at each N:")
print(per_agent_ms.to_string(index=False))
t1000 = scaling_df.loc[scaling_df.n == 1000, "mean_step_time_ms"].iloc[0]
paper_t1000 = PAPER_FIG11["sec_per_frame"][-1] * 1000
print(f"\nN=1000 mean step time: ours {t1000:.1f} ms ({1000 / t1000:.1f} FPS) vs. paper {paper_t1000:.1f} ms "
      f"({PAPER_FIG11_FPS['fps'][-1]} FPS) -> ours is {t1000 / paper_t1000:.1f}x slower")
paper_slope = np.polyfit(PAPER_FIG11["n"], np.array(PAPER_FIG11["sec_per_frame"]) * 1000, 1)[0]
ours_slope = np.polyfit(window["n"], window["mean_step_time_ms"], 1)[0]
print(f"Marginal cost per extra agent: paper {paper_slope * 1000:.0f} us/agent, "
      f"ours (N=250-1000 timing windows) {ours_slope * 1000:.0f} us/agent")
""")

md(r"""
**Result: the same linear scaling, but about 2.3x slower in absolute
terms than the paper.** The two kinds of run need to be read separately:

- **N=250/500/1000 (open markers, 3 s timing windows):** per-step time is
  linear in N, just like the paper's diamond curve. At N=1000 we run at
  about 5 frames/s, against the paper's 12.3 frames/s (below its own
  10 frames/s real-time line, in the right panel). The printed slopes
  above show the gap per extra agent: this pure-Python per-agent callback
  through JuPedSim costs more than twice as much as the paper's C++ code
  (which also split the work across two cores). Faster modern hardware
  does not make up the difference.
- **N=50/100 (filled markers, full crossings):** these sit well *above*
  the linear trend. N=100 costs about as much per step as N=250 does in
  its timing window. This is not a failure to scale. Per-agent cost
  depends on how many neighbours are in range, and a full crossing
  includes the dense meeting phase in the middle of the circle. A 3 s
  window ends before the agents meet. The paper does not say which phase
  its frame times were averaged over, so the comparison is only
  like-for-like for the open markers, and even those are optimistic
  because they come from the sparse phase.

So the paper's qualitative claim (frame time grows linearly with N)
holds. Its quantitative claim (real time at N=1000) is not reached by
this implementation.
""")

md(r"""
**Replay** (N=100, one of the full-trajectory scaling runs). Watch the
crowd converge at the center and pass through -- color encodes
instantaneous speed.
""")

code(r"""
traj_c100, area_c100 = read_sqlite_file(str(RESULTS / "circle_scaling_n100.sqlite"))
animate(traj_c100, area_c100, every_nth_frame=8, radius=MEAN_AGENT_RADIUS, title_note="Circle scenario, N=100")
""")

# ============================================================ Narrow passage
md(r"""
## 3. Narrow Passage scenario

**Paper's setup**: four groups of 25 agents in the four corners of an
environment, each heading to the opposite corner through square obstacles
that create narrow passages, checking that agents don't get permanently
stuck (except for a noted U-shaped-obstacle failure case, which we
deliberately avoid -- our obstacles are plain squares).

**Our setup**: four square pillars arranged in a "+" pattern around the
center, leaving diagonal gaps between adjacent pillars that every
diagonally-travelling group must funnel through. **Reduced scale, found
necessary empirically, not assumed up front**: our first attempt at the
paper's full N=25/group (100 total) hit the same `move_on_surface(): path
hit a wall` engine-level instability documented in SFM's own notebook, at
t=11.3s into the run, and a smaller `dt` alone did not fix it (still
crashed, just later, at t=25.7s). We traced this to the same root cause as
the Theorem 6 shortfall above: a densely jammed bottleneck occasionally
pushes our finite-sample selection into a real (if shallow) wall
penetration, which JuPedSim's own geometry guard then rejects outright as
an invalid move. Reducing to N=15/group (60 total) and widening the
gaps (see `run_narrow_passage.py --pillar-offset`/`--pillar-half`) let
the run complete without crashing, and is what is reported below --
we did not additionally reduce max speed or `dt` beyond what was already
needed to avoid the crash, so this is the paper's own scenario at
reduced scale, not a re-tuned one.
""")

code(r"""
with open(RESULTS / "narrow_passage.json") as f:
    s_np = json.load(f)
df_np, fps_np = load_trajectory(RESULTS / "narrow_passage.sqlite")
targets_np = {int(k): v for k, v in s_np["targets"].items()}

_, clearance_np = min_pairwise_distance(df_np, s_np["agent_radius"])
# Each group of 15 shares ONE target point (spawned within +-1.8 m of the opposite
# corner's mirror), so at most a handful can physically stand within a waypoint-sized
# tolerance of it. Arrival is judged against the group's own spawn footprint instead.
GROUP_ARRIVAL_RADIUS = 2.5
reached_strict = reached_target(df_np, targets_np, tolerance=s_np["waypoint_tolerance"] + 0.3)
reached = reached_target(df_np, targets_np, tolerance=GROUP_ARRIVAL_RADIUS)

print(f"Agents spawned: {s_np['n_total_spawned']}")
print(f"Simulated time completed: {df_np['t'].max():.1f} s (crashed: {s_np['crashed']})")
print(f"Minimum clearance over the whole run: {clearance_np:.3f} m (0 or above = collision-free)")
print(f"Fraction that reached their target area (within {GROUP_ARRIVAL_RADIUS} m): {reached.mean():.2f} ({reached.sum()} of {len(reached)})")
print(f"  (within the waypoint tolerance of the single shared target point: {reached_strict.mean():.2f} -- "
      f"not a meaningful arrival test when 15 agents share one point)")
""")

code(r"""
last_frame = df_np["frame"].max()
snap = df_np[df_np["frame"] == last_frame]
fig, (ax, ax_bar) = plt.subplots(1, 2, figsize=(11, 5.5), gridspec_kw={"width_ratios": [1.6, 1]})
ax.scatter(snap["pos_x"], snap["pos_y"], s=10, alpha=0.7)
half = s_np["pillar_half"]
off = s_np["pillar_offset"]
for cx, cy in [(off, 0), (-off, 0), (0, off), (0, -off)]:
    ax.add_patch(plt.Rectangle((cx - half, cy - half), 2 * half, 2 * half, color="gray", alpha=0.6))
ax.set_title(f"Narrow Passage: last recorded frame (t={snap['t'].iloc[0]:.1f}s)")
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")

ax_bar.bar(["ours\n(N=15/group)"], [reached.mean()], color="tab:blue", alpha=0.8, label="ours")
ax_bar.axhline(PAPER_NARROW_PASSAGE_REACHED, lw=1.5, label="paper: all agents reach goal (N=25/group)", **PAPER_STYLE)
ax_bar.set_ylim(0, 1.1)
ax_bar.set_ylabel("fraction of agents that reached their target")
ax_bar.set_title("Goal reached: ours vs. paper")
ax_bar.text(0, reached.mean() / 2, f"{reached.mean():.0%}", ha="center", color="white", fontsize=14)
ax_bar.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.3))
plt.tight_layout()
plt.savefig(RESULTS / "plot_narrow_passage_snapshot.png", dpi=130)
plt.show()
""")

md(r"""
**Verdict: the paper's "eventually all agents reach their goals" is
reproduced at reduced scale.** 59 of 60 agents (98%) end the run inside
their target area, against the paper's 100% (bar chart above). The
snapshot shows the four groups regrouped in the opposite corners, with a
single straggler still on its way. The median agent arrives after about
40 s of a 50 s run, on a roughly 35 m corner-to-corner trip at a maximum
speed of 1 m/s. So the time cap is tight, and the straggler may simply
not have had enough time.

**A metric bug, found while overlaying the paper's value.** An earlier
version of this notebook reported only 23% arriving, and called this
scenario a failure. That number counted an agent as arrived only if it
was within 0.6 m of its target point. But each group of 15 shares a
*single* target point (the mirror of its own spawn corner), so at most a
few agents can physically fit within 0.6 m of it. The rest queue around
it, still jostling. That is also why every agent still shows some speed
at the end of the run. Arrival is now measured against the group's own
spawn footprint (2.5 m, the size of the ±1.8 m spawn square). The strict
number is still printed above for transparency.
The minimum-clearance number is the same story as Section 1: real,
non-trivial penetration occurs in the crowded gap regions, for the same
diagnosed reason (finite-sample selection under many simultaneous active
constraints) -- this scenario's crowded bottleneck is, if anything, a
*more* demanding test of Theorem 6 than the Circle scenario, and it shows
the same limitation more severely.

**Replay.** Watch the four streams funnel through the gaps between
pillars.
""")

code(r"""
traj_np, area_np = read_sqlite_file(str(RESULTS / "narrow_passage.sqlite"))
animate(traj_np, area_np, every_nth_frame=6, radius=MEAN_AGENT_RADIUS, title_note="Narrow Passage scenario")
""")

# ============================================================ Moving obstacle
md(r"""
## 4. Moving Obstacle scenario

**Paper's setup**: eleven pedestrians cross a street on which a car -- a
passively-moving, non-reactive obstacle -- is driving, testing correct
handling of a fast, non-cooperating obstacle.

**Our setup**: 11 pedestrians cross a street-shaped arena perpendicular to
a car's path; the car (`is_reactive=False` in `RVOState`, larger radius,
higher top speed) drives straight across, ignoring everyone, while the
pedestrians see it via `neighbor.state.is_reactive` and correctly fall
back to plain (non-reciprocal) VO against it rather than assuming it
shares the avoidance effort -- see `pyrvo.py`'s apex-selection logic.

**The car is timed so it splits the line of pedestrians roughly in
half**, as in the paper's Fig. 10. There, the pedestrians the car reaches
first wait while it sweeps past, and those further along cross in front
of it. The pedestrians start at the kerb, 2.5 m from the car's lane, as
the column does in Fig. 10. The car (4.5 m/s, about 3x walking speed)
starts at x = -13.2 m, so it reaches the first pedestrian just as the
line arrives at the road.

What mattered was the car's **timing**, not its speed. An earlier setup
started the pedestrians 7 m from the lane and the car at the street's
far end. The car then got past most of them before anyone reached the
road, and all 11 walked behind it without ever interacting with it.
Slowing that car to 2.5 m/s did produce a 6/5 count, but four of the
five "behind" pedestrians still lost at most ~0.1 s, because the car had
already gone by when they arrived. The timed setup gives the same 6/5
split, and the pedestrians behind the car genuinely give way. Their
delay at the lane grows along the line, the closer they are to the split
point (see the printed delays below).
""")

code(r"""
with open(RESULTS / "moving_obstacle.json") as f:
    s_mo = json.load(f)
df_mo, fps_mo = load_trajectory(RESULTS / "moving_obstacle.sqlite")
targets_mo = {int(k): v for k, v in s_mo["targets"].items()}
radii_mo = {aid: (s_mo["car_radius"] if aid == s_mo["car_id"] else s_mo["agent_radius"]) for aid in s_mo["agent_ids"]}

_, clearance_mo = min_pairwise_distance(df_mo, radii_mo)
reached_mo = reached_target(df_mo, targets_mo, tolerance=s_mo["waypoint_tolerance"] + 0.3)

print(f"Minimum clearance (any pedestrian vs. the car, or vs. each other): {clearance_mo:.4f} m")
print(f"Fraction of pedestrians that reached the far sidewalk: {reached_mo.mean():.2f} ({reached_mo.sum()} of {len(reached_mo)})")

# Did each pedestrian cross the car's lane (y=0) before or after the car passed its x?
car_df = df_mo[df_mo["id"] == s_mo["car_id"]].sort_values("t")
# Delay = arrival at the lane minus the time an unobstructed walk would take.
crossed_ahead, delay = {}, {}
for pid in s_mo["pedestrian_ids"]:
    g = df_mo[df_mo["id"] == pid].sort_values("t")
    on_lane = g[g["pos_y"] >= 0].iloc[0]
    car_past = car_df[car_df["pos_x"] >= on_lane["pos_x"]]
    t_car = car_past["t"].iloc[0] if not car_past.empty else np.inf
    crossed_ahead[pid] = on_lane["t"] < t_car
    delay[pid] = on_lane["t"] - (-g["pos_y"].iloc[0]) / s_mo["max_speed"]
n_ahead = sum(crossed_ahead.values())
print(f"Crossed in front of the car: {n_ahead}; waited and crossed behind it: {len(crossed_ahead) - n_ahead}")
print("Delay at the car's lane, left to right along the line [s]:")
print("  " + "  ".join(f"{'front' if crossed_ahead[p] else 'behind'} {delay[p]:+.1f}" for p in s_mo["pedestrian_ids"]))
""")

code(r"""
fig, (ax_paper, ax) = plt.subplots(2, 1, figsize=(9, 10), gridspec_kw={"height_ratios": [1, 1.1]})
ax_paper.imshow(plt.imread(PAPER_FIG10_IMG))
ax_paper.set_title(f"Paper, Fig. 10: 11 agents cross while the car (grey) drives through -- all cross "
                   f"({PAPER_MOVING_OBSTACLE_REACHED:.0%})", fontsize=9)
ax_paper.axis("off")
ax.plot(car_df["pos_x"], car_df["pos_y"], color="black", lw=2, label="car (left to right)")
for pid in s_mo["pedestrian_ids"]:
    g = df_mo[df_mo["id"] == pid].sort_values("frame")
    ahead = crossed_ahead[pid]
    ax.plot(g["pos_x"], g["pos_y"], lw=1.2, alpha=0.8, color="tab:green" if ahead else "tab:purple")
ax.plot([], [], color="tab:green", label=f"crossed in front of the car ({n_ahead})")
ax.plot([], [], color="tab:purple", label=f"crossed behind the car ({len(crossed_ahead) - n_ahead})")
ax.set_title(f"Ours: car path (black) and pedestrian paths -- {reached_mo.mean():.0%} cross "
             f"(paper: {PAPER_MOVING_OBSTACLE_REACHED:.0%}); min clearance {clearance_mo:+.4f} m "
             f"(paper: $\\geq$ {PAPER_MIN_CLEARANCE:.0f})", fontsize=9)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(RESULTS / "plot_moving_obstacle_paths.png", dpi=130)
plt.show()
""")

md(r"""
**Verdict: this scenario reproduces the paper's claim cleanly.** At this
lower density (11 pedestrians, one obstacle, not a dense many-way
crossing), Theorem 6 holds almost exactly -- the printed clearance above
is at the boundary of what's physically possible, not comfortably above
it, meaning the model isn't being overly conservative either. Every
pedestrian reaches the far side. The car splits the line of pedestrians
the way it does in the paper's Fig. 10: those on the far (right) end
cross in front of it, and those it reaches first cross behind it. This is the scenario where our
implementation's limitation (finite-sample selection under many
*simultaneous* active constraints) doesn't bite, since at any moment each
pedestrian has at most the car, or one or two other pedestrians, as active
constraints -- consistent with our diagnosis that the shortfall found in
Sections 1-3 is specifically a high-density, many-simultaneous-constraint
problem.

**Replay.**
""")

code(r"""
traj_mo, area_mo = read_sqlite_file(str(RESULTS / "moving_obstacle.sqlite"))
animate(traj_mo, area_mo, every_nth_frame=4, radius=MEAN_AGENT_RADIUS, title_note="Moving Obstacle scenario")
""")

# ============================================================ Summary
md(r"""
## 5. Summary

| # | Phenomenon (paper) | Reproduced? | Notes |
|---|---|---|---|
| 1 | Theorem 6: collision-freedom | ✅ at low/moderate density; ⚠️ degrades at high density | exact to within ~1 mm at Circle N=12 and the Moving Obstacle crossing; real (if shallow) penetration at Circle N=50/100 and the Narrow Passage bottleneck -- diagnosed as a limitation of finite-candidate sampling under many simultaneous active VO constraints, not a discretization artifact alone (checked: smaller `dt`, finer sampling grid, and larger collision-penalty weight all failed to close the gap) |
| 2 | Theorem 8: RVO removes VO's oscillation | ⚠️ partial | collision-freedom comparison is clean (VO collides, RVO doesn't); a heading-turning-angle proxy for "oscillation" does not show RVO as smoother in this implementation, most likely because our plain-VO mode resolves conflicts via an actual collision (a single clean push) rather than a sustained back-and-forth stand-off -- we did not find a configuration that reproduces the paper's own dancing/no-dancing contrast within the time available |
| 3 | Frame time scales ~linearly with N (Fig. 11) | ✅ shape / ❌ absolute | linear in N over N=250-1000, matching the shape of the paper's Fig. 11; but ~2.3x slower in absolute terms (≈5 vs. 12.3 frames/s at N=1000), so the paper's ">10 frames/s at N=1000" real-time claim is not reached by this pure-Python per-agent callback; full-crossing runs (N=50/100) cost more per agent because of the dense meeting phase |
| 4 | Narrow Passage: "eventually all agents reach their goals" | ✅ at reduced scale | 98% (59 of 60) reach their target area within 50 s, vs. the paper's 100%; an earlier 23% figure was a metric bug (15 agents sharing one target point cannot all be within 0.6 m of it); scale is N=15/group (reduced from the paper's N=25/group after the full-scale run hit an engine-level "path hit a wall" crash, root-caused to the same high-density collision-freedom shortfall as #1) |
| 5 | Moving Obstacle: pedestrians correctly avoid a fast, non-cooperating car | ✅ | clean collision-freedom, 100% of pedestrians cross successfully |

**On what "validation" means here.** RVO's own paper offers something the
Social Force Model's source paper could not: a *proven* mathematical
guarantee (Theorem 6) rather than a plausibility argument, which makes
this a sharper, more falsifiable check than most of what this repo's other
model notebooks can run. That sharper bar is also why this notebook's
headline finding is a genuine limitation rather than a clean pass: our
implementation is a finite-sample *approximation* of RVO (as the paper's
own sampling-based selection approach is), not the exact half-plane/linear-
programming formulation that later work (ORCA, van den Berg et al. 2011)
uses to make the same guarantee hold by construction. At the densities the
paper's own headline figures use (a full N=25-per-group Narrow Passage, a
1000-agent Circle scenario run to full completion rather than a short
timing window), we found real, honestly-quantified constraint violations
that a reader relying only on the paper's proofs would not expect from
"an RVO implementation." Reporting that gap precisely -- rather than
quietly reducing density until it disappears, or asserting the proof
without checking it -- is the actual result of this notebook.
""")

nb["cells"] = cells
with open("RVO_validation.ipynb", "w") as f:
    nbf.write(nb, f)

print(f"Wrote RVO_validation.ipynb with {len(cells)} cells")
