"""Programmatically builds SPH_Crowds_validation.ipynb from markdown/code cell
source strings below, so the notebook's content is versionable as plain
Python/Markdown rather than raw ipynb JSON. Run, then execute with:

    jupyter nbconvert --to notebook --execute --inplace SPH_Crowds_validation.ipynb
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
# Validating the Python/JuPedSim SPH Crowds model against van Toll, Chatagnon, Braga, Solenthaler & Pettré (2021)

This notebook checks a Python implementation of **SPH Crowds**, running
inside **JuPedSim**'s experimental `CustomOperationalModel` plugin API like
the Social Force and RVO models elsewhere in this repo, against the paper
that introduced it:

> W. van Toll, T. Chatagnon, C. Braga, B. Solenthaler, J. Pettré. *SPH
> crowds: Agent-based crowd simulation up to extreme densities using fluid
> dynamics.* Computers & Graphics 98, 306–321 (2021).
> [doi:10.1016/j.cag.2021.06.005](https://doi.org/10.1016/j.cag.2021.06.005)

SPH Crowds does not replace a collision-avoidance model; it **augments**
one. Every agent is also a Smoothed Particle Hydrodynamics particle: it
measures a local SPH density, keeps a personal, slowly adapting rest
density (clamped to at most $\rho^{0,\max}$), and feels a pressure force
whenever the density exceeds it. Agents can also **blend** from a
navigation profile (social forces or RVO) to pure SPH as the density
rises.

**How the paper validated it** (see `model_description.md`): not against
recorded trajectories, but with three synthetic experiments whose set-ups
are motivated by real ones, plus a performance measurement:

1. **Room evacuation** (Sec. 6.1, Fig. 5, **Table 1**): 400 agents leave a
   20 × 20 m room through a 0.8 m door, with 7 profiles and a sweep of
   $K^{ag}$ or $\rho^{0,\max}$. Reported: evacuees, mean SPH density at
   15 s, and flow rate. The claim: $\rho^{0,\max}$ controls the density
   directly, while $K^{ag}$ barely does.
2. **Crossing and bottleneck** (Sec. 6.2, Fig. 6, **Table 2**): a crossing
   flow in front of a bottleneck. The claim: only density-dependent
   blending (RVO→SPH) handles both the crossing and the bottleneck.
3. **Concert shockwave** (Sec. 6.3, **Figs. 7–10**): 10,000 agents in
   front of a stage, and a push that travels to the stage and back. The
   claims are qualitative: $k$ sets the wave's speed and thickness,
   $\rho^{0,\max}$ sets the density but barely the wave, viscosity and a
   dynamic rest density improve stability, and contact forces keep the
   boundary intact.
4. **Performance** (Sec. 6.4, **Fig. 11**): frame time vs. the number of
   agents (4.94 ms at 10,000 agents on six threads).

**Code layout** (all in `SPH_Crowds/`):

- `pySPH_Crowds.py`: the model (`SPHCrowdModel`, `SPHCrowdState`), with
  all seven profiles of Sec. 5.5.
- `validation/run_room_evacuation.py`, `run_crossing_bottleneck.py`,
  `run_concert.py`: one runner per paper experiment.
- `validation/sweep_all.py`: runs every data point as its own OS process.
- `validation/analysis.py`: shared trajectory loading, metrics and the
  paper's colour schemes (Fig. 3).
- `validation/results/`: `.sqlite` trajectories, `.json` summaries,
  concert snapshots (`.npz`), our plots (`plot_*.png`), and figures
  cropped from the paper (`paper_fig*.png`).
""")

# ============================================================ sys.path cell
code(r"""
import pathlib
import sys

# jupedsim is not pip-installed; its Python package and compiled bindings are
# made importable via PYTHONPATH (see jupedsim/build/environment). Jupyter
# kernels don't inherit that, so replicate it here. The kernel cwd is this
# notebook's folder (SPH_Crowds/), matching the pathlib.Path("validation") usage below.
_REPO = pathlib.Path.cwd().parent
_JP = _REPO / "jupedsim"
for _d in (
    _JP / "python_modules" / "jupedsim_examples",
    _JP / "build" / "stage",
):
    _p = str(_d.resolve())
    if _p not in sys.path:
        sys.path.insert(0, _p)

# sanity check
from jupedsim.internal.notebook_utils import animate, read_sqlite_file
print("jupedsim import OK")
""")

# ============================================================ Setup
code(r"""
import json
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pedpy
import shapely

import sys
sys.path.insert(0, str(pathlib.Path("validation").resolve()))
from analysis import (
    load_json, load_trajectory, add_velocity, DENSITY_CMAP, velocity_colors,
    wave_profile, wave_front, boundary_jitter,
)

RESULTS = pathlib.Path("validation/results")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

# Same caveat as the SFM and RVO notebooks: this is JuPedSim's own internal
# replay utility. Its module docstring says explicitly "We make no promises
# about the functions from this file w.r.t. API stability... Do not use the
# code here. Use it at your own peril." We use it anyway, for the same
# reason -- there isn't a public, stable alternative.
from jupedsim.internal.notebook_utils import animate, read_sqlite_file

MEAN_AGENT_RADIUS = 0.24  # agents have D in [0.215, 0.265] m (Sec. 5.6)
SAFE = {"SF": "SF", "RVO": "RVO", "SPH": "SPH", "SF+SPH": "SFplusSPH", "RVO+SPH": "RVOplusSPH",
        "SF->SPH": "SFtoSPH", "RVO->SPH": "RVOtoSPH"}
PRETTY = {p: p.replace("->", "→") for p in SAFE}
BLEND_LOW, BLEND_HIGH = 2.0, 4.0  # profile sequence {(SF or RVO, 2), (SPH, 4)}, Sec. 5.5


def blend_weight(rho):
    '''SPH weight kappa of the blended profiles (Sec. 4.4), same for SF->SPH and RVO->SPH.'''
    return np.clip((np.asarray(rho, float) - BLEND_LOW) / (BLEND_HIGH - BLEND_LOW), 0.0, 1.0)


def sub_animation(db, keep, every_nth_frame, title, t_window=None):
    '''Animate only agents/frames selected by `keep(df)`, to keep the widget small.'''
    df, fps = load_trajectory(db)
    if t_window is not None:
        df = df[(df["t"] >= t_window[0]) & (df["t"] <= t_window[1])]
    df = df[keep(df)]
    traj = pedpy.TrajectoryData(data=df.rename(columns={"pos_x": "x", "pos_y": "y"})[["frame", "id", "x", "y"]],
                                frame_rate=fps)
    _, area = read_sqlite_file(str(db))
    return animate(traj, area, every_nth_frame=every_nth_frame, radius=MEAN_AGENT_RADIUS, title_note=title)
""")

# ============================================================ Paper reference data
md(r"""
### Reference data from the paper

Every plot below also shows the paper's own result, drawn in black and
labelled "paper".

- **Tables 1 and 2** are numeric tables, so their values are
  **transcribed exactly** (`PAPER_TABLE1`, `PAPER_TABLE2`), not digitized.
- **Fig. 11** (frame time vs. number of agents) is the only numeric plot.
  Its "Total" curve was **hand-digitized**: page 14 was rendered at
  300 dpi with `pdftoppm`, the plot was cropped, and each × marker was
  read off against the printed gridlines (10.4 px per ms). The error is
  about ±0.3 ms. Two points are also stated in the text: 4.94 ms at
  10,000 and 18.57 ms at 30,000 agents. The digitized values (4.9, 18.6)
  agree with them.
- **Claims stated in words** become reference values:
  - "the average crowd density is indeed close to $\rho^{0,\max}$" for the
    SPH profile: the line density = $\rho^{0,\max}$;
  - real evacuation flows are **2–4 P/s** and densities **around 4 P/m²**
    (Garcimartín et al. 2016, cited in Sec. 6.1): a shaded band;
  - in the concert, a wave "moves towards the stage, bounces off, and then
    moves backward", faster and thicker for larger $k$.
- **Qualitative figures** (Figs. 5–10: snapshots) were cropped from the
  PDF into `validation/results/paper_fig*.png`. Each is shown next to our
  version, with the same layout and colour schemes (Fig. 3: density
  colours from blue = 0 to purple = 8 P/m²; velocity colours with hue =
  direction and saturation = speed, full at 0.75 m/s).
""")

code(r"""
# Table 1 (Sec. 6.1), transcribed: (#evacuated, mean density at 15 s, std, flow rate)
PAPER_TABLE1 = pd.DataFrame([
    ("SF", "K_ag", 50, 398, 4.39, 1.05, 2.70), ("SF", "K_ag", 100, 398, 4.32, 1.00, 2.36),
    ("SF", "K_ag", 250, 398, 4.25, 0.92, 2.01), ("SF", "K_ag", 500, 398, 4.19, 0.85, 1.49),
    ("SF", "K_ag", 1000, 398, 4.21, 0.85, 1.29),
    ("RVO", "K_ag", 1000, 400, 3.23, 0.81, 0.97),
    ("SPH", "rho0max", 3, 400, 3.27, 0.27, 3.08), ("SPH", "rho0max", 4, 400, 4.21, 0.45, 4.17),
    ("SPH", "rho0max", 5, 400, 5.09, 0.64, 4.89), ("SPH", "rho0max", 6, 400, 5.89, 0.90, 5.80),
    ("SPH", "rho0max", 7, 400, 6.61, 1.25, 6.45), ("SPH", "rho0max", 8, 400, 7.23, 1.60, 7.20),
    ("SF+SPH", "rho0max", 3, 398, 3.03, 0.39, 2.34), ("SF+SPH", "rho0max", 4, 398, 3.70, 0.74, 2.63),
    ("SF+SPH", "rho0max", 5, 398, 4.13, 1.01, 2.73), ("SF+SPH", "rho0max", 6, 398, 4.24, 1.17, 2.73),
    ("SF+SPH", "rho0max", 7, 398, 4.28, 1.18, 2.78), ("SF+SPH", "rho0max", 8, 398, 4.23, 1.23, 2.79),
    ("RVO+SPH", "rho0max", 3, 400, 2.61, 0.52, 1.43), ("RVO+SPH", "rho0max", 4, 400, 2.96, 0.81, 1.42),
    ("RVO+SPH", "rho0max", 5, 400, 3.00, 0.91, 1.43), ("RVO+SPH", "rho0max", 6, 400, 3.04, 0.92, 1.43),
    ("RVO+SPH", "rho0max", 7, 400, 3.04, 0.92, 1.44), ("RVO+SPH", "rho0max", 8, 400, 3.06, 0.89, 1.44),
    ("SF->SPH", "rho0max", 4, 400, 4.07, 0.65, 3.72), ("SF->SPH", "rho0max", 5, 400, 4.79, 0.98, 4.43),
    ("SF->SPH", "rho0max", 6, 400, 5.32, 1.34, 4.85),
    ("RVO->SPH", "rho0max", 4, 400, 3.85, 0.95, 3.49), ("RVO->SPH", "rho0max", 5, 400, 4.58, 1.30, 4.39),
    ("RVO->SPH", "rho0max", 6, 400, 5.20, 1.70, 4.43),
], columns=["profile", "param", "value", "n_evac", "density", "density_std", "flow"])

# Table 2 (Sec. 6.2), transcribed: mean density in the green rectangle at 45 s (std), flow in first 60 s
PAPER_TABLE2 = pd.DataFrame([
    ("SF", 4.00, 1.09, 1.71), ("RVO", 3.21, 0.77, 1.18), ("SPH", 4.14, 1.33, 4.72),
    ("SF+SPH", 3.81, 1.27, 2.85), ("RVO+SPH", 3.05, 0.80, 1.71), ("SF->SPH", 4.11, 1.77, 4.56),
    ("RVO->SPH", 3.83, 1.55, 4.49),
], columns=["profile", "density", "density_std", "flow"])

# Fig. 11, "Total" curve, hand-digitized (see markdown above); N in agents, frame time in ms
PAPER_FIG11 = {"n": [5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000],
               "frame_ms": [2.2, 4.9, 12.1, 18.6, 25.5, 32.7, 40.3, 49.3]}
PAPER_FIG11_TEXT = {10000: 4.94, 30000: 18.57}  # Sec. 6.4.1 / 6.4.2
PAPER_DT_FINE = 0.02                            # 50 FPS real-time budget, 20 ms

# Fig. 7, forward wave-front x-position [m] vs. time since the push [s], read off
# the six snapshots per row (see the concert section for the method, +-1 m).
PAPER_FIG7_FRONT = {
    "baseline": {"t": [0.3, 1, 2], "x": [44.7, 50.7, 61.3]},
    "k50":      {"t": [0.3, 1, 2, 3], "x": [44.5, 48.2, 54.3, 60.5]},
    "k500":     {"t": [0.3, 1], "x": [44.8, 53.8]},
    "rho4":     {"t": [0.3, 1, 2], "x": [44.7, 50.4, 60.3]},
    "rho6":     {"t": [0.3, 1, 2], "x": [44.7, 51.3, 61.0]},
    "mu0":      {"t": [0.3, 1, 2], "x": [44.7, 50.1, 61.3]},
}
# First snapshot in which the paper shows the wave travelling back (cyan) near the stage
PAPER_FIG7_RETURN = {"baseline": 3, "k50": None, "k500": 2, "rho4": 3, "rho6": 3, "mu0": 3}

# Real-world ranges cited in Sec. 6.1 (Garcimartín et al. 2016)
REAL_FLOW_RANGE = (2.0, 4.0)
REAL_DENSITY_TYPICAL = 4.0

PAPER_IMG = {k: RESULTS / f for k, f in {
    "fig5": "paper_fig5_room_snapshots.png", "fig6": "paper_fig6_crossing_snapshots.png",
    "fig7": "paper_fig7_concert_wave.png", "fig8": "paper_fig8_rest_density.png",
    "fig9": "paper_fig9_concert_no_sph.png", "fig10": "paper_fig10_kag0.png",
    "fig11": "paper_fig11_frame_time.png"}.items()}
PAPER_STYLE = dict(color="black", ls="--", alpha=0.75)
PAPER_MARK = dict(color="black", marker="s", mfc="none", ls="none", ms=7)
""")

# ============================================================ Implementation notes
md(r"""
## Implementation notes: what the model does, where it deviates, and `dt`

`pySPH_Crowds.py` follows Sections 4–5 of the paper equation by equation:
2D Poly6 / spiky / viscosity kernels with $h = 1$ m (Eqs. 13–15), density
with wall contributions (Eqs. 9–10), the exponential-moving-average rest
density clamped to $[\rho^{0,\min}, \rho^{0,\max}]$ (Eq. 8), pressure
$p_i = k(\rho_i - \rho^0_i)$ with the pressure force dropped when
$\rho_i < \rho_i^0$ (Sec. 5.2), pressure and viscosity forces with wall
terms (Eqs. 6, 7, 11, 12), social forces with the relative-velocity
ellipse and a 100° field of view (Eqs. 16–20), RVO's sampled cost
$\|v - v^{pref}\| + w/\mathrm{TTC}(2v - v_i)$ (Eq. 21), Helbing contact
forces (Eqs. 22–23), density-based blending (Sec. 4.4), and all
parameter values of Secs. 5.5–5.6. Every agent has a random radius
$D_i \in [0.215, 0.265]$ m and mass $m_i = (D_i/0.24)^2$.

**`dt` is the paper's own:** $\Delta t_{fine} = 0.02$ s for everything,
and $\Delta t_{coarse} = 0.1$ s for RVO (Sec. 5.6; RVO's chosen
acceleration $(v^* - v_i)/\Delta t_{coarse}$ is reused for 5 fine steps).
The paper states that 0.02 s satisfies the SPH stability (CFL) criterion.
No run in this notebook crashed or showed numerical blow-up at this `dt`,
so we had no reason to search for a different value.

**Where we had to deviate or fill a gap (each is a comment in the code):**

1. **Exact SPH ordering under JuPedSim's per-agent callback.** SPH needs
   every particle's density and pressure *before* any force is computed.
   JuPedSim calls one Python function per agent, so a neighbour's density
   from this step is not available. Storing it in the state would lag by
   one step. Instead, agent $i$ **recomputes** the density, rest-density
   update and pressure of every neighbour $j$ within $h$ from its own $2h$
   neighbourhood. All of $j$'s neighbours (and walls) within $h$ lie
   within $2h$ of $i$, so this reproduces exactly what $j$ computes for
   itself. It is exact, not an approximation, and costs about 2× more
   arithmetic.
2. **Wall shadow area $a(P_k)$.** The paper says only that the area "can
   be computed via simple geometric operations". We integrate it with 64
   rays, assigning each ray to the *first* wall it hits. This matches the
   analytic circular-segment area of a single wall to within about 1%, and
   it avoids double-counting the back face of a 1 m thick wall (or any
   wall hidden behind another), which a per-wall formula would count.
3. **RVO when two disks already overlap** (not covered by the paper): a
   candidate that keeps closing in gets TTC = 0 (maximum penalty), and one
   that separates gets no penalty, so overlapping agents prefer to part.
   RVO samples 100 random velocities in the disk $|v| \le s_{max}$, plus
   $v^{pref}$. The paper only says "many".
4. **Walls are hard in JuPedSim.** A move that leaves the walkable area
   aborts the whole run (`move_on_surface(): path hit a wall`). The
   paper's engine has no such constraint. If a step would cross a wall,
   we remove its component into the nearest wall (the agent slides), or
   stand still for that step if it is still blocked.
5. **Preferred direction.** The paper uses no path planning ($v^{pref}$
   points straight at the goal). JuPedSim only exposes
   `orientation_to_next_target`, which follows a shortest path, so agents
   near the side walls of the rooms head for the door jamb rather than
   straight at the goal behind it. The concert's goal is on the stage (an
   obstacle), and JuPedSim needs targets inside the walkable area, so the
   waypoint is at the tip of the stage bulge, (65.9, 0), 8 m in front of
   the paper's goal at (73.8, 0).
6. **Neighbour search every fine step.** The paper refreshes neighbour
   lists only every 0.1 s, for speed. JuPedSim queries every step, which
   is more accurate.

**Problems found while building and checking this (all fixed before the
results below):**

- **The ρ⁰ = 4 concert runs were not settled at the push.** All concert
  runs first started from the same 5 P/m² lattice. The two runs with
  $\rho^{0,\max} = 4$ (Fig. 7d and Fig. 8) spent the 20 s settling
  period expanding (back edge 32.7 → 26 m) and were still sloshing at
  0.13–0.20 m/s when the push started, while the baseline was at rest
  (0.03 m/s). We found this from the pre-push velocity snapshot, not from
  the mean density, which had already levelled off. Those two runs were
  redone starting from a 4 P/m² lattice.
- **The first "boundary splashing" metric compared different sets of
  agents.** Defining the boundary by SPH density (< 0.6 $\rho^{0,\max}$)
  selected 167 agents in the static-$\rho^0$ run and 4 in the dynamic
  one, because a dynamic rest density lets the edge agents compress. We
  now define the boundary geometrically (Section 3).
- **The wave-front tracker jumped to the pushers.** Taking the x of the
  largest forward velocity sometimes picked the pushed agents at the
  back instead of the wave ahead of them. The tracker now only looks
  ahead of the push rectangle (x > 45.5 m) and follows one front
  forward.
- **The two blended profiles did not use exactly the same transition.**
  RVO→SPH skipped RVO while an agent was fully in SPH mode and then
  reused a stale cached RVO acceleration when the agent left the dense
  crowd; SF→SPH had no such memory. Fixed and RVO→SPH rerun, see
  "Density-dependent blending" below.
- **Notebook size.** A first build embedded 75 MB of animations
  (`animate` draws every agent in every frame as a shape). The replays
  are now thinned in time, and for the concert restricted to a 3 m strip.
""")

# ============================================================ Blending
md(r"""
## Density-dependent blending: how SF→SPH and RVO→SPH are computed

Both blended profiles run through **one** code path in
`SPHCrowdModel.compute_next_state`. The only difference between them is
which navigation algorithm fills the low-density profile. This is
Sec. 4.4 of the paper, with the profile definitions of Sec. 5.5 and the
time steps of Secs. 5.1 and 5.6.

**Two profiles.** A *profile* is a navigation algorithm plus SPH
settings plus contact-force settings, and it produces one overall
acceleration per agent (Sec. 4.4). The blended agents have the sequence
$\{(\mathcal{P}^0, \rho^0 = 2), (\mathcal{P}^1, \rho^1 = 4)\}$ (Sec. 5.5):

| | $\mathcal{P}^0$: navigation profile | $\mathcal{P}^1$: SPH profile |
|---|---|---|
| SF→SPH | $a^0 = K^{goal}\frac{v^{pref}_i - v_i}{\tau} + \sum_j f^{av,ag}_{ij} + \sum_k f^{av,obs}_{ik}$ (Eqs. 17–20) $+ \frac{1}{m_i}\sum f^c$ with $K^{ag} = K^{obs} = 1000$ | $a^1 = K^{goal}\frac{v^{pref}_i - v_i}{\tau} + \frac{-\nabla p_i + \mu \nabla^2 v_i}{\rho_i} + \frac{1}{m_i}\sum f^c$ with $K^{ag} = 50$, $K^{obs} = 200$ |
| RVO→SPH | $a^0 = \frac{v^* - v_i}{\Delta t_{coarse}}$, $v^* = \arg\min_v \lVert v - v^{pref}_i\rVert + w/\mathrm{TTC}(i, 2v - v_i)$ (Eq. 21) $+ \frac{1}{m_i}\sum f^c$ with $K^{ag} = K^{obs} = 1000$ | identical to the row above |

These are exactly the SF (or RVO) profile and the SPH profile used on
their own in Scenario 1. $K^{goal} = 1$, $\tau = 0.5$ s, $k = 200$,
$\mu = 0$, $\rho^{0,\max} = 5$.

**One blend weight.** Every fine step (0.02 s), agent $i$ first computes
its SPH density $\rho_i$, including walls (Eqs. 9–10). This happens in
*every* profile, so $\rho_i$ is always available. Its rest density
$\rho^0_i$ (Eq. 8) and pressure are updated as well. Then

$$\kappa_i = \mathrm{clamp}\!\left(\frac{\rho_i - \rho^0}{\rho^1 - \rho^0},\, 0,\, 1\right) = \mathrm{clamp}\!\left(\frac{\rho_i - 2}{2},\, 0,\, 1\right), \qquad a_i = (1 - \kappa_i)\, a^0_i + \kappa_i\, a^1_i ,$$

which is the paper's rule: $a_i = a^0_i$ if $\rho_i \le 2$, $a_i = a^1_i$ if
$\rho_i > 4$, linear in between. The velocity and position are then
integrated once from $a_i$ (Euler, $|v_i| \le 1.8$ m/s).

**What is blended.** The whole profile acceleration, *including its
contact forces*. The paper defines a profile as navigation + SPH +
contact-force parameters and blends the profiles' overall accelerations,
so the contact stiffness also goes smoothly from $K^{ag} = 1000$ to 50
(and $K^{obs}$ from 1000 to 200) as the density rises. The goal force is
contained in both SF's $a^0$ and in $a^1$, so an SF→SPH agent keeps the
full goal force throughout. An RVO→SPH agent gets its goal-seeking from
RVO's choice of $v^*$ at low density and from the goal force at high
density.

**The one thing that differs: time steps.** Social forces are evaluated
every fine step (the paper: they "need to use the fine simulation step
size"). RVO chooses $v^*$ every $\Delta t_{coarse} = 0.1$ s, and the
resulting acceleration is reused for the 4 fine steps in between
(Sec. 5.1). The contact forces of $\mathcal{P}^0$ and everything in
$\mathcal{P}^1$ are recomputed every fine step for both. This is the
paper's own multi-rate scheme, not a choice of ours.

**A bug in our first version (fixed; RVO→SPH rerun).** To save time,
the first implementation skipped RVO whenever its weight was 0
($\rho_i \ge 4$). But RVO's result is *held* between coarse steps. So
when an agent left the dense crowd, $a^0$ was an RVO acceleration cached
the last time the agent had been below 4 P/m², possibly seconds earlier,
for up to 4 fine steps. SF has no such memory, so the two transitions
were not the same operation. Now RVO is refreshed at every coarse step
whatever $\kappa_i$ is, and $a^0$ is always the current value of its
profile, as in the paper. Skipping SF while $\kappa_i = 1$ remains, and
is exact (its weight is 0 and it has no state). SF→SPH did not change.
RVO→SPH was rerun; the before/after numbers are below.
""")

code(r"""
rho = np.linspace(0, 6, 601)
kap = blend_weight(rho)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
ax = axes[0]
ax.plot(rho, 1 - kap, label="weight of the SF / RVO profile, $1-\\kappa$")
ax.plot(rho, kap, label="weight of the SPH profile, $\\kappa$")
ax.axvline(BLEND_LOW, color="gray", ls=":"); ax.axvline(BLEND_HIGH, color="gray", ls=":")
ax.set_xlabel("SPH density $\\rho_i$ [P/m²]"); ax.set_ylabel("weight"); ax.legend(fontsize=8)
ax.set_title("Blend weights (identical for SF→SPH and RVO→SPH)", fontsize=10)
ax = axes[1]
ax.plot(rho, (1 - kap) * 1000 + kap * 50, label="effective $K^{ag}$")
ax.plot(rho, (1 - kap) * 1000 + kap * 200, label="effective $K^{obs}$")
ax.set_xlabel("SPH density $\\rho_i$ [P/m²]"); ax.set_ylabel("contact stiffness"); ax.legend(fontsize=8)
ax.set_title("Contact forces are blended with the profiles", fontsize=10)
plt.tight_layout(); plt.show()

before = {t: load_json(RESULTS / "before_blend_fix" / f"{t}.json") for t in
          ["room_RVOtoSPH_rho4", "room_RVOtoSPH_rho5", "room_RVOtoSPH_rho6", "crossing_RVOtoSPH"]}
rows = []
for t, b in before.items():
    a = load_json(RESULTS / f"{t}.json")
    key_d, key_f = ("density_15s", "flow_rate_first_to_350th") if t.startswith("room") else ("density_45s", "flow_rate_first_60s")
    rows.append({"run": t, "density before fix": b[key_d]["mean"], "density after fix": a[key_d]["mean"],
                 "flow before fix": b[key_f], "flow after fix": a[key_f]})
pd.DataFrame(rows).style.format(precision=2).hide(axis="index")
""")

# ============================================================ 1. Room evacuation
md(r"""
## 1. Scenario 1 — Room evacuation (Sec. 6.1, Table 1, Fig. 5)

| | Paper | Ours |
|---|---|---|
| Room | 20 × 20 m, 0.8 m door (Fig. 4a: 1 m thick wall) | same: x ∈ [0, 20], y ∈ [−10, 10], wall x ∈ [20, 21] |
| Agents | 400 on a 1 m grid | same, $D_i \in [0.215, 0.265]$, $m_i = (D_i/0.24)^2$ |
| Goal / removal | 1 m beyond the exit; removed within 0.5 m | exit stage = disk of radius 0.5 m around (22, 0) |
| Profiles | SF ($K^{ag}$ = 50 … 1000), RVO, SPH, SF+SPH, RVO+SPH ($\rho^{0,\max}$ = 3 … 8), SF→SPH, RVO→SPH ($\rho^{0,\max}$ = 4, 5, 6) | all 30 rows of Table 1, one OS process each |
| Metrics | #evacuated; mean (std) SPH density of all agents at 15 s; flow from the 1st to the 350th evacuee | same; flow = 349 / ($t_{350} - t_1$) |
| `dt` | 0.02 s (RVO 0.1 s) | same; runs capped at 500 s simulated |

The density at 15 s is each agent's own SPH density as computed by the
model (stored in its state), not a post-hoc estimate.
""")

code(r"""
rows = []
for _, r in PAPER_TABLE1.iterrows():
    tag = f"room_{SAFE[r.profile]}_{'k' if r.param == 'K_ag' else 'rho'}{r.value:g}"
    path = RESULTS / f"{tag}.json"
    if not path.exists():
        continue
    s = load_json(path)
    d = s["density_15s"] or {"mean": np.nan, "std": np.nan}
    rows.append({
        "profile": PRETTY[r.profile], "setting": f"{'K_ag' if r.param == 'K_ag' else 'ρ0max'}={r.value:g}",
        "evac (paper)": r.n_evac, "evac (ours)": s["n_evacuated"],
        "density (paper)": f"{r.density:.2f} [{r.density_std:.2f}]",
        "density (ours)": f"{d['mean']:.2f} [{d['std']:.2f}]",
        "flow (paper)": r.flow, "flow (ours)": s["flow_rate_first_to_350th"],
        "t_end [s]": s["sim_time_reached"], "crashed": s["crashed"] is not None,
        "_dp": r.density, "_do": d["mean"], "_fp": r.flow, "_fo": s["flow_rate_first_to_350th"],
        "_profile": r.profile, "_value": r.value, "_tag": tag,
    })
room = pd.DataFrame(rows)
room.drop(columns=[c for c in room.columns if c.startswith("_")]).style.format(
    {"flow (paper)": "{:.2f}", "flow (ours)": "{:.2f}", "t_end [s]": "{:.0f}"}, na_rep="—").hide(axis="index")
""")

code(r"""
fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
groups = [("SF / RVO: sweep of $K^{ag}$", ["SF", "RVO"], "$K^{ag}$", True),
          ("SPH, SF+SPH, RVO+SPH: sweep of $\\rho^{0,\\max}$", ["SPH", "SF+SPH", "RVO+SPH"], "$\\rho^{0,\\max}$", False),
          ("Blended SF→SPH, RVO→SPH", ["SF->SPH", "RVO->SPH"], "$\\rho^{0,\\max}$", False)]
colors = {"SF": "tab:blue", "RVO": "tab:orange", "SPH": "tab:red", "SF+SPH": "tab:green",
          "RVO+SPH": "tab:purple", "SF->SPH": "tab:olive", "RVO->SPH": "tab:brown"}
for col, (title, profiles, xlabel, logx) in enumerate(groups):
    for row, (ycol_o, ycol_p, ylabel) in enumerate([("_do", "_dp", "mean SPH density at 15 s [P/m²]"),
                                                     ("_fo", "_fp", "flow rate, 1st→350th evacuee [P/s]")]):
        ax = axes[row, col]
        for prof in profiles:
            g = room[room._profile == prof].sort_values("_value")
            ax.plot(g._value, g[ycol_o], "o-", color=colors[prof], label=f"ours: {PRETTY[prof]}")
            ax.plot(g._value, g[ycol_p], **PAPER_MARK, label=None)
            ax.plot(g._value, g[ycol_p], color="black", ls="--", alpha=0.5)
        if row == 0 and col > 0:
            xs = np.array([3, 8])
            ax.plot(xs, xs, color="gray", ls=":", lw=1.2, label="density = $\\rho^{0,\\max}$ (paper: SPH ≈ this)")
        if row == 0:
            ax.axhline(REAL_DENSITY_TYPICAL, color="tab:gray", lw=0.8, alpha=0.6)
        if row == 1:
            ax.axhspan(*REAL_FLOW_RANGE, color="tab:gray", alpha=0.15, label="real doors: 2–4 P/s (paper, [39])")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if row == 0:
            ax.set_title(title, fontsize=10)
        ax.plot([], [], **PAPER_MARK, label="paper, Table 1")
        ax.legend(fontsize=7)
plt.suptitle("Room evacuation: ours (colours) vs. the paper's Table 1 (black squares)")
plt.tight_layout()
plt.savefig(RESULTS / "plot_room_table1.png", dpi=130)
plt.show()
""")

code(r"""
from run_room_evacuation import build_geometry as room_geometry


def draw_walls(ax, walkable, bbox):
    ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], color="#b4b4b4", zorder=0))
    geoms = walkable.geoms if hasattr(walkable, "geoms") else [walkable]
    for g in geoms:
        ax.fill(*g.exterior.xy, color="white", zorder=0.5)
        for hole in g.interiors:
            ax.fill(*hole.xy, color="#b4b4b4", zorder=0.6)
    ax.set_xlim(bbox[0], bbox[2]); ax.set_ylim(bbox[1], bbox[3]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


panels = [("SF", "k", v) for v in (50, 100, 250, 500, 1000)] + [("RVO", "k", 1000)]
panels += [("SPH", "rho", v) for v in (3, 4, 5, 6, 7, 8)]
panels += [("SF+SPH", "rho", v) for v in (3, 4, 5)] + [("RVO+SPH", "rho", v) for v in (3, 4, 5)]
panels += [("SF->SPH", "rho", v) for v in (4, 5, 6)] + [("RVO->SPH", "rho", v) for v in (4, 5, 6)]
walk = room_geometry(0.8)
fig_p = plt.figure(figsize=(10, 14))
plt.imshow(plt.imread(PAPER_IMG["fig5"])); plt.axis("off")
plt.title("Paper, Fig. 5: room evacuation after 15 s, density colouring")
plt.show()
fig, axes = plt.subplots(4, 6, figsize=(13, 12.5))
for ax, (prof, kind, v) in zip(axes.ravel(), panels):
    s = load_json(RESULTS / f"room_{SAFE[prof]}_{kind}{v:g}.json")
    a = np.array(s["snapshot_15s"])
    draw_walls(ax, walk, (8.0, -10.5, 21.5, 10.5))
    ax.scatter(a[:, 0], a[:, 1], c=a[:, 2], cmap=DENSITY_CMAP, vmin=0, vmax=8, s=9, zorder=2)
    ax.set_title(f"{PRETTY[prof]}, {'$K^{ag}$' if kind == 'k' else '$\\rho^{0,max}$'}={v:g}", fontsize=8)
plt.suptitle("Ours: the same 24 panels after 15 s, density colouring (paper's Fig. 3 scale, 0–8 P/m²)")
plt.tight_layout()
plt.savefig(RESULTS / "plot_room_snapshots.png", dpi=110)
plt.show()
""")

md(r"""
**Replays.** The SPH profile with $\rho^{0,\max} = 5$ (the paper's default)
and the SF profile with $K^{ag} = 1000$ (the SF default), one frame every
5 s (SPH) or 10 s (SF, which takes over twice as long). The colour is the instantaneous speed.
""")

code(r"""
traj, area = read_sqlite_file(str(RESULTS / "room_SPH_rho5.sqlite"))
animate(traj, area, every_nth_frame=50, radius=MEAN_AGENT_RADIUS, title_note="Room evacuation, SPH, ρ0max=5")
""")

code(r"""
traj, area = read_sqlite_file(str(RESULTS / "room_SF_k1000.sqlite"))
animate(traj, area, every_nth_frame=100, radius=MEAN_AGENT_RADIUS, title_note="Room evacuation, SF, K_ag=1000")
""")

md(r"""
**Verdict, Scenario 1.**

- **SPH: reproduced, including the absolute numbers.** The mean density
  after 15 s follows $\rho^{0,\max}$ almost exactly: 3.28 / 4.21 / 5.12 /
  5.91 / 6.62 / 7.24 P/m², against the paper's 3.27 / 4.21 / 5.09 / 5.89 /
  6.61 / 7.23. Every value is within 0.03 P/m². The flow rates are within
  5% (2.95 … 7.23 vs. 3.08 … 7.20 P/s). This is the paper's central claim
  ("$\rho^{0,\max}$ is an intuitive parameter for controlling the density")
  and the one experiment that tests the SPH equations directly.
- **SF+SPH and SF→SPH: density reproduced** within 0.1 P/m² everywhere.
  SF+SPH saturates at about 4.2 P/m², the SF value, exactly as in the
  paper. Flow: SF→SPH matches except at $\rho^{0,\max} = 6$ (5.34 vs.
  4.85). SF+SPH flows at 3.3–3.4 P/s where the paper has 2.7–2.8.
- **SF: density reproduced** (4.33 → 4.21 vs. 4.39 → 4.21), and so is the
  *shape* of the flow curve: flow falls as $K^{ag}$ rises, because stiffer
  disks prolong clogs at the door. The absolute flow is **20–50% higher**
  than the paper's (3.31 … 1.95 vs. 2.70 … 1.29 P/s). Also, all 400 agents
  get out in every one of our runs. With SF (and SF+SPH) the paper
  reports 398, and says the last two agents never pass the door. We did
  not find the cause. Candidates we can name but did not isolate: our
  agents head for the door jamb (JuPedSim's routing, Implementation note
  5) rather than straight at the goal behind the wall, and our wall-slide
  safeguard (note 4) can let an agent slip along a door post where the
  paper's weaker constraint would hold it.
- **RVO-based profiles: shape reproduced, density too low.** RVO avoids
  high density "altogether" and evacuates slowly (1.18 P/s, paper 0.97).
  RVO+SPH does not respond to $\rho^{0,\max}$ at all, as the paper says.
  But our RVO crowd settles at about **2.4–2.5 P/m² where the paper
  reports 3.0–3.2**, and RVO→SPH is 0.5–0.9 P/m² below the paper for the
  same reason. We checked two obvious suspects on the RVO profile at 15 s.
  A different random seed gives 2.32 P/m² (vs. 2.47). Four times more
  candidate velocities (400 instead of 100) gives 2.62 P/m². Neither
  closes the gap to 3.23. The paper does not specify its sample count,
  how TTC treats walls, or what RVO does once two disks overlap
  (Implementation note 3). We did not tune these to match the table.
- **Snapshots (Fig. 5):** the same 24 panels look alike. SF gives a dense
  red/orange cap at the door. SPH gives a compact semicircle whose colour
  steps through the density scale with $\rho^{0,\max}$. RVO and RVO+SPH
  are sparse and spread out. The blends have a dense SPH core with a
  looser fringe.
""")

# ============================================================ 2. Crossing and bottleneck
md(r"""
## 2. Scenario 2 — Crossing and bottleneck (Sec. 6.2, Table 2, Fig. 6)

| | Paper (Fig. 4b) | Ours |
|---|---|---|
| Geometry | hall with a crossing corridor, then a 0.8 m door | hall x ∈ [−6, 20], y ∈ [−6, 6]; corridor x ∈ [4, 10], y ∈ [−10, 10]; door as in Scenario 1 (read off the 1 m grid) |
| "Brown" agents | 224 in the green rectangle, leave by the door | 14 × 16 grid in x ∈ [10, 20], y ∈ [−6, 6], ±0.1 m jitter |
| "Blue" agents | 8 every 2 s on the left, same goal | 8 at x = −5, y = −3.5 … 3.5, every 2 s |
| "Pink" agents | 4 every second at the top, moving down | 4 at y = 9, x = 4.75 … 9.25, every second; they leave at y < −9 (not stated in the paper) |
| Profiles | all 7, $\rho^{0,\max} = 5$ | same |
| Metrics | mean (std) SPH density in the green rectangle at 45 s; mean flow in the first 60 s | same, plus two crossing metrics of our own (below) |

The paper's claims about the crossing are in words: SF "is not capable
of letting all agents pass the crossing" (some pink agents are pushed
aside and get stuck against the lower walls), and SPH "does not contain
any collision avoidance, so it cannot handle the crossing". To test
them, we measure, from the trajectories:

- **blue–pink overlaps per frame** inside the corridor: pairs closer than
  0.43 m (a smaller-than-average $D_i + D_j$). Zero means the two flows
  avoid each other.
- **pink agents displaced more than 3 m sideways**, and pink agents still
  present more than 25 s after they appeared (an unobstructed crossing
  takes about 14 s).
""")

code(r"""
from run_crossing_bottleneck import build_geometry as crossing_geometry

rows = []
for _, r in PAPER_TABLE2.iterrows():
    s = load_json(RESULTS / f"crossing_{SAFE[r.profile]}.json")
    df, fps = load_trajectory(RESULTS / f"crossing_{SAFE[r.profile]}.sqlite")
    grp = {int(k): v for k, v in s["groups"].items()}
    df["g"] = df["id"].map(grp)
    tmax = df.t.max()
    first, last = df.groupby("id")["t"].min(), df.groupby("id")["t"].max()
    pink = [i for i, g in grp.items() if g == "pink" and i in first.index]
    stuck = sum(1 for i in pink if last[i] >= tmax - 0.5 and tmax - first[i] > 25)
    pk = df[df.g == "pink"]
    lateral = (pk.pos_x - pk.groupby("id")["pos_x"].transform("first")).abs().groupby(pk.id).max()
    cor = df[(df.pos_x > 4) & (df.pos_x < 10) & (df.pos_y > -6) & (df.pos_y < 6) & df.g.isin(["blue", "pink"])]
    n_ov, n_fr = 0, 0
    for _, f in cor.groupby("frame"):
        b = f[f.g == "blue"][["pos_x", "pos_y"]].to_numpy(); q = f[f.g == "pink"][["pos_x", "pos_y"]].to_numpy()
        if len(b) and len(q):
            n_ov += int((np.linalg.norm(b[:, None] - q[None], axis=2) < 0.43).sum()); n_fr += 1
    d = s["density_45s"]
    rows.append({"profile": PRETTY[r.profile],
                 "density (paper)": f"{r.density:.2f} [{r.density_std:.2f}]",
                 "density (ours)": f"{d['mean']:.2f} [{d['std']:.2f}]",
                 "flow (paper)": r.flow, "flow (ours)": s["flow_rate_first_60s"],
                 "blue–pink overlaps / frame": n_ov / max(n_fr, 1),
                 "pink displaced > 3 m": int((lateral > 3).sum()), "pink stuck > 25 s": stuck,
                 "pink spawned": len(pink), "crashed": s["crashed"] is not None,
                 "_dp": r.density, "_do": d["mean"], "_fp": r.flow, "_fo": s["flow_rate_first_60s"]})
cross = pd.DataFrame(rows)
cross.drop(columns=[c for c in cross.columns if c.startswith("_")]).style.format(
    {"flow (paper)": "{:.2f}", "flow (ours)": "{:.2f}", "blue–pink overlaps / frame": "{:.2f}"}).hide(axis="index")
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
x = np.arange(len(cross))
for ax, (o, p, lab) in zip(axes[:2], [("_do", "_dp", "mean SPH density in green rectangle at 45 s [P/m²]"),
                                      ("_fo", "_fp", "mean flow in first 60 s [P/s]")]):
    ax.bar(x, cross[o], color="tab:blue", alpha=0.8, label="ours")
    ax.plot(x, cross[p], **PAPER_MARK, label="paper, Table 2")
    for xi, v in zip(x, cross[o]):
        ax.text(xi, v + 0.05, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, cross["profile"], rotation=30); ax.set_ylabel(lab, fontsize=9); ax.legend(fontsize=8)
ax = axes[2]
ax.bar(x, cross["blue–pink overlaps / frame"], color="tab:red", alpha=0.8, label="ours")
ax.axhline(0, lw=1.5, label="collision-free crossing (reference)", **PAPER_STYLE)
ax.set_xticks(x, cross["profile"], rotation=30)
ax.set_ylabel("blue–pink overlaps per frame in the corridor", fontsize=9)
ax.set_title("paper: SPH 'cannot handle the crossing'", fontsize=9); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(RESULTS / "plot_crossing_table2.png", dpi=130)
plt.show()
""")

code(r"""
plt.figure(figsize=(14, 7.5))
plt.imshow(plt.imread(PAPER_IMG["fig6"])); plt.axis("off")
plt.title("Paper, Fig. 6: after 45 s, velocity colouring")
plt.show()
walk = crossing_geometry(0.8)
fig, axes = plt.subplots(2, 4, figsize=(19, 7.2))
panels = [("SF", "(a) SF"), ("RVO", "(b) RVO"), ("SPH", "(c) SPH"), ("SF+SPH", "SF+SPH (not in the paper's Fig. 6)"),
          ("RVO+SPH", "(d) RVO+SPH"), ("SF->SPH", "(e) SF→SPH"), ("RVO->SPH", "(f) RVO→SPH")]
for ax, (prof, title) in zip(axes.ravel(), panels):
    s = load_json(RESULTS / f"crossing_{SAFE[prof]}.json")
    a = np.array([r[:5] for r in s["snapshot_45s"]], dtype=float)
    draw_walls(ax, walk, (-6.5, -10.5, 22.5, 10.5))
    ax.scatter(a[:, 0], a[:, 1], c=velocity_colors(a[:, 2], a[:, 3]), s=14, zorder=2)
    ax.set_title(title, fontsize=10)
# colour key (Fig. 3 left): hue = direction, saturation = speed (full at 0.75 m/s)
ax = axes.ravel()[-1]
g = np.linspace(-0.9, 0.9, 181)
VX, VY = np.meshgrid(g, g)
key = velocity_colors(VX, VY)
key[np.hypot(VX, VY) > 0.9] = 1.0
ax.imshow(key, origin="lower", extent=(-0.9, 0.9, -0.9, 0.9))
ax.set_title("colour key: velocity (m/s)", fontsize=10); ax.set_xlabel("$v_x$"); ax.set_ylabel("$v_y$"); ax.grid(False)
plt.suptitle("Ours after 45 s, velocity colouring: the paper's six profiles (a)–(f), plus SF+SPH")
plt.tight_layout()
plt.savefig(RESULTS / "plot_crossing_snapshots.png", dpi=110)
plt.show()
""")

md(r"""
**Where the blended agents switch behaviour.** The two blended profiles
use the same transition (see "Density-dependent blending" above). Here
each agent is coloured by its SPH weight
$\kappa_i = \mathrm{clamp}((\rho_i - 2)/2, 0, 1)$ at 45 s, computed from
the SPH density the model itself used in that step: grey = pure
navigation (SF or RVO), dark red = pure SPH. The histograms count agents
in the two pure modes and in between.
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1, 1, 0.8]})
kap_stats = {}
for ax, prof in zip(axes[:2], ["SF->SPH", "RVO->SPH"]):
    s = load_json(RESULTS / f"crossing_{SAFE[prof]}.json")
    a = np.array([r[:5] for r in s["snapshot_45s"]], dtype=float)
    kap = blend_weight(a[:, 4])
    draw_walls(ax, walk, (-6.5, -10.5, 22.5, 10.5))
    sc = ax.scatter(a[:, 0], a[:, 1], c=kap, cmap="Reds", vmin=-0.15, vmax=1.0, s=14, zorder=2,
                    edgecolors="k", linewidths=0.2)
    ax.set_title(f"{PRETTY[prof]} at 45 s, coloured by SPH weight κ", fontsize=10)
    kap_stats[PRETTY[prof]] = {"navigation only (κ = 0)": int((kap == 0).sum()),
                               "blending (0 < κ < 1)": int(((kap > 0) & (kap < 1)).sum()),
                               "SPH only (κ = 1)": int((kap == 1).sum())}
fig.colorbar(sc, ax=axes[:2], label="κ (0 = SF/RVO profile, 1 = SPH profile)", shrink=0.8)
ks = pd.DataFrame(kap_stats).T
ks.plot.bar(ax=axes[2], stacked=True, color=["0.7", "salmon", "darkred"], rot=0)
axes[2].set_ylabel("agents at 45 s"); axes[2].set_ylim(0, 380); axes[2].legend(fontsize=8, loc="upper center", ncol=1)
plt.savefig(RESULTS / "plot_crossing_blend_weight.png", dpi=110)
plt.show()
ks
""")

md(r"""
**Replays**, one frame every 3 s: RVO→SPH (the paper's recommended
profile), SF→SPH (the same transition with social forces), and SF+SPH
(social forces and SPH always added together, no transition).
""")

code(r"""
traj, area = read_sqlite_file(str(RESULTS / "crossing_RVOtoSPH.sqlite"))
animate(traj, area, every_nth_frame=30, radius=MEAN_AGENT_RADIUS, title_note="Crossing and bottleneck, RVO→SPH")
""")

code(r"""
traj, area = read_sqlite_file(str(RESULTS / "crossing_SFtoSPH.sqlite"))
animate(traj, area, every_nth_frame=30, radius=MEAN_AGENT_RADIUS, title_note="Crossing and bottleneck, SF→SPH")
""")

code(r"""
traj, area = read_sqlite_file(str(RESULTS / "crossing_SFplusSPH.sqlite"))
animate(traj, area, every_nth_frame=30, radius=MEAN_AGENT_RADIUS, title_note="Crossing and bottleneck, SF+SPH")
""")

md(r"""
**Verdict, Scenario 2.**

- **Table 2: ranking reproduced, numbers close.** The profiles with SPH
  at the bottleneck and no collision-avoidance limit (SPH, SF→SPH,
  RVO→SPH) have the highest flow: 4.38 / 4.37 / 4.00 P/s vs. the paper's
  4.72 / 4.56 / 4.49. RVO and RVO+SPH have the lowest: 1.38 / 1.47 vs.
  1.18 / 1.71. Densities are within 0.2 P/m² for SF, SPH, SF+SPH and
  SF→SPH. For the RVO-based profiles they are again 0.3–0.5 lower, as in
  Scenario 1.
- **"SPH cannot handle the crossing": reproduced.** SPH is the only
  profile whose blue and pink agents overlap in the corridor (about 1.75
  overlapping pairs per frame; every other profile has 0), and it shoves
  40 pink agents more than 3 m sideways. RVO→SPH crosses cleanly and then
  compacts at the door, the combination the paper recommends.
- **"SF is not capable of letting all agents pass the crossing": not
  reproduced.** In the paper's Fig. 6(a), some pink agents end up stuck
  against the lower walls. In ours no pink agent is displaced more than
  3 m, none is still present 25 s after appearing, and SF's crossing is
  as clean as RVO's by these measures. One likely reason is that our pink
  agents leave the scene at the bottom (y < −9). The paper does not say
  what happens to them, and agents that stay in the corridor would pile
  up and be pushed aside.
- **SF+SPH (a static combination, no transition):** density 3.85 P/m²
  (paper 3.81), flow 3.25 P/s (paper 2.85). Like SF, it crosses cleanly
  (no overlaps, no pink agent pushed aside). The paper says that adding
  SPH to a collision-avoidance model *statically* lowers the density at
  the door slightly and raises the flow, but $\rho^{0,\max}$ is not
  reached. We see the same: SF+SPH vs. SF is 3.85 vs. 3.97 P/m² and
  3.25 vs. 2.15 P/s, and RVO+SPH vs. RVO is 2.76 vs. 2.84 P/m² and 1.47
  vs. 1.38 P/s. Both densities stay far below 5.
- **The two blends, same transition:** both cross cleanly (0 overlaps) and
  then compact at the door. At 45 s, SF→SPH has 108 agents fully in SPH
  mode, 30 blending and 134 on SF alone. RVO→SPH has 76 / 45 / 168: RVO
  keeps the queue looser, so fewer agents pass $\rho_i = 4$. That is why
  RVO→SPH's density (3.29 vs. paper 3.83) and flow (4.00 vs. 4.49) are
  below SF→SPH's (4.10, 4.37), the same RVO shortfall as in Scenario 1.
- **Snapshots (Fig. 6):** the same structure: columns of blue agents on
  the left, a diagonal stream through the crossing, a queue at the door.
  The RVO and RVO+SPH bottleneck crowds are loose and multicoloured (slow,
  milling), and the SPH-based ones are compact.
""")

# ============================================================ 3. Concert
md(r"""
## 3. Scenario 3 — Concert with a shockwave (Sec. 6.3, Figs. 7–10)

| | Paper (Fig. 4c) | Ours |
|---|---|---|
| Venue | about 69 × 68 m, stage on the right with a curved front | x ∈ [0, 69], y ∈ [−34, 34]; stage = x > 69 plus a circular bulge (centre (102.5, 0), radius 36.4 m: 28.5 m wide, 2.9 m deep), read off the 1 m / 10 m grid |
| Crowd | 10,000 agents added in rows of 100/s for 100 s; converge until t = 150 s | **the same 10,000 agents placed directly** on a hexagonal lattice around the goal, then **20 s** of settling. At 5 P/m² the lattice fills exactly the region the paper's crowd occupies at 150 s (Fig. 7: back edge = arc of radius ≈ 40.4 m around the goal). The two $\rho^0 = 4$ runs start at 4 P/m² (see Implementation notes) |
| Profile | SPH with $K^{goal} = 0.1$; baseline $k = 200$, $\mu = 5$, $\rho^{0,\max} = 5$ | same |
| Push | agents in a 1 × 20 m rectangle: $K^{goal} = 1$, no contact or SPH forces, 0.5 s | same rectangle, x ∈ [43.8, 45.0], y ∈ [−10, 10] (≈120 agents) |
| Variants | Fig. 7: $k$ = 50 / 500, $\rho^{0,\max}$ = 4 / 6, $\mu$ = 0; Fig. 8: static $\rho^0 = 4$; Fig. 9: no SPH with $K^{ag}$ = 200 / 1000; Fig. 10: $K^{ag} = 0$ | all 10 as separate runs |

**Why we shortened the start.** At 10,000 agents, one step costs about
4–6 s of wall-clock time in this pure-Python callback (vs. 4.94 ms in
the paper's multithreaded C++, Section 4 below). Filling and converging
for 150 s would take about 10 hours per run, for ten runs. Starting from
the converged shape and settling for 20 s costs about 2 hours per run. The
agent count, geometry, density, parameters and push are unchanged. What
changes is the history: our crowd did not walk in, so there are no
internal velocity patterns left over from the approach. The density
series below shows the crowd has settled before the push.

**What the paper claims, and how we check it:**

1. A clearly visible wave moves toward the stage, bounces off and moves
   back (all variants). → Snapshots at the same six times as Fig. 7, and
   a space–time map of the mean forward velocity $\bar v_x(x, t)$ in the
   band |y| < 10 m. The forward wave front is the x of maximum $\bar v_x$.
2. Higher $k$ gives a faster, thicker wave. → Front position vs. time,
   compared with positions read off Fig. 7.
3. $\rho^{0,\max}$ sets the density but barely affects the wave. →
   Crowd density before the push, and the fronts.
4. Without viscosity, agents are jittery after the wave (Fig. 7f). → Mean
   agent speed 6 s after the push.
5. A static rest density makes boundary agents "splash" (Fig. 8). → Mean
   speed of boundary agents (SPH density < 0.6 $\rho^{0,\max}$) over the
   5 s before the push.
6. Without SPH, the result depends on $K^{ag}$ and agents shake (Fig. 9);
   without contact forces the boundary deforms (Fig. 10). → Density,
   post-wave speed and overlaps.

**Reading positions off Fig. 7.** Page 12 was rendered at 300 dpi. In
each of the paper's panels, pixels with saturated red/magenta hue
(= moving toward the stage in the Fig. 3 colour wheel) were counted per
pixel column inside the band of the push rectangle, and the column with
the most such pixels is the wave front. The scale comes from the pushed
rectangle itself (20 m = 147 px; centred at x = 44.4 m). Checked
visually by drawing the detected positions back onto the figure. The
error is about ±1 m. The backward wave is too pale to locate reliably, so
we only note in which snapshot it first appears.
""")

code(r"""
CONCERT_RUNS = {
    "baseline": "(a) baseline: ρ0max=5, k=200, μ=5", "k50": "(b) k=50", "k500": "(c) k=500",
    "rho4": "(d) ρ0max=4", "rho6": "(e) ρ0max=6", "mu0": "(f) μ=0",
    "static_rho4": "Fig. 8: static ρ0=4", "noSPH_kag200": "Fig. 9a: no SPH, K_ag=200",
    "noSPH_kag1000": "Fig. 9b: no SPH, K_ag=1000", "kag0": "Fig. 10: K_ag=0",
}
REL_TIMES = [0.3, 1.0, 2.0, 3.0, 4.0, 6.0]
concert, snaps = {}, {}
for key in CONCERT_RUNS:
    p = RESULTS / f"concert_{key}.json"
    if p.exists():
        concert[key] = load_json(p)
        snaps[key] = dict(np.load(RESULTS / f"concert_{key}_snapshots.npz"))


def snap(key, rel):
    return snaps[key][f"t{rel:.2f}"]


rows = []
for key, s in concert.items():
    pre = snap(key, 0.0)
    post = snap(key, 6.0)
    region = (pre[:, 0] > 35) & (np.abs(pre[:, 1]) < 20)
    rows.append({"run": CONCERT_RUNS[key], "k": s["k_gas"], "μ": s["mu"], "ρ0 range": f"[{s['rho0_min']:g}, {s['rho0_max']:g}]",
                 "K_ag": s["k_ag"], "K_obs": s["k_obs"], "pushed": s["n_pushed"],
                 "density before push (mean)": pre[:, 4].mean(),
                 "density before push, |y|<20, x>35": pre[region, 4].mean(),
                 "90th pct": np.percentile(pre[:, 4], 90),
                 "mean speed 6 s after push [m/s]": np.hypot(post[:, 2], post[:, 3]).mean(),
                 "crashed": s["crashed"] is not None, "wall clock [h]": s["wall_clock_s"] / 3600})
concert_table = pd.DataFrame(rows)
concert_table.style.format(precision=2).hide(axis="index")
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 4))
for key, s in concert.items():
    ser = np.array(s["density_series"])
    ser = ser[ser[:, 1] > 0]
    ax.plot(ser[:, 0] - s["t_push"], ser[:, 1], label=CONCERT_RUNS[key])
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("time relative to the push [s]"); ax.set_ylabel("mean SPH density of all agents [P/m²]")
ax.set_title("Settling before the push (we start from the converged shape, see above)")
ax.legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.show()
""")

code(r"""
plt.figure(figsize=(12, 14))
plt.imshow(plt.imread(PAPER_IMG["fig7"])); plt.axis("off")
plt.title("Paper, Fig. 7: t = 150.3, 151, 152, 153, 154, 156 s (push at 150 s), velocity colouring")
plt.show()

from run_concert import build_geometry as concert_geometry
walk_c = concert_geometry(34.0)
BOX_C = (28.0, -27.0, 70.5, 27.0)


def wave_grid(keys, fname, title):
    fig, axes = plt.subplots(len(keys), 6, figsize=(13, 2.95 * len(keys)))
    axes = np.atleast_2d(axes)
    for r, key in enumerate(keys):
        for c, rel in enumerate(REL_TIMES):
            ax = axes[r, c]
            a = snap(key, rel)
            draw_walls(ax, walk_c, BOX_C)
            ax.scatter(a[:, 0], a[:, 1], c=velocity_colors(a[:, 2], a[:, 3]), s=0.6, zorder=2)
            if c == 0:
                ax.set_ylabel(CONCERT_RUNS[key], fontsize=8)
                ax.add_patch(plt.Rectangle((43.8, -10), 1.2, 20, fill=False, color="red", lw=0.6, zorder=3))
            if r == 0:
                ax.set_title(f"push + {rel:g} s", fontsize=9)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(RESULTS / fname, dpi=110)
    fig.set_dpi(60)  # inline copy at lower resolution; full size in the saved PNG
    plt.show()


wave_grid([k for k in ["baseline", "k50", "k500", "rho4", "rho6", "mu0"] if k in snaps], "plot_concert_fig7.png",
          "Ours: same rows and times as Fig. 7, velocity colouring")
""")

code(r"""
def forward_front(t, x, V, x_min=45.5, thr=0.2):
    '''Leading forward wave: x of max mean v_x ahead of the pushed rectangle,
    followed from push + 0.3 s until it fades (< thr m/s) or reaches the stage.'''
    ahead = x > x_min
    pts = []
    for ti, row in zip(t, V):
        if ti < 0.3:
            continue
        r = np.where(ahead, row, np.nan)
        if not np.isfinite(r).any() or np.nanmax(r) < thr:
            if pts:
                break
            continue
        xf = x[np.nanargmax(r)]
        if pts and xf < pts[-1][1] - 1.0:  # the forward front never moves back
            break
        pts.append((ti, xf))
        if xf > 63.0:
            break
    return np.array(pts)


def kymograph(key, band=10.0, bins=np.arange(28.0, 70.0, 1.0), t_window=(-0.5, 6.5)):
    s = concert[key]
    df, fps = load_trajectory(RESULTS / f"concert_{key}.sqlite")
    t0 = s["t_push"]
    df = df[(df.t >= t0 + t_window[0] - 0.2) & (df.t <= t0 + t_window[1] + 0.2)]
    df = add_velocity(df)
    df = df[(df.pos_y.abs() < band) & df.vx.notna() & (df.t >= t0 + t_window[0]) & (df.t <= t0 + t_window[1])]
    df["xb"] = np.digitize(df.pos_x, bins) - 1
    df = df[(df.xb >= 0) & (df.xb < len(bins) - 1)]
    grid = df.pivot_table(index="frame", columns="xb", values="vx", aggfunc="mean")
    times = grid.index.to_numpy() / fps - t0
    centres = 0.5 * (bins[1:] + bins[:-1])[grid.columns.to_numpy()]
    return times, centres, grid.to_numpy()


kymo = {key: kymograph(key) for key in concert}

show = [k for k in ["baseline", "k50", "k500", "rho4", "rho6", "mu0", "noSPH_kag200", "noSPH_kag1000"] if k in kymo]
fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), sharex=True, sharey=True)
for ax, key in zip(axes.ravel(), show):
    t, x, V = kymo[key]
    im = ax.pcolormesh(x, t, V, cmap="RdBu_r", vmin=-0.8, vmax=0.8, shading="nearest")
    # our forward front: x of max mean vx per frame, while the wave is moving forward
    front = forward_front(t, x, V)
    if len(front):
        ax.plot(front[:, 1], front[:, 0], "-", color="gold", lw=1.5, label="ours: forward front")
    ref = PAPER_FIG7_FRONT.get(key)
    if ref:
        ax.plot(ref["x"], ref["t"], "s--", color="black", mfc="none", label="paper, Fig. 7 (read off)")
    ax.axvline(66.1, color="gray", lw=0.8)
    ax.set_title(CONCERT_RUNS[key], fontsize=9)
    ax.legend(fontsize=6, loc="upper left")
for ax in axes[1]:
    ax.set_xlabel("x [m] (stage tip at 66.1)")
for ax in axes[:, 0]:
    ax.set_ylabel("time since push [s]")
fig.colorbar(im, ax=axes, label="mean $v_x$ in |y|<10 m [m/s] (red = toward stage)", shrink=0.8)
plt.suptitle("Space-time maps of the shockwave: forward (red) and returning (blue) motion")
plt.savefig(RESULTS / "plot_concert_kymographs.png", dpi=110)
plt.show()
""")

code(r"""
def front_speed(key):
    p = forward_front(*kymo[key])
    p = p[p[:, 1] < 62.0] if len(p) else p  # before it reaches the stage
    if len(p) < 3:
        return np.nan
    return np.polyfit(p[:, 0], p[:, 1], 1)[0]


def paper_speed(key):
    ref = PAPER_FIG7_FRONT.get(key)
    if not ref or len(ref["t"]) < 2:
        return np.nan
    return np.polyfit(ref["t"], ref["x"], 1)[0]


def return_time(key):
    '''First time (after 1 s) at which the band next to the stage (x in 58-65 m) moves back at < -0.2 m/s.'''
    t, x, V = kymo[key]
    near = (x > 58) & (x < 65)
    back = [ti for ti, row in zip(t, V) if ti > 1.0 and np.nanmean(row[near]) < -0.2]
    return back[0] if back else np.nan


wave_df = pd.DataFrame([{"run": CONCERT_RUNS[k], "front speed, ours [m/s]": front_speed(k),
                         "front speed, paper Fig. 7 [m/s]": paper_speed(k),
                         "return near stage, ours [s after push]": return_time(k),
                         "return visible in paper at [s]": PAPER_FIG7_RETURN.get(k)}
                        for k in kymo])
wave_df.style.format(precision=1, na_rep="—").hide(axis="index")
""")

md(r"""
**Verdict, the wave (Fig. 7): reproduced, including its speed.**

- **To the stage, a bounce, and back:** every SPH variant shows the same
  sequence as the paper. A red band leaves the push rectangle, reaches the
  stage about 2 s later, turns into a cyan band moving back, and crosses
  the crowd back to its rear by about push + 5–6 s (the blue diagonal in the
  space–time maps). The snapshots match Fig. 7 panel by panel, including
  the red trail of the pushers behind the wave and the curved shape of the
  front.
- **Speed vs. $k$:** the forward front moves at 5.9 / 9.6 / 14.1 m/s for
  $k$ = 50 / 200 / 500. The positions read off Fig. 7 give 6.0 / 9.8 /
  12.9 m/s. For k = 500 the paper only has two usable snapshots before the
  wave reaches the stage. It returns from the stage at about push + 2.7 s
  (baseline) and 1.9 s (k = 500). Fig. 7 first shows it moving back in
  the push + 3 s and push + 2 s snapshots respectively. With k = 50, in
  both the paper and ours, it is still on its way at 3 s. The wave is
  also visibly thicker for large $k$ (row c), as the paper says.
- **$\rho^{0,\max}$ sets the density, not the wave:** the crowd density
  before the push is 4.17 / 5.11 / 5.91 P/m² for $\rho^{0,\max}$ = 4 / 5 /
  6, and the front speed is 9.1 / 9.6 / 10.0 m/s (paper Fig. 7: 9.2 / 9.8
  / 9.6). Reproduced.
- **"If k is too low, the density far exceeds $\rho^{0,\max}$":** with
  k = 50 the density is 5.49 P/m² (90th percentile 6.09) at
  $\rho^{0,\max} = 5$, vs. 5.11 (5.31) at k = 200. Reproduced.
""")

code(r"""
# Fig. 8: static vs dynamic rest density -- boundary 'splashing'
fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [1.7, 1, 1]})
axes[0].imshow(plt.imread(PAPER_IMG["fig8"])); axes[0].axis("off")
axes[0].set_title("Paper, Fig. 8: static ρ0=4 (left) vs dynamic ρ0 ∈ [0, 4] (right)", fontsize=9)
def back_edge_agents(a, width=1.5, bin_deg=2.0):
    '''uids of agents within `width` m of the crowd's free back edge: in 2-degree
    sectors around the goal, those whose distance to the goal is within `width`
    of the sector's maximum. Only sectors whose outermost agent is away from
    the side walls (|y| < 30) count.'''
    dx, dy = a[:, 0] - 73.8, a[:, 1]
    r, th = np.hypot(dx, dy), np.degrees(np.arctan2(dy, dx))
    sec = np.floor(th / bin_deg).astype(int)
    keep = []
    for sid in np.unique(sec):
        m = np.where(sec == sid)[0]
        k = m[np.argmax(r[m])]
        if abs(a[k, 1]) < 30:
            keep += list(m[r[m] > r[k] - width])
    return set(a[keep, 5].astype(int))


jit = {}
for ax, key in zip(axes[1:], ["static_rho4", "rho4"]):
    if key not in snaps:
        continue
    b = snap(key, 0.0)
    df, fps = load_trajectory(RESULTS / f"concert_{key}.sqlite")
    t0 = concert[key]["t_push"]
    df = add_velocity(df[(df.t >= t0 - 5.2) & (df.t <= t0 + 0.05)])
    df = df[(df.t >= t0 - 5) & (df.t <= t0) & df.vx.notna()]
    uid_of = {int(k): v for k, v in concert[key]["uid_of"].items()}
    edge = back_edge_agents(snap(key, -5.0))
    is_b = df["id"].map(uid_of).isin(edge)
    sp = np.hypot(df.vx, df.vy)
    jit[CONCERT_RUNS[key] if key != "rho4" else "dynamic ρ0 ∈ [0, 4] (= row d)"] = {
        "back-edge agents": len(edge), "back-edge mean speed [m/s]": sp[is_b].mean(),
        "back-edge 90th pct speed [m/s]": sp[is_b].quantile(0.9), "rest of crowd mean speed [m/s]": sp[~is_b].mean()}
    draw_walls(ax, walk_c, (26.0, -8.0, 42.0, 12.0))
    ax.scatter(b[:, 0], b[:, 1], c=velocity_colors(b[:, 2], b[:, 3]), s=14, zorder=2)
    ax.set_title(f"Ours: {CONCERT_RUNS[key] if key != 'rho4' else 'dynamic ρ0 ∈ [0, 4]'} (just before push)", fontsize=9)
plt.tight_layout(); plt.savefig(RESULTS / "plot_concert_fig8.png", dpi=110); plt.show()
pd.DataFrame(jit).T.style.format(precision=3)
""")

code(r"""
# Fig. 9: no SPH, contact forces only (K_ag = 200 / 1000)
plt.figure(figsize=(13, 5.5))
plt.imshow(plt.imread(PAPER_IMG["fig9"])); plt.axis("off")
plt.title("Paper, Fig. 9: no SPH, K_ag = 200 (top) and 1000 (bottom), K_obs = 1000")
plt.show()
wave_grid([k for k in ["noSPH_kag200", "noSPH_kag1000"] if k in snaps], "plot_concert_fig9.png",
          "Ours: no SPH (k = 0, μ = 0), same times as Fig. 9")
""")

code(r"""
# Fig. 10: K_ag = 0 -- overlaps and the boundary of the crowd
def overlap_stats(a, walk, D=0.24, deep=0.1):
    '''Pairs of agents closer than 2D (D = mean radius), per agent; the share
    of pairs overlapping by more than `deep` m; the same for agents within
    1.5 m of a wall or the stage.'''
    from scipy.spatial import cKDTree
    pairs = cKDTree(a[:, :2]).query_pairs(2 * D, output_type="ndarray")
    depth = 2 * D - np.linalg.norm(a[pairs[:, 0], :2] - a[pairs[:, 1], :2], axis=1)
    near = shapely.distance(walk.boundary, shapely.points(a[:, 0], a[:, 1])) < 1.5
    pn = near[pairs[:, 0]] | near[pairs[:, 1]]
    return {"pairs closer than 2D, per agent": len(pairs) / len(a),
            "pairs overlapping > 0.1 m, per agent": (depth > deep).sum() / len(a),
            "same, agents ≤ 1.5 m from a wall": (depth[pn] > deep).sum() / max(near.sum(), 1),
            "deepest overlap [m]": float(depth.max()) if len(depth) else 0.0}


fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [1.6, 1, 1]})
axes[0].imshow(plt.imread(PAPER_IMG["fig10"])); axes[0].axis("off")
axes[0].set_title("Paper, Fig. 10: K_ag = 0 deforms the crowd boundary", fontsize=9)
ov = {}
for ax, key in zip(axes[1:], ["baseline", "kag0"]):
    if key not in snaps:
        continue
    a = snap(key, 6.0)
    ov[CONCERT_RUNS[key]] = overlap_stats(a, walk_c)
    draw_walls(ax, walk_c, (40.0, 14.0, 70.5, 34.5))
    ax.scatter(a[:, 0], a[:, 1], c=velocity_colors(a[:, 2], a[:, 3]), s=5, zorder=2)
    ax.set_title(f"Ours: {CONCERT_RUNS[key]}, upper corner, push + 6 s", fontsize=9)
plt.tight_layout(); plt.savefig(RESULTS / "plot_concert_fig10.png", dpi=110); plt.show()
pd.DataFrame(ov).T.style.format(precision=3)
""")

md(r"""
**Verdict, stability (Figs. 7f, 8, 9, 10).**

- **Viscosity (Fig. 7f):** without it ($\mu = 0$), the same wave still
  travels at the same speed (9.7 m/s), but the crowd is **3.4× more
  agitated** 6 s after the push (mean speed 0.16 vs. 0.05 m/s). The
  snapshots show the paper's speckled "jittery" look, strongest at the
  stage. Reproduced.
- **Dynamic rest density (Fig. 8):** with a static $\rho^0 = 4$, the
  agents at the crowd's back edge move at 0.20 m/s on average (90th
  percentile 0.34) while the interior is at rest (0.05). With the dynamic
  rest density they move at 0.06 m/s (0.11), no faster than the interior.
  The snapshots show the paper's picture: a speckled, splashing edge vs.
  a calm one. Reproduced.
- **No SPH, contact forces only (Fig. 9):** a wave still forms. But
  $K^{ag}$ changes the density (5.36 → 4.83 P/m² from $K^{ag}$ = 200 to
  1000) *and* the wave speed (6.4 → 12.9 m/s) together, and the crowd
  keeps shaking after the wave (0.08–0.16 m/s, vs. 0.05 with SPH). That
  is the paper's argument, in numbers. Reproduced.
- **No inter-agent contact forces (Fig. 10):** the crowd overlaps
  noticeably more, above all against walls and the stage. Pairs
  overlapping by more than 0.1 m rise from 0.003 to 0.16 per agent next
  to walls, and the deepest overlap from 0.10 to 0.27 m. A group of agents
  at the top wall is squeezed out of the crowd's smooth outline, like
  the bump in the paper's picture. Reproduced.
- **No run crashed.** The wall-slide safeguard (Implementation note 4) is
  needed so that JuPedSim does not abort. We did not count how often it
  fired.
""")

md(r"""
**Replay** (baseline). Only a 3 m-wide strip through the push region
(|y| < 1.5 m, x > 30 m, about 500 agents) is animated, from 0.5 s before
the push to 6.5 s after it, at 0.3 s per frame. `animate` draws every
agent in every frame as a separate shape, so all 10,000 agents would make
the notebook far too large.
""")

code(r"""
_t0 = concert["baseline"]["t_push"]
sub_animation(RESULTS / "concert_baseline.sqlite", lambda d: (d.pos_y.abs() < 1.5) & (d.pos_x > 30), every_nth_frame=3,
              title="Concert shockwave (strip |y|<1.5 m)", t_window=(_t0 - 0.5, _t0 + 6.5))
""")

# ============================================================ 4. Performance
md(r"""
## 4. Performance (Sec. 6.4, Fig. 11)

**Paper:** the concert scenario at the baseline settings, frame time for
5,000–70,000 agents on a 6-thread Intel i7-7920HQ in C++ (UMANS).
Numbers: 4.94 ms at 10,000 agents, 18.57 ms at 30,000, both below the
20 ms real-time budget of $\Delta t_{fine}$.

**Ours:** the same concert geometry and baseline settings, N = 1,000,
2,500, 5,000 and 10,000 agents at 5 P/m² (the N agents closest to the
goal on the same lattice), 1 s of simulation each. We report the mean
wall-clock time of `sim.iterate()`, one run at a time on an otherwise
idle machine (load average about 1.5 on 12 cores; single-threaded, since
the Python callback holds the GIL). Only the SPH profile is timed, as in
the paper. The diamond is the full baseline concert run during the wave,
measured while 11 other simulations shared the machine, so it is slower.
""")

code(r"""
perf = []
for n in (1000, 2500, 5000, 10000):
    p = RESULTS / f"perf_n{n}.json"
    if p.exists():
        s = load_json(p)
        perf.append({"N": s["n_agents"], "ms per step (ours)": s["mean_step_time_ms"],
                     "µs per agent-step (ours)": 1000 * s["mean_step_time_ms"] / s["n_agents"]})
perf = pd.DataFrame(perf)
full_runs = [(concert[k]["n_agents"], concert[k]["wave_step_time_ms"]) for k in ("baseline",) if k in concert]

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axes[0]
ax.plot(PAPER_FIG11["n"], PAPER_FIG11["frame_ms"], "x--", color="black", label="paper, Fig. 11 'Total' (digitized)")
ax.plot(list(PAPER_FIG11_TEXT), list(PAPER_FIG11_TEXT.values()), "o", color="black", mfc="none", ms=9,
        label="paper, Sec. 6.4 text (4.94, 18.57 ms)")
ax.plot(perf["N"], perf["ms per step (ours)"], "o-", color="tab:red", label="ours, 1 s timing runs")
for n, t in full_runs:
    ax.plot(n, t, "D", color="tab:red", mfc="white", label="ours, baseline run (wave phase)")
ax.axhline(1000 * PAPER_DT_FINE, color="gray", ls=":", lw=1)
ax.text(900, 1000 * PAPER_DT_FINE * 1.15, "20 ms = real time at Δt_fine", ha="left", fontsize=7, color="gray")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("number of agents"); ax.set_ylabel("mean time per simulation step [ms] (log)")
ax.set_title("Frame time vs. number of agents"); ax.legend(fontsize=7)
axes[1].imshow(plt.imread(PAPER_IMG["fig11"])); axes[1].axis("off"); axes[1].set_title("Paper, Fig. 11", fontsize=9)
plt.tight_layout(); plt.savefig(RESULTS / "plot_performance.png", dpi=130); plt.show()
if len(perf):
    slope = np.polyfit(np.log(perf["N"]), np.log(perf["ms per step (ours)"]), 1)[0]
    print(f"log-log slope of our frame time vs N: {slope:.2f} (1 = linear)")
    t10k = perf.loc[perf.N == perf.N.max(), "ms per step (ours)"].iloc[0]
    print(f"at N = {perf.N.max()}: ours {t10k:.0f} ms/step vs paper {PAPER_FIG11_TEXT[10000]} ms "
          f"-> {t10k / PAPER_FIG11_TEXT[10000]:.0f}x slower")
perf
""")

md(r"""
**Verdict, performance:** the *shape* of Fig. 11 is reproduced, the
absolute speed is not. Frame time is linear in N (log-log slope 0.98,
about 160 µs per agent-step at every N), as in the paper, where it is
also linear (≈ 0.7 µs per agent-step on six threads). At 10,000 agents
one step takes about 1.6 s against the paper's 4.94 ms, **about 320×
slower**, and nowhere near real time (20 ms). That is not a hardware
difference: our machine (Apple M3 Pro, 2023) is newer than the paper's
Intel i7-7920HQ (2017), so the gap would only be larger on equal hardware. It is
the cost of a pure-Python callback, called once per agent per step on a
single thread, that also recomputes each neighbour's density (note 1).
The paper's claim of real-time performance at tens of thousands of
agents is therefore **not** reproduced by this implementation, and the
concert runs had to be shortened because of it.
""")

# ============================================================ Summary
md(r"""
## 5. Summary

| # | Phenomenon (paper) | Reproduced? | Notes |
|---|---|---|---|
| 1 | Sc. 1: SPH — crowd density follows $\rho^{0,\max}$ (Table 1) | ✅ | density within 0.03 P/m² of the paper at all six settings (3.28 … 7.24 vs. 3.27 … 7.23); flow within 5% |
| 2 | Sc. 1: SF — density insensitive to $K^{ag}$, flow falls with $K^{ag}$ | ✅ shape / ⚠️ numbers | density within 0.06; flow falls 3.31 → 1.95 P/s like the paper's 2.70 → 1.29, but 20–50% higher; all 400 evacuate (paper: 398, two stuck); cause not isolated |
| 3 | Sc. 1: SF+SPH — low $\rho^{0,\max}$ limits the density, high → SF values | ✅ density / ⚠️ flow | density within 0.05; flow 3.3 vs. 2.7–2.8 P/s at $\rho^{0,\max} \ge 5$ |
| 4 | Sc. 1: RVO avoids high density, slow evacuation; RVO+SPH flat in $\rho^{0,\max}$ | ✅ shape / ❌ numbers | RVO 2.47 vs. 3.23 P/m², RVO+SPH ≈ 2.4 vs. 3.0 P/m²; a second seed (2.32) and 4× more RVO samples (2.62) do not close the gap; the paper leaves RVO details unspecified |
| 5 | Sc. 1: blending lets the crowd compress to $\rho^{0,\max}$ (SF→SPH, RVO→SPH) | ✅ SF→SPH / ⚠️ RVO→SPH | SF→SPH density within 0.1; RVO→SPH 0.5–0.9 P/m² low, inherited from #4 |
| 6 | Sc. 2: Table 2 — only blending (and SPH) evacuate fast; RVO profiles slow | ✅ | same ranking; SPH-based flows 4.00–4.38 vs. 4.49–4.72 P/s; RVO-based densities again lower |
| 7 | Sc. 2: SPH cannot handle the crossing | ✅ | only SPH has blue–pink overlaps (1.75 per frame); 40 pink agents pushed > 3 m aside |
| 8 | Sc. 2: SF pushes crossing agents aside until they are stuck | ❌ | no pink agent stuck or displaced > 3 m; our pink agents leave at the bottom (the paper does not say) |
| 9 | Sc. 3: wave goes to the stage, bounces, comes back (Fig. 7) | ✅ | all SPH variants; snapshots match panel by panel |
| 10 | Sc. 3: higher $k$ → faster, thicker wave | ✅ | 5.9 / 9.6 / 14.1 m/s for k = 50 / 200 / 500 vs. 6.0 / 9.8 / 12.9 read off Fig. 7 |
| 11 | Sc. 3: $\rho^{0,\max}$ sets the density, barely the wave | ✅ | 4.17 / 5.11 / 5.91 P/m²; front 9.1 / 9.6 / 10.0 m/s |
| 12 | Sc. 3: viscosity and a dynamic rest density stabilise the crowd (Figs. 7f, 8) | ✅ | μ = 0: 3.4× more motion after the wave; static ρ⁰: edge agents 3.3× faster than with dynamic ρ⁰ |
| 13 | Sc. 3: without SPH $K^{ag}$ couples density and wave, agents shake (Fig. 9); without $K^{ag}$ the boundary deforms (Fig. 10) | ✅ | K_ag 200 → 1000: density 5.36 → 4.83, wave 6.4 → 12.9 m/s; K_ag = 0: deep overlaps near walls 0.003 → 0.16 per agent |
| 14 | Frame time linear in N; real time at 10k–30k agents (Fig. 11) | ✅ shape / ❌ absolute | linear (slope 0.98), but about 1.6 s per step at 10k agents vs. 4.94 ms (about 320× slower, pure-Python per-agent callback) |

**Deviations from the paper's setup, all documented above:** the concert
starts from the converged crowd shape with 20 s of settling instead of
150 s of arrival (compute cost, see #14); JuPedSim's shortest-path
orientation replaces straight-to-goal steering, and the concert's goal
sits at the stage tip; a wall-slide safeguard keeps JuPedSim from
aborting; RVO's sample count and overlap rule are our own choices where
the paper is silent.

**Bottom line.** The parts of the paper that are about SPH itself are
reproduced closely, in absolute numbers, not only in shape: density
control by $\rho^{0,\max}$, the SPH-based evacuation flows, and the
shockwave's existence, speed, dependence on $k$ and the stabilising
roles of viscosity, the dynamic rest density and contact forces. The
differences are all in the collision-avoidance components the paper
takes from other work: our RVO produces a looser crowd, and our SF
evacuates faster, without the two agents that never pass the door. The
paper leaves those components partly unspecified, and we did not tune
them to match. Real-time performance is out of reach for a pure-Python
per-agent implementation.
""")

nb["cells"] = cells
with open("SPH_Crowds_validation.ipynb", "w") as f:
    nbf.write(nb, f)

print(f"Wrote SPH_Crowds_validation.ipynb with {len(cells)} cells")
