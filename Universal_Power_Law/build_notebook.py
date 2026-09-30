"""Programmatically builds Universal_Power_Law_validation.ipynb from
markdown/code cell source strings below, so the notebook's content is
versionable as plain Python/Markdown rather than raw ipynb JSON. Run, then
execute with:

    jupyter nbconvert --to notebook --execute --inplace Universal_Power_Law_validation.ipynb
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
# Validating the Python/JuPedSim Universal Power Law model against Karamouzas, Skinner & Guy (2014)

This notebook checks a Python implementation of the **anticipatory
time-to-collision force** of the "universal power law" paper, running inside
**JuPedSim**'s experimental `CustomOperationalModel` plugin API (the same
pattern as the SFM and RVO notebooks in this repo), against the paper that
introduced it:

> I. Karamouzas, B. Skinner, S. J. Guy. *Universal Power Law Governing
> Pedestrian Interactions.* Phys. Rev. Lett. 113, 238701 (2014).
> [doi:10.1103/PhysRevLett.113.238701](https://doi.org/10.1103/PhysRevLett.113.238701)

**How the paper validated the model** (see `model_description.md`). Its
core claim is a measurement on real crowd data, not a simulation: the pair
distribution function of real pedestrians, written as a function of the
time to collision τ, gives an interaction energy E(τ) ∝ ln(1/g(τ)) that
follows E ∝ τ⁻² (Fig. 2; exponent 2.05 Outdoor, 2.02 Bottleneck). The
authors then turned that energy into a force (Eq. 3) and checked their
simulation against it:

1. **Self-consistency (Fig. 4, Fig. S4B, Fig. S6):** simulated trajectories,
   put through the same g(τ) analysis, give back E ∝ τ⁻². A distance-based
   force (Helbing et al. 2000) does not: its E shows no dependence on τ.
2. **g(r) by approach rate (Fig. S5 vs. Fig. 1c):** the simulations
   reproduce the strong dependence of g(r) on the rate of approach seen in
   the Outdoor data.
3. **Fundamental diagram (Fig. S4A):** simulated speed-density data
   "in good agreement" with Weidmann (1993).
4. **Qualitative crowd phenomena (Fig. 3):** arching at an exit (a),
   lanes in a hallway (b), clogging and "zipping" in a bottleneck (c),
   diagonal stripes in a crossing (d), and a vortex of goal-less walkers (e).
5. **Anticipation (Fig. 1a/b):** strong avoidance of a distant head-on
   pedestrian, no reaction to a close side-by-side one.

The main text only gives Eq. 3 and a list of phenomena. The force
expression, every simulation setup and parameter, and Figs. S4-S6 are in
the paper's **Supplemental Material**, and the complete model (driving
force, wall force, acceleration cap) is in the authors' **C++ reference
code**. Both were downloaded from the authors' page
(<http://motion.cs.umn.edu/PowerLaw/>): the Supplemental Material is saved
next to the paper as `UniversalPowerLaw_SupplementalMaterial_Karamouzas2014.pdf`,
and the authors' published E(τ) data (Fig. 2) as `validation/paper_data/`.

**Code layout** (all in `Universal_Power_Law/`):

- `pyUniversal_Power_Law.py`: the model (`UniversalPowerLawModel`, `PowerLawState`).
- `validation/run_two_agents.py`, `run_hallway.py`, `run_bottleneck.py`,
  `run_evacuation.py`, `run_crossing.py`, `run_collective.py`: one runner
  per paper experiment; `validation/common.py` holds the shared simulation loop.
- `validation/analysis.py`: trajectory loading, time to collision, g(τ), g(r),
  E(τ), Edie cells, lane/arch/layer/vortex metrics.
- `validation/results/`: `.sqlite` trajectories, `.json` run summaries,
  `plot_*.png`, and `paper_fig*.png` crops of the paper's figures.
""")

code(r"""
import pathlib
import sys

# jupedsim is not pip-installed; its Python package and compiled bindings are
# made importable via PYTHONPATH (see jupedsim/build/environment). Jupyter
# kernels don't inherit that, so replicate it here. The kernel cwd is this
# notebook's folder (Universal_Power_Law/), matching the pathlib.Path("validation")
# usage below.
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
    load_trajectory, observed_and_scrambled, interaction_energy, fit_power_law,
    pair_distribution, edie_cells, binned_fundamental_diagram, weidmann_speed,
    min_clearance, passage_times, steady_flow, jam_front, lateral_layers,
    crossing_agent_metrics, stripe_orientation, lane_order, collective_order,
    ANALYSIS_RADIUS,
)

RESULTS = pathlib.Path("validation/results")
PAPER_DATA = pathlib.Path("validation/paper_data")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

# Same caveat as the SFM and RVO notebooks: this is JuPedSim's own internal
# replay utility. Its module docstring says explicitly "We make no promises
# about the functions from this file w.r.t. API stability... Do not use the
# code here. Use it at your own peril." We use it anyway, for the same reason
# -- there isn't a public, stable alternative.
from jupedsim.internal.notebook_utils import animate, read_sqlite_file

MEAN_AGENT_RADIUS = 0.25


def summary(tag):
    with open(RESULTS / f"{tag}.json") as f:
        return json.load(f)
""")


md(r"""
### Reference values from the paper

Every plot below also shows the paper's own result, drawn in black
(dashed lines or markers) and labelled "paper". Three kinds of reference
are used:

**1. The authors' own data (exact).** Fig. 2's E(τ) for the real Outdoor
and Bottleneck crowds is published as CSV on the authors' page
(`validation/paper_data/*_E_ttc.csv`, τ step 0.01 s, normalized so that
E(1) ≈ 1). It is used as is.

**2. Digitized figures.** Fig. 4 (E(τ) of the authors' hallway and
bottleneck simulations), Fig. S4A (their speed-density markers) and
Fig. S5A/B (their g(r) per approach rate) have no published data. The
pages were rendered with `pdftoppm -r 300`, and the axis box of each panel
was located from its black frame lines, which fixes the pixel-to-unit
mapping. Because every curve and marker series has its own colour,
points were then read automatically: for lines, the median pixel row of
that colour in a 5-pixel-wide column at each abscissa (τ every 0.05 s,
r every 0.01 m below 1.2 m and every 0.05 m above); for Fig. S4A, the
centre of each marker's bounding box. Legend pixels were masked out, and
the result was overlaid on the crop to check it by eye. Resolution is
about ±0.01 in E, ±0.03 in g and ±0.01 m/s in speed. Markers hidden
under other markers in Fig. S4A are missing, and the shaded ±1σ bands of
Fig. 4 were not digitized.

**3. Claims stated in words**, turned into reference values:

- Fig. 2 / Supplement: free power-law exponent 2.05 ± 0.12 (Outdoor) and
  2.02 ± 0.19 (Bottleneck); τ⁻² fits with R² = 0.92 and 0.94.
- Fig. 4 inset / Fig. S6A: distance-based force, "the inferred interaction
  energy does not show a dependence on τ", i.e. E ≈ 0 over 0.5-2.5 s.
- Fig. 1b: two pedestrians side by side show "no relative acceleration",
  i.e. 0 m/s².
- Bottleneck 2.5 m: "5-6 overlapping layers" inside the constriction.
- Crossing: pedestrians "prefer to slow down ... rather than deviate", i.e.
  more than half of each agent's delay comes from slowing down; stripes are
  "diagonal", i.e. at 45° to both walking directions.
- Collective motion: "all pedestrians are walking in unison" in a vortex,
  i.e. a milling order parameter near 1.

**4. Qualitative figures** (Fig. 1a/b, Fig. 3a-e) were cropped from the
PDF into `validation/results/paper_fig*.png` and are shown next to our own
plots.
""")

code(r"""
# Exact data published by the authors (Fig. 2 of the paper).
PAPER_FIG2_OUTDOOR = pd.read_csv(PAPER_DATA / "Outdoor_E_ttc.csv")
PAPER_FIG2_BOTTLENECK = pd.read_csv(PAPER_DATA / "Bottleneck_E_ttc.csv")

# Digitized from the paper's figures (see markdown above for method).
PAPER_FIG4 = {  # main text Fig. 4, E(tau) of the authors' simulations (tau step 0.05 s)
    "hallway": {"tau": [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5],
              "E": [1.94, 1.52, 1.06, 0.97, 0.84, 0.8, 0.68, 0.7, 0.67, 0.67, 0.62, 0.58, 0.57, 0.52, 0.54, 0.52, 0.45, 0.43, 0.36, 0.28, 0.38, 0.27, 0.29, 0.28, 0.22, 0.22, 0.21, 0.21, 0.15, 0.14, 0.14, 0.14, 0.14, 0.15, 0.08, 0.13]},
    "bottleneck": {"tau": [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5],
              "E": [1.34, 1.23, 1.15, 0.96, 1.01, 0.89, 0.92, 0.76, 0.77, 0.74, 0.59, 0.55, 0.53, 0.53, 0.51, 0.53, 0.47, 0.36, 0.34, 0.33, 0.4, 0.32, 0.36, 0.37, 0.28, 0.29, 0.3, 0.19, 0.25, 0.23, 0.19, 0.2, 0.15, 0.2, 0.1, 0.1]},
}
PAPER_FIGS5A = {
    "0<v<=1": {"r": [0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95],
               "g": [0.1, 0.28, 0.52, 0.89, 1.39, 2.04, 3.07, 3.93, 4.72, 5.27, 5.45, 5.25, 5.0, 4.66, 4.13, 3.65, 3.22, 2.93, 2.62, 2.46, 2.35, 2.27, 2.17, 2.07, 2.0, 1.94, 1.88, 1.82, 1.76, 1.73, 1.69, 1.67, 1.65, 1.65, 1.59, 1.55, 1.54, 1.5, 1.47, 1.45, 1.41, 1.37, 1.33, 1.27, 1.22, 1.2, 1.2, 1.21, 1.24, 1.25, 1.24, 1.22, 1.21, 1.18, 1.17, 1.16, 1.14, 1.13, 1.14, 1.16, 1.18, 1.18, 1.19, 1.18, 1.17, 1.17, 1.16, 1.14, 1.13, 1.11, 1.09, 1.06, 1.04, 1.02, 1.05, 1.03, 0.98, 0.98, 0.98, 1.04, 1.12, 1.22, 1.44, 1.31, 1.27, 1.25, 1.16, 1.12, 1.12, 1.09, 1.0, 1.02, 1.05, 1.06, 1.07, 1.06, 1.02, 1.02, 1.0, 1.06, 1.04, 1.0, 1.09, 1.09, 1.16, 1.16, 1.08, 1.04, 1.05]},
    "1<v<=2": {"r": [0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95],
               "g": [0.03, 0.03, 0.05, 0.07, 0.11, 0.16, 0.21, 0.29, 0.46, 0.63, 0.77, 0.89, 1.0, 1.08, 1.14, 1.2, 1.24, 1.27, 1.3, 1.34, 1.38, 1.43, 1.47, 1.51, 1.55, 1.58, 1.6, 1.59, 1.55, 1.5, 1.45, 1.4, 1.39, 1.39, 1.42, 1.42, 1.42, 1.41, 1.39, 1.38, 1.35, 1.34, 1.31, 1.31, 1.33, 1.36, 1.39, 1.41, 1.42, 1.41, 1.41, 1.39, 1.35, 1.33, 1.3, 1.27, 1.27, 1.27, 1.27, 1.27, 1.27, 1.29, 1.29, 1.29, 1.29, 1.27, 1.26, 1.24, 1.1, 1.1, 1.14, 1.05, 1.18, 1.1, 0.96, 1.02, 1.14, 1.0, 1.06, 1.1, 1.02, 1.06, 1.04, 1.06, 1.13, 1.08, 1.0, 1.17, 1.08, 1.05, 1.16, 1.05, 0.93, 1.04, 0.98, 0.98, 0.98, 0.94, 0.97, 0.94, 0.96, 0.98, 0.94]},
    "v>2": {"r": [0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95],
               "g": [0.03, 0.03, 0.03, 0.04, 0.04, 0.04, 0.04, 0.05, 0.06, 0.07, 0.1, 0.12, 0.14, 0.15, 0.18, 0.2, 0.24, 0.27, 0.31, 0.35, 0.35, 0.35, 0.34, 0.34, 0.32, 0.31, 0.3, 0.3, 0.32, 0.36, 0.39, 0.42, 0.46, 0.5, 0.54, 0.56, 0.57, 0.59, 0.6, 0.61, 0.63, 0.64, 0.65, 0.67, 0.6, 0.63, 0.73, 0.83, 0.79, 0.8, 0.84, 0.86, 0.76, 0.88, 0.9, 0.83, 0.84, 0.93, 0.91, 0.96, 0.93, 0.94, 0.81, 1.0, 0.94, 0.88, 0.98, 0.94, 0.88, 1.0, 1.0, 0.94, 0.94, 0.97, 1.05, 1.05, 0.94, 0.92, 0.94]},
}
PAPER_FIGS5B = {
    "0<v<=1": {"r": [0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9],
               "g": [0.02, 0.02, 0.02, 0.04, 0.06, 0.09, 0.14, 0.18, 0.22, 0.27, 0.31, 0.35, 0.39, 0.43, 0.47, 0.51, 0.57, 0.68, 0.72, 0.76, 0.8, 0.83, 0.87, 0.92, 0.97, 1.08, 1.17, 1.34, 1.48, 1.63, 1.8, 1.94, 2.09, 2.2, 2.34, 2.4, 2.4, 2.37, 2.33, 2.28, 2.22, 2.15, 2.08, 2.0, 1.87, 1.78, 1.7, 1.63, 1.56, 1.51, 1.46, 1.42, 1.4, 1.41, 1.42, 1.42, 1.41, 1.4, 1.39, 1.37, 1.35, 1.32, 1.28, 1.22, 1.17, 1.13, 1.1, 1.07, 1.03, 1.02, 1.02, 1.02, 1.03, 1.05, 1.06, 1.06, 1.06, 1.05, 1.05, 1.05, 0.99, 0.92, 0.94, 0.99, 1.01, 1.03, 1.07, 1.08, 1.07, 1.05, 1.09, 1.05, 1.08, 1.12, 1.07, 1.14, 1.11, 1.12, 1.13, 1.09, 1.09, 1.05, 1.09, 1.06, 1.09, 1.05, 1.05, 0.99, 1.02, 1.03, 1.09, 1.07, 1.09, 1.05]},
    "1<v<=2": {"r": [0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95],
               "g": [0.26, 0.34, 0.44, 0.55, 0.62, 0.72, 0.82, 0.95, 1.04, 1.14, 1.24, 1.33, 1.4, 1.46, 1.5, 1.52, 1.55, 1.55, 1.53, 1.51, 1.47, 1.42, 1.35, 1.29, 1.25, 1.24, 1.28, 1.29, 1.27, 1.22, 1.16, 1.1, 1.07, 1.11, 1.18, 1.28, 1.3, 1.28, 1.23, 1.19, 1.17, 1.19, 1.23, 1.26, 1.26, 1.17, 1.09, 1.01, 0.97, 0.99, 1.02, 1.06, 1.08, 1.09, 1.1, 1.1, 1.09, 1.08, 1.07, 1.05, 1.03, 1.01, 1.0, 0.98, 0.97, 0.99, 1.01, 1.03, 1.03, 1.01, 0.99, 0.96, 0.95, 0.96, 0.98, 1.0, 1.02, 1.03, 1.03, 1.03, 1.03, 0.98, 0.99, 1.03, 1.03, 1.0, 0.83, 0.88, 1.02, 1.11, 1.03, 1.1, 1.02, 1.19, 1.18, 1.17, 1.11, 1.11, 1.07, 1.02, 1.2, 1.08, 0.99, 0.95, 1.05, 1.05, 1.03, 1.1, 1.0, 1.17, 1.25, 1.01, 1.09, 1.17, 1.12, 1.01]},
    "v>2": {"r": [0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95],
               "g": [0.02, 0.02, 0.03, 0.06, 0.12, 0.2, 0.3, 0.4, 0.47, 0.53, 0.55, 0.57, 0.59, 0.62, 0.65, 0.68, 0.68, 0.67, 0.64, 0.61, 0.6, 0.61, 0.63, 0.65, 0.64, 0.58, 0.53, 0.46, 0.45, 0.48, 0.59, 0.66, 0.72, 0.77, 0.81, 0.86, 0.87, 0.83, 0.77, 0.68, 0.62, 0.57, 0.54, 0.5, 0.49, 0.53, 0.58, 0.63, 0.65, 0.61, 0.56, 0.52, 0.52, 0.56, 0.61, 0.66, 0.71, 0.74, 0.77, 0.78, 0.76, 0.71, 0.66, 0.61, 0.57, 0.54, 0.53, 0.51, 0.49, 0.47, 0.44, 0.42, 0.41, 0.42, 0.44, 0.46, 0.49, 0.54, 0.57, 0.61, 0.65, 0.69, 0.73, 0.76, 0.78, 0.65, 0.59, 0.4, 0.64, 0.88, 0.8, 0.62, 0.72, 0.85, 0.83, 0.88, 0.7, 0.8, 0.91, 0.95, 0.94, 0.76, 0.92, 0.89, 1.01, 0.94, 0.94, 0.88, 0.96, 0.89, 0.95, 0.92, 0.84, 0.95, 0.99, 1.05, 0.89, 0.95, 1.01, 0.98]},
}
PAPER_FIGS4A = {  # Fig. S4A markers: (density bin [1/m^2], mean speed [m/s])
    "hallway": {"density": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.7, 2.8, 2.9, 3.0],
                   "speed": [1.3, 1.43, 1.48, 1.29, 1.23, 1.11, 1.21, 1.23, 1.2, 1.12, 1.13, 1.15, 1.06, 1.06, 0.94, 0.82, 0.74, 0.49, 0.49, 0.26, 0.26, 0.39, 0.51, 0.2, 0.21, 0.22, 0.22]},
    "bottleneck": {"density": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 1.9, 2.0, 2.1, 2.2, 2.3, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 4.4, 4.5, 4.9],
                   "speed": [1.01, 0.91, 0.71, 1.1, 1.1, 0.98, 1.0, 0.88, 1.0, 0.67, 0.48, 0.44, 0.49, 0.42, 0.66, 0.84, 0.53, 0.63, 0.2, 0.27, 0.31, 0.41, 0.41, 0.15, 0.16, 0.28, 0.44, 0.47, 0.47, 0.13, 0.12, 0.14, 0.18, 0.12]},
    "evacuation": {"density": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9],
                   "speed": [0.78, 0.75, 0.56, 0.75, 0.88, 1.06, 1.25, 1.05, 0.89, 0.53, 0.69, 0.98, 0.84, 0.92, 0.8, 0.66, 0.69, 0.77, 0.69, 0.07, 0.09, 0.17, 0.12, 0.08, 0.11, 0.12, 0.15, 0.09, 0.09, 0.2, 0.2, 0.2, 0.19, 0.15, 0.16, 0.23, 0.23, 0.05, 0.06, 0.12, 0.14, 0.2, 0.24, 0.18, 0.15, 0.21, 0.24]},
}

# Claims stated in words.
PAPER_EXPONENT = {"Outdoor": (2.05, 0.123), "Bottleneck": (2.017, 0.192)}  # Supplement, Fig. 2 fit
PAPER_R2 = {"Outdoor": 0.92, "Bottleneck": 0.94}
PAPER_DISTANCE_E = 0.0            # Fig. 4 inset / Fig. S6A: no dependence on tau
PAPER_PARALLEL_ACCEL = 0.0        # Fig. 1b: "without any relative acceleration"
PAPER_LAYERS_W25 = (5, 6)         # Supplement: "5-6 overlapping layers" at 2.5 m
PAPER_STRIPE_ANGLE = 45.0         # Fig. 3d: "diagonal line-shaped patterns"
PAPER_SLOWDOWN_SHARE = 0.5        # "prefer to slow down ... rather than deviate"
PAPER_MILLING = 1.0               # Fig. 3e: "all pedestrians are walking in unison"

PAPER_IMG = {k: RESULTS / f"paper_{k}.png" for k in [
    "fig1ab_anticipation", "fig3a_evacuation", "fig3b_hallway", "fig3c_bottleneck",
    "fig3d_crossing", "fig3e_collective", "fig4_energy", "figS4_fd_energy",
    "figS5_gr", "figS6_models"]}
PAPER_STYLE = dict(color="black", ls="--", alpha=0.7)
""")

# ============================================================ Implementation notes
md(r"""
## Implementation notes: the model, one extension, the time step, and the problems found

**The model (`pyUniversal_Power_Law.py`).** With x = xᵢ − xⱼ, v = vᵢ − vⱼ,
R = rᵢ + rⱼ, a = |v|², b = −x·v, c = |x|² − R², d = b² − ac, the time to
collision is τ = (b − √d)/a, and the interaction force is the
Supplemental Material's Eq. S2,

$$\mathbf F_{ij} = -\frac{k\,e^{-\tau/\tau_0}}{|\mathbf v|^2\tau^2}\left(\frac{2}{\tau}+\frac{1}{\tau_0}\right)\left[\mathbf v - \frac{|\mathbf v|^2\mathbf x-(\mathbf x\cdot\mathbf v)\mathbf v}{\sqrt{d}}\right],$$

which is zero when no future collision exists. The cell below checks it
against a finite-difference gradient of E(τ) = k τ⁻² e^(−τ/τ₀). Everything
the paper leaves to its reference code is taken from the authors' C++
`Agent::computeForces`: the driving force (v_pref − v)/ξ with ξ = 0.54 s,
k = 1.5, τ₀ = 3 s, unit mass, a 10 m neighbour range, the total
acceleration clamped to 20 m/s², overlapping pairs using R − |x| as their
radius, and walls repelling through the time to collision with the wall
segment inflated to a capsule. Preferred speeds are drawn from N(1.3, 0.3)
m/s as in the Supplemental Material, clipped to [0.5, 2.1] m/s.

**Agent radius, 0.25 m.** The paper never states it (the C++ example uses
0.5 m, the Python demo 0.2 m). The authors' own simulated g(r) (Fig. S5A)
is zero below r ≈ 0.5 m and peaks at 0.56 m, i.e. a contact distance of
2 × 0.25 m. We use 0.25 m.

**The distance-based control** (Fig. 4 inset, S5B, S6A) is the same model
class with `interaction="distance"`: the pedestrian force of Helbing,
Farkas & Vicsek (2000), the paper's Ref. [9], with that paper's constants
(A = 2000 N, B = 0.08 m, k = 1.2·10⁵ kg/s², κ = 2.4·10⁵ kg/(m s), 80 kg)
and the same driving force.

**One JuPedSim-specific extension.** JuPedSim raises an error whenever a
step would carry an agent across a wall, while the reference model has no
contact force at all (an agent already touching a wall feels no wall
force). `_wall_guard` removes only the part of a step that would bring the
agent's centre within 5 cm of a wall. It changes nothing for an agent that
is not already overlapping a wall.

**Goals.** The reference code deletes an agent at its goal, so every
goal-directed scenario ends in an exit stage. Exit zones are 3-4 m deep:
see problem 1 below.
""")

code(r"""
# Eq. S2 as implemented vs. -grad E(tau) by central differences.
sys.path.insert(0, str(pathlib.Path.cwd()))  # the model module lives next to this notebook
from pyUniversal_Power_Law import UniversalPowerLawModel, PowerLawState

def tau_of(xi, vi, xj, vj, R):
    x, v = xi - xj, vi - vj
    a, b, c = v @ v, -x @ v, x @ x - R * R
    with np.errstate(invalid="ignore"):  # NaN = no future collision
        return (b - np.sqrt(b * b - a * c)) / a

state = PowerLawState(velocity=(1.3, 0.0))
E = lambda t: state.k / t**2 * np.exp(-t / state.tau0)
rng = np.random.default_rng(0)
rows = []
for _ in range(5):
    xi, vi = np.zeros(2), np.array([1.3, 0.0])
    xj, vj = np.array([4.0, rng.uniform(-0.3, 0.3)]), np.array([-1.1, rng.uniform(-0.2, 0.2)])
    h = 1e-6
    grad = np.array([(E(tau_of(xi + h * e, vi, xj, vj, 0.5)) - E(tau_of(xi - h * e, vi, xj, vj, 0.5))) / (2 * h)
                     for e in np.eye(2)])
    force = UniversalPowerLawModel._ttc_agent_force(state, (xj - xi)[None], (vi - vj)[None], np.array([0.5]))
    rows.append({"tau [s]": tau_of(xi, vi, xj, vj, 0.5), "-dE/dx": -grad[0], "F_x (Eq. S2)": force[0],
                 "-dE/dy": -grad[1], "F_y (Eq. S2)": force[1]})
pd.DataFrame(rows).round(8)
""")

md(r"""
**Time step: dt = 0.01 s, chosen empirically.** The paper gives none (the
authors' C++ example uses 0.005 s, their Python demo 0.02 s). We ran the
cheapest scenario (Crossing, 80 agents, 3 seeds) and the densest one
(1 m Bottleneck, 150 agents) at several dt:
""")

code(r"""
rows = []
for dt in ["0.05", "0.02", "0.01", "0.005", "0.0025"]:
    for sfx in ["", "_s2", "_s3"]:
        f = RESULTS / f"dt_crossing_{dt}{sfx}.json"
        if not f.exists():
            continue
        s = json.load(open(f))
        df, _ = load_trajectory(RESULTS / f"dt_crossing_{dt}{sfx}.sqlite")
        m = crossing_agent_metrics(df, s)
        angle, _, _ = stripe_orientation(df, s)
        rows.append({"dt": float(dt), "seed": s["seed"], "min_clearance_m": min_clearance(df, 0.25),
                     "mean_delay_s": m["delay"].mean(), "stripe_angle_deg": angle})
dt_cross = pd.DataFrame(rows)
print("Crossing, mean (std) over seeds:")
display(dt_cross.groupby("dt")[["min_clearance_m", "mean_delay_s", "stripe_angle_deg"]]
        .agg(["mean", "std"]).round(3).sort_index(ascending=False))

rows = []
for dt in ["0.05", "0.02", "0.01", "0.005"]:
    for sfx in ["", "_s2", "_s3"]:
        f = RESULTS / f"dt_bottleneck_w1.0_{dt}{sfx}.json"
        if not f.exists():
            continue
        s = json.load(open(f))
        df, _ = load_trajectory(RESULTS / f"dt_bottleneck_w1.0_{dt}{sfx}.sqlite")
        pt = passage_times(df, 5.0)
        rows.append({"dt": float(dt), "seed": s["seed"], "min_clearance_m": min_clearance(df[df.frame % 2 == 0], 0.25),
                     "flow_1_per_s": steady_flow(pt), "passed_in_120s": len(pt), "crashed": s["crashed"]})
dt_bn = pd.DataFrame(rows)
print("Bottleneck, width 1 m, first 120 s (seeds 2-3 only at dt = 0.01 and 0.005):")
display(dt_bn.round(3))
dt_bn.groupby("dt")[["min_clearance_m", "flow_1_per_s"]].agg(["mean", "std", "count"]).round(3).sort_index(ascending=False)
""")

md(r"""
At dt = 0.05 and 0.02 s, two agents in the jam at the 1 m bottleneck's
entrance end up on top of each other (clearance −0.50 m, i.e. zero centre
distance) and then walk through the neck overlapping; the flow looks
normal, so only the clearance reveals it. In the Crossing, dt = 0.02 s
overestimates the mean delay by about 70% (0.79 vs. 0.46 s), and
dt = 0.05 s by a factor of 6 with real overlaps. From 0.01 s down,
clearance stays ≥ 0 (apart from a 6 mm overlap in one 0.005 s run), and
the Crossing delay and stripe angle agree with 0.005 and 0.0025 s within
the seed-to-seed spread. One residual dt effect remains: in the 1 m
bottleneck the flow is about 5% lower at 0.01 s than at 0.005 s
(1.24 ± 0.06 vs. 1.31 ± 0.04 /s over three seeds). We accept that for
the halved cost; no result below depends on a 5% change in flow.
dt = 0.01 s is used everywhere; trajectories are written at 10 frames/s.

**Problems found while building this, and what was done:**

1. **Exit placed 1 m in front of a wall slows arrivals.** In the first
   Crossing run the exit zone was the last 1 m before the arena wall. The
   wall's anticipatory force already acts at that distance (time to
   collision ≈ 0.6 s at walking speed), so the two slowest agents reached
   the exit line at only 60-65% of their preferred speed (0.32 vs. 0.50 m/s
   and 0.39 vs. 0.66 m/s). Exit zones are now 3-4 m deep, so agents are
   removed before the far wall matters.
2. **Agents merging at dt ≥ 0.02 s** (above): found through the clearance
   check, not visible in the flow.
3. **E(τ) normalization.** Normalizing both τ histograms as densities over
   0-8 s shifts E down by about 0.3-0.5, because the suppressed short-τ
   region pushes g above 1 at large τ. The Supplemental Material defines
   g so that g → 1 for non-interacting separations, so g is scaled to 1
   on 3-8 s (beyond τ₀), and the global version is shown for comparison.
4. **A lane metric computed before the groups meet reads "perfect lanes"**
   (φ = 1 when both groups are in the central section but still apart).
   A first filter (≥ 30 agents of each group in the section) still let
   such frames in, from t = 2.5 s; φ is now only evaluated while ≥ 30
   agents of each group are inside the other group's extent along x.
5. **One exit polygon per side made both hallway groups funnel to the
   centre line.** JuPedSim steers agents towards a single point of their
   exit polygon: with one exit spanning the whole 20 m end, agents near
   the walls headed ±12° inwards and each group's lateral spread shrank
   from σ = 5.5 m to 3.4 m before the groups met, crowding the middle.
   The paper's pedestrians walk "from opposite ends", and the reference
   code gives each agent its own goal point, so every agent now has its
   own 1 m wide exit box straight ahead (Hallway and Crossing). All
   Hallway and Crossing runs were redone after this fix.
6. **A metric bug in the Crossing.** Agents are removed as soon as they
   enter their exit zone, so their last recorded position (10 frames/s)
   often lies just short of it; measured against the zone's edge, only
   16 of 80 agents "arrived" and the truncated paths hid part of their
   detour (in the runs of that time, it inflated the slow-down share from
   0.67-0.69 to 0.73-0.79). Delays
   are now measured to a line 0.5 m before the exit zone.
7. **Too-short runs.** The first evacuation run's 240 s cap would have ended
   with agents still inside (the 0.8 m door passes only ~0.6 agents/s); it
   was rerun with 400 s. The paper-size collective-motion run (750 agents)
   was not feasible (see Section 9).
""")

# ============================================================ 1. Anticipation
md(r"""
## 1. Anticipation: two pedestrians (Fig. 1a/b)

**Paper**: real pedestrians on a collision course "react strongly to avoid
an upcoming collision even though they are far from each other" (Fig. 1a),
while two pedestrians walking side by side, close together, show "no
relative acceleration" (Fig. 1b). This is the observation the whole law is
built on: interaction depends on the time to collision, not on distance.

**Our setup** (`run_two_agents.py`): two agents at 1.3 m/s, (i) head-on,
starting 16 m apart with a 0.2 m lateral offset (an exactly head-on pair
has no sideways force by symmetry); (ii) side by side, 0.7 m apart
(a 0.2 m gap), same velocity. Each run with the anticipatory force and with
the distance-based control.
""")

code(r"""
def two_agent_metrics(tag):
    df, _ = load_trajectory(RESULTS / f"{tag}.sqlite")
    a, b = [g.sort_values("t").reset_index(drop=True) for _, g in df.groupby("id")]
    n = min(len(a), len(b)); a, b = a.iloc[:n], b.iloc[:n]
    sep = np.hypot(a.pos_x - b.pos_x, a.pos_y - b.pos_y).to_numpy()
    acc_a = np.stack([np.gradient(a.vx, a.t), np.gradient(a.vy, a.t)], axis=1)
    acc_b = np.stack([np.gradient(b.vx, b.t), np.gradient(b.vy, b.t)], axis=1)
    lateral = np.abs(acc_a[:, 1])
    onset = np.flatnonzero(lateral > 0.05)
    return {
        "t": a.t.to_numpy(), "sep": sep, "lateral_acc": lateral,
        "rel_acc": np.linalg.norm(acc_a - acc_b, axis=1), "a": a, "b": b,
        "row": {
            "min_separation_m": sep.min(),
            "reaction_onset_separation_m": sep[onset[0]] if len(onset) else np.nan,
            "max_accel_m_s2": np.linalg.norm(acc_a, axis=1).max(),
            "max_relative_accel_m_s2": np.linalg.norm(acc_a - acc_b, axis=1).max(),
            "max_lateral_deviation_m": np.abs(a.pos_y - a.pos_y.iloc[0]).max(),
        },
    }

two = {(c, i): two_agent_metrics(f"two_{c}_{i}") for c in ["headon", "parallel"] for i in ["ttc", "distance"]}
two_df = pd.DataFrame({f"{c} / {i}": m["row"] for (c, i), m in two.items()}).T
two_df.round(3)
""")

code(r"""
fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1])
ax = fig.add_subplot(gs[0, 0])
ax.imshow(plt.imread(PAPER_IMG["fig1ab_anticipation"])); ax.axis("off")
ax.set_title("Paper, Fig. 1(a) head-on: early, strong avoidance;\n(b) side by side: no relative acceleration", fontsize=9)

ax = fig.add_subplot(gs[0, 1])
for (c, i), m in two.items():
    if c != "headon":
        continue
    style = "-" if i == "ttc" else "--"
    ax.plot(m["a"].pos_x, m["a"].pos_y, style, color="tab:blue", label=f"{i}: agent walking +x")
    ax.plot(m["b"].pos_x, m["b"].pos_y, style, color="tab:red", label=f"{i}: agent walking -x")
ax.set_xlim(-9, 9); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title("Ours, head-on paths (solid: anticipatory, dashed: distance-based)", fontsize=9)
ax.legend(fontsize=7)

ax = fig.add_subplot(gs[1, 0])
for i, col in [("ttc", "tab:blue"), ("distance", "tab:orange")]:
    m = two[("headon", i)]
    approaching = np.arange(len(m["sep"])) <= np.argmin(m["sep"])
    ax.plot(m["sep"][approaching], m["lateral_acc"][approaching], color=col,
            label=f"{i}: reacts at {m['row']['reaction_onset_separation_m']:.1f} m")
ax.set_xlabel("separation between the two agents [m] (approaching)")
ax.set_ylabel("sideways acceleration [m/s$^2$]"); ax.set_yscale("log"); ax.set_ylim(1e-3, 20)
ax.invert_xaxis(); ax.legend(fontsize=8)
ax.set_title("Head-on: when does each agent start to steer away?", fontsize=9)

ax = fig.add_subplot(gs[1, 1])
for i, col in [("ttc", "tab:blue"), ("distance", "tab:orange")]:
    m = two[("parallel", i)]
    ax.plot(m["t"], m["rel_acc"], color=col, label=f"ours, {i}")
ax.axhline(PAPER_PARALLEL_ACCEL, lw=1.5, label="paper, Fig. 1b: no relative acceleration", **PAPER_STYLE)
ax.set_xlabel("t [s]"); ax.set_ylabel("|relative acceleration| [m/s$^2$]")
ax.set_title("Side by side, 0.7 m apart", fontsize=9); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(RESULTS / "plot_two_agents.png", dpi=130)
plt.show()
""")

md(r"""
**Verdict: reproduced.** Head-on, the anticipatory agents start to steer
at a separation of about 5 m, i.e. about 2 s before contact at their
2.6 m/s closing speed, with a gentle peak acceleration below
0.5 m/s², and pass each other 0.6 m apart (centre to centre). The
distance-based agents react only when they are
less than 1 m apart, and then with an acceleration more than fifteen
times larger (about 8 m/s²). Side by side, the anticipatory pair feels
no force at all (the time to collision is undefined for equal
velocities, so Eq. S2 is exactly zero; the residual is finite-difference
noise), exactly as in Fig. 1b, while the distance-based pair is pushed
apart from 0.70 m to almost 1 m. This is the contrast the whole paper is
built on, and it holds by construction of Eq. S2; the table above gives
the exact values.

**Replay** (head-on pair, anticipatory force).
""")

code(r"""
traj, area = read_sqlite_file(str(RESULTS / "two_headon_ttc.sqlite"))
animate(traj, area, every_nth_frame=2, radius=MEAN_AGENT_RADIUS, title_note="Head-on pair, anticipatory force")
""")

# ============================================================ 2. E(tau)
md(r"""
## 2. The interaction energy E(τ): does the simulation give back its own power law? (Fig. 4, Fig. S4B, Fig. S6)

**Paper**: trajectories simulated with Eq. 3, put through the same
analysis as the real data, give E(τ) = ln(1/g(τ)) ∝ τ⁻² in the Hallway,
Bottleneck (2.5 m) and Evacuation scenes (Fig. 4, Fig. S4B), over roughly
τ = 0.75-2.5 s. The same analysis of a distance-based force (Helbing
et al. 2000) "does not show a dependence on τ" (Fig. 4 inset, Fig. S6A).

**Analysis** (`analysis.py`, following the Supplemental Material): for
every pair of agents present at the same time, τ is computed from their
positions and finite-difference velocities, treating each as a disc of
the *analysis radius*. The non-interacting reference is the same data
with time "randomly permut[ed] ... between different pedestrians" (three
independent scrambles). g(τ) is the ratio of the two τ histograms (0.01 s
bins), E = ln(1/g), then averaged in 0.05 s windows with ±1σ, as in
Fig. 4 / Fig. S6. Pairs with undefined τ are excluded.

**Which analysis radius?** The paper states 0.1 m for its real data, and
says nothing for its simulations. The result depends on it, so both are
shown: **0.1 m** (the paper's stated value) and **0.25 m** (our agents' body
radius).

**Our data**: pooled over seeds so that the number of finite-τ samples is
comparable to the paper's 120-180k per data set: Hallway 3 seeds,
Bottleneck 2.5 m 5 seeds, Evacuation 1 seed; distance-based control:
Hallway 3 seeds, Bottleneck 3 seeds. Every run uses the setup of its own
section below.
""")

code(r"""
SCENES = {
    "Hallway":         ["hallway_ttc", "hallway_ttc_s2", "hallway_ttc_s3"],
    "Bottleneck 2.5 m": ["bottleneck_w2.5"] + [f"bottleneck_w2.5_s{s}" for s in (2, 3, 4, 5)],
    "Evacuation":      ["evacuation"],
    "Hallway (distance)":    ["hallway_distance", "hallway_distance_s2", "hallway_distance_s3"],
    "Bottleneck (distance)": ["bottleneck_w2.5_distance", "bottleneck_w2.5_distance_s2", "bottleneck_w2.5_distance_s3"],
}
RADII = [0.1, 0.25]
MIN_SAMPLES = 20   # a 0.05 s window needs this many observed pairs to enter the fit
TAU_HI = 2.5

_traj_cache = {}
def traj(tag):
    if tag not in _traj_cache:
        _traj_cache[tag] = load_trajectory(RESULTS / f"{tag}.sqlite")[0]
    return _traj_cache[tag]

energy = {}
for scene, tags in SCENES.items():
    for R in RADII:
        obs, scr = [], []
        for k, tag in enumerate(tags):
            o, s = observed_and_scrambled(traj(tag), n_scrambles=3, radius=R, seed=10 * k)
            obs.append(o); scr.append(s)
        obs, scr = pd.concat(obs), pd.concat(scr)
        fine, coarse = interaction_energy(obs, scr)
        _, coarse_global = interaction_energy(obs, scr, norm_range=None)
        ok = (coarse["count"] >= MIN_SAMPLES) & coarse["E_mean"].notna()
        tau_lo = max(0.5, coarse.loc[ok, "tau"].min())
        fit = fit_power_law(coarse["tau"], coarse["E_mean"], tau_lo, TAU_HI)
        energy[(scene, R)] = dict(coarse=coarse, coarse_global=coarse_global, fit=fit,
                                  tau_lo=tau_lo, n_finite=int(obs["tau"].notna().sum()))

rows = []
for (scene, R), e in energy.items():
    rows.append({"scene": scene, "analysis radius [m]": R, "finite-tau samples": e["n_finite"],
                 "fit range [s]": f"{e['tau_lo']:.2f}-{TAU_HI}",
                 "free exponent": e["fit"]["free_exponent"], "R2 of tau^-2 fit": e["fit"]["r2_fixed_exponent"],
                 "amplitude c": e["fit"]["c"]})
# The authors' own data, through the same fit (their Fig. 2 intervals).
for name, d, lo, hi in [("Outdoor (paper data)", PAPER_FIG2_OUTDOOR, 0.4, 2.4),
                        ("Bottleneck (paper data)", PAPER_FIG2_BOTTLENECK, 0.2, 1.4)]:
    f = fit_power_law(d["tau"], d["E(tau)"], lo, hi)
    rows.append({"scene": name, "analysis radius [m]": 0.1, "finite-tau samples": np.nan,
                 "fit range [s]": f"{lo}-{hi}", "free exponent": f["free_exponent"],
                 "R2 of tau^-2 fit": f["r2_fixed_exponent"], "amplitude c": f["c"]})
energy_df = pd.DataFrame(rows)
energy_df.round(3)
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
colors = {"Hallway": "tab:red", "Bottleneck 2.5 m": "tab:blue", "Evacuation": "tab:cyan"}
tt = np.linspace(0.5, 2.5, 200)
for ax, R in zip(axes, RADII):
    for scene, col in colors.items():
        e = energy[(scene, R)]
        c = e["coarse"]; c = c[(c["tau"] >= e["tau_lo"]) & (c["tau"] <= TAU_HI)]
        y, s = c["E_mean"] / e["fit"]["c"], c["E_std"] / e["fit"]["c"]
        ax.plot(c["tau"], y, color=col, lw=1.5,
                label=f"ours, {scene} (exp. {e['fit']['free_exponent']:.2f})")
        ax.fill_between(c["tau"], y - s, y + s, color=col, alpha=0.15)
    for scene, mk in [("hallway", "o"), ("bottleneck", "s")]:
        ax.plot(PAPER_FIG4[scene]["tau"], PAPER_FIG4[scene]["E"], mk, ms=3.5, mfc="none",
                color="black", alpha=0.7, label=f"paper, Fig. 4 {scene} (digitized)")
    for d, lo, hi, lab in [(PAPER_FIG2_OUTDOOR, 0.4, 2.4, "Outdoor"), (PAPER_FIG2_BOTTLENECK, 0.2, 1.4, "Bottleneck")]:
        dd = d[(d["tau"] >= lo) & (d["tau"] <= hi)]
        cc = fit_power_law(d["tau"], d["E(tau)"], lo, hi)["c"]
        win = dd.groupby(np.floor(dd["tau"] / 0.05)).mean()   # 0.05 s windows, like ours
        ax.plot(win["tau"], win["E(tau)"] / cc, "x", ms=4, color="0.5",
                label="paper, Fig. 2 real data (Outdoor, Bottleneck)" if lab == "Outdoor" else None)
    ax.plot(tt, tt**-2.0, color="black", lw=1.5, label=r"$E \propto \tau^{-2}$")
    ax.set_xlim(0.5, 2.5); ax.set_ylim(-0.2, 3.0)
    ax.set_xlabel(r"$\tau$ [s]")
    ax.set_title(f"analysis radius {R} m" + (" (paper's value)" if R == 0.1 else " (agent radius)"))
axes[0].set_ylabel(r"$E(\tau)$, scaled so that the $\tau^{-2}$ fit is $1/\tau^2$")
axes[1].legend(fontsize=7, loc="upper right")
plt.suptitle("Anticipatory force: E(τ) from our simulations vs. the paper's Fig. 4 and Fig. 2")
plt.tight_layout()
plt.savefig(RESULTS / "plot_energy_ttc.png", dpi=130)
plt.show()
""")

code(r"""
# Same layout as the paper's Fig. S6: the black line is the tau^-2 law, not a
# result. The paper's panels are in arbitrary units in which its anticipatory
# curve follows tau^-2, so in each panel both of our curves are divided by the
# amplitude c of the anticipatory curve's tau^-2 fit (same scene, same
# analysis radius). The distance-based curve then has the same units as in
# Fig. S6A, where it stays within about +-0.3 of zero (grey band).
tt = np.linspace(0.5, TAU_HI, 200)
rows = []
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
for row, R in enumerate(RADII):
    for col, (ttc, dist) in enumerate([("Hallway", "Hallway (distance)"), ("Bottleneck 2.5 m", "Bottleneck (distance)")]):
        ax = axes[row, col]
        scale = energy[(ttc, R)]["fit"]["c"]
        for scene, color, lab in [(ttc, "tab:red", "ours, anticipatory"), (dist, "tab:blue", "ours, distance-based")]:
            c = energy[(scene, R)]["coarse"]
            c = c[(c["tau"] >= 0.5) & (c["tau"] <= TAU_HI) & (c["count"] >= MIN_SAMPLES)]
            y, sd = c["E_mean"] / scale, c["E_std"] / scale
            ax.plot(c["tau"], y, color=color, label=lab)
            ax.fill_between(c["tau"], y - sd, y + sd, color=color, alpha=0.15)
            if scene == dist:
                rows.append({"scene": ttc.split()[0], "analysis radius [m]": R, "scale c (anticipatory fit)": scale,
                             "distance-based E at tau=0.5-0.55 s": y.iloc[0], "at 1.0-1.05 s": y[c["tau"].between(1.0, 1.05)].mean(),
                             "at 2.0-2.05 s": y[c["tau"].between(2.0, 2.05)].mean(),
                             "max |E| over 0.5-2.5 s": y.abs().max()})
        ax.plot(tt, tt**-2.0, color="black", lw=1.5, label=r"$E \propto \tau^{-2}$ (as in the paper)")
        ax.axhspan(-0.3, 0.3, color="black", alpha=0.08, label="paper, Fig. S6A: distance-based curves (≈ 0 ± 0.3)")
        ax.set_title(f"{ttc.split()[0]}, analysis radius {R} m", fontsize=10)
        ax.set_xlim(0.5, TAU_HI); ax.set_ylim(-0.5, 2)
        if col == 0:
            ax.set_ylabel(r"$E(\tau)$ [arb. units]")
        if row == 1:
            ax.set_xlabel(r"$\tau$ [s]")
axes[0, 0].legend(fontsize=8)
plt.suptitle("Anticipatory vs. distance-based force, in the layout of the paper's Fig. S6 (D vs. A)\n"
             "both curves of a panel divided by the anticipatory curve's $\\tau^{-2}$ amplitude")
plt.tight_layout()
plt.savefig(RESULTS / "plot_energy_distance_control.png", dpi=130)
plt.show()
display(pd.DataFrame(rows).round(3))

fig, ax = plt.subplots(figsize=(13, 3))
ax.imshow(plt.imread(PAPER_IMG["figS6_models"])); ax.axis("off")
ax.set_title("Paper, Fig. S6: (A) distance-based [Helbing 2000] ... (D) this model", fontsize=9)
plt.show()
""")

md(r"""
**Verdict: the τ⁻² law comes back out of the simulation, as the paper
claims; the contrast with the distance-based force is reproduced, but not
as cleanly as Fig. 4 shows it.**

- *Shape (anticipatory force).* In all three scenes a τ⁻² law with no
  free exponent describes E(τ) with R² = 0.93-0.97, except the Evacuation
  at 0.1 m (0.74: at that radius its data only start at τ ≈ 1.6 s). Fitting
  the exponent freely gives 1.8-2.2 at 0.1 m for the Hallway and Bottleneck
  and 1.6 at 0.25 m. For calibration, the same unweighted fit gives 2.27
  and 2.04 on the authors' own Outdoor and Bottleneck data (they report
  2.05 and 2.02 with a bisquare-weighted fit), so the method itself carries
  about ±0.2. Scaled by their amplitude, our curves lie on the paper's
  digitized Fig. 4 curves over 1.2-2.5 s.
- *Range.* Our agents avoid short times to collision more completely than
  the paper's: with the paper's 0.1 m analysis radius, hardly any pair
  (fewer than 20 per 0.05 s window, pooled over seeds) has τ below 1.0 s
  (Hallway) or 1.2 s (Bottleneck), where the paper's Fig. 4 curves start
  at 0.75 s. With the agents' own radius (0.25 m) our curves
  do reach 0.5-0.8 s. We use the authors' published parameters and
  reference-code details unchanged; we did not tune them to move this
  boundary.
- *Distance-based control.* The second figure uses the layout of the
  paper's Fig. S6: the black line is the τ⁻² law, and both curves of a
  panel are divided by the amplitude of the anticipatory curve's τ⁻² fit,
  so that they are in the same arbitrary units as the paper's panels. In
  these units our distance-based E stays within 0.32 of zero over
  0.5-2.5 s at both analysis radii (0.25 m: at most 0.06 in the
  Bottleneck, 0.17 in the Hallway; 0.1 m: at most 0.25 and 0.32, reached
  only at the shortest τ), i.e. inside the ±0.3 spread of the paper's
  Fig. S6A curves and far from the τ⁻² line. Two caveats: in raw
  ln(1/g) the 0.1 m curves do rise with decreasing τ (to about 1 at
  0.5 s, vs. 3.5-4.9 for the anticipatory amplitude), a weak τ-dependence
  that only looks negligible next to the anticipatory curve; and the paper
  does not say how its distance-based panels were normalized, or which
  analysis radius it used for simulations. Dividing by the anticipatory
  amplitude is our choice.
""")

# ============================================================ 3. g(r)
md(r"""
## 3. g(r) depends on the rate of approach (Fig. S5 vs. Fig. 1c)

**Paper**: in the Hallway simulation, g(r) binned by the rate of approach
v = −dr/dt (Fig. S5A) shows the same strong dependence on v as the
Outdoor data (Fig. 1c): pairs closing slowly (0 < v ≤ 1 m/s) are often
found close together (a sharp peak just above contact, from lanes of
people walking the same way), fast-closing pairs (v > 2 m/s) are kept far
apart. With the distance-based force (Fig. S5B) the dependence on v is
"weaker".

**Ours**: the same Hallway runs as Section 2 (3 seeds per force), g(r) with
0.04 m bins over r < 8 m, pairs with v ≤ 0 excluded, time-scrambled
reference as before.
""")

code(r"""
V_BINS = [("0<v<=1", 0, 1, "tab:cyan"), ("1<v<=2", 1, 2, "tab:red"), ("v>2", 2, np.inf, "orange")]
r_edges = np.arange(0.0, 8.0 + 1e-9, 0.04)
pairs = {}
for scene in ["Hallway", "Hallway (distance)"]:
    obs, scr = [], []
    for k, tag in enumerate(SCENES[scene]):
        o, s = observed_and_scrambled(traj(tag), n_scrambles=3, seed=10 * k)
        obs.append(o); scr.append(s)
    pairs[scene] = (pd.concat(obs), pd.concat(scr))

g_r, rows = {}, []
for scene, (obs, scr) in pairs.items():
    for lab, lo, hi, _ in V_BINS:
        o, s = obs[(obs.v > lo) & (obs.v <= hi)], scr[(scr.v > lo) & (scr.v <= hi)]
        g, rc = pair_distribution(o.r, s.r, r_edges)
        g_r[(scene, lab)] = (rc, g)
        near = rc < 3
        i = np.nanargmax(np.where(near, g, np.nan))
        rows.append({"scene": scene, "approach rate": lab, "pairs": len(o),
                     "peak g (r<3 m)": g[i], "r at peak [m]": rc[i],
                     "r where g first > 0.5 [m]": rc[np.argmax(g > 0.5)]})
for key, paper in [("Hallway", PAPER_FIGS5A), ("Hallway (distance)", PAPER_FIGS5B)]:
    for lab, *_ in V_BINS:
        r, g = np.array(paper[lab]["r"]), np.array(paper[lab]["g"])
        i = np.argmax(np.where(r < 3, g, -1))
        rows.append({"scene": f"paper, Fig. S5{'A' if key == 'Hallway' else 'B'}", "approach rate": lab,
                     "pairs": np.nan, "peak g (r<3 m)": g[i], "r at peak [m]": r[i],
                     "r where g first > 0.5 [m]": r[np.argmax(g > 0.5)]})
gr_df = pd.DataFrame(rows)
gr_df.round(2)
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, scene, paper, panel, ymax in [(axes[0], "Hallway", PAPER_FIGS5A, "S5A", 6),
                                      (axes[1], "Hallway (distance)", PAPER_FIGS5B, "S5B", 3)]:
    for lab, lo, hi, col in V_BINS:
        rc, g = g_r[(scene, lab)]
        ax.plot(rc, g, color=col, lw=1.5, label=f"ours, {lab}")
        ax.plot(paper[lab]["r"], paper[lab]["g"], "--", color=col, lw=1, alpha=0.9)
    ax.plot([], [], "k--", lw=1, label=f"paper, Fig. {panel} (digitized, same colours)")
    ax.set_xlim(0, 3); ax.set_ylim(0, ymax)
    ax.set_xlabel("r [m]"); ax.set_ylabel("g(r)")
    ax.set_title(("Anticipatory force" if scene == "Hallway" else "Distance-based force") + f" vs. paper Fig. {panel}")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(RESULTS / "plot_gr_approach_rate.png", dpi=130)
plt.show()
""")

md(r"""
**Verdict: reproduced for the anticipatory force (shape, and position of
the features); not for the distance-based control.**

- *Anticipatory force vs. Fig. S5A.* Same ordering and same shapes: slowly
  approaching pairs show a sharp peak just above contact (g = 4.7 at
  r = 0.62 m; paper 5.4 at 0.57 m), 1 < v ≤ 2 m/s pairs a weak one
  (1.3 at 0.78 m; paper 1.6 at 0.79 m), and fast-approaching pairs are
  kept apart (g > 0.5 only beyond 1.06 m; paper 1.11 m), approaching 1 at
  a few metres. This is the strong v-dependence of the Outdoor data
  (Fig. 1c) that a purely distance-dependent interaction cannot produce.
- *Distance-based force vs. Fig. S5B.* The paper finds a weaker
  v-dependence for the distance-based force, with all three curves close
  to 1 beyond 1 m. Ours has a clear peak for slow pairs (2.1 at 0.86 m;
  paper 2.4 at 0.76 m), but its fast-approaching pairs (v > 2 m/s) stay
  suppressed out to 3 m (g ≈ 0.6), much more than in Fig. S5B. We have
  not found why; one candidate is that the paper's distance-based agents
  were set up differently (e.g. other A, B or radius), which it does not
  report.
""")

# ============================================================ 4. Fundamental diagram
md(r"""
## 4. Fundamental diagram (Fig. S4A)

**Paper**: "Each simulation area was divided into two-dimensional cells
measuring 0.5 m × 0.5 m and for each cell we determined its average
density and speed over 4 s intervals", with Edie's generalized
definitions, then "clustering the data into bins of 0.1 ppl/m²". Hallway,
Bottleneck (1 m wide) and Evacuation samples are plotted against
Weidmann's curve and called "in good agreement".

**Ours**: exactly that procedure (`edie_cells`, `binned_fundamental_diagram`)
on one run of each scene (seed 1): the whole walkable area, every 4 s
interval, only occupied cells. Weidmann's curve:
v = 1.34 [1 − exp(−1.913 (1/ρ − 1/5.4))].
""")

code(r"""
FD_RUNS = {"Hallway": "hallway_ttc", "Bottleneck": "bottleneck_w1.0", "Evacuation": "evacuation"}
FD_COLORS = {"Hallway": "orange", "Bottleneck": "tab:red", "Evacuation": "tab:cyan"}
MIN_CELLS = 10   # a density bin needs this many cell-intervals to be plotted

fd = {}
for scene, tag in FD_RUNS.items():
    b = binned_fundamental_diagram(edie_cells(traj(tag)))
    fd[scene] = b[b["n_cells"] >= MIN_CELLS]

def rms_to_weidmann(rho, v, lo, hi):
    rho, v = np.asarray(rho), np.asarray(v)
    m = (rho >= lo) & (rho < hi)
    return float(np.sqrt(np.mean((v[m] - weidmann_speed(rho[m])) ** 2))) if m.any() else np.nan

rows = []
for scene in FD_RUNS:
    paper = PAPER_FIGS4A[scene.lower()]
    for lo, hi in [(0, 1), (1, 2), (2, 5)]:
        rows.append({"scene": scene, "density range [1/m2]": f"{lo}-{hi}",
                     "ours: RMS to Weidmann [m/s]": rms_to_weidmann(fd[scene]["rho"], fd[scene]["speed"], lo, hi),
                     "paper: RMS to Weidmann [m/s]": rms_to_weidmann(paper["density"], paper["speed"], lo, hi),
                     "ours: max density bin": fd[scene]["rho"].max()})
fd_df = pd.DataFrame(rows)
fd_df.round(3)
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
rr = np.linspace(0.05, 5, 300)
for ax, (scene, col) in zip(axes, FD_COLORS.items()):
    ax.plot(fd[scene]["rho"], fd[scene]["speed"], "o", color=col, ms=5, label=f"ours, {scene}")
    paper = PAPER_FIGS4A[scene.lower()]
    ax.plot(paper["density"], paper["speed"], "^" if scene != "Hallway" else "D", mfc="none",
            color="black", ms=5, alpha=0.7, label=f"paper, Fig. S4A {scene} (digitized)")
    ax.plot(rr, weidmann_speed(rr), color="black", lw=1.5, label="Weidmann (1993)")
    ax.set_xlim(0, 5); ax.set_ylim(0, 1.6)
    ax.set_xlabel("density [1/m$^2$]"); ax.set_title(f"{scene}" + (" (1 m wide)" if scene == "Bottleneck" else ""))
    ax.legend(fontsize=8)
axes[0].set_ylabel("speed [m/s]")
plt.suptitle("Speed-density relation (Edie, 0.5 m cells, 4 s intervals, 0.1 /m$^2$ bins): ours vs. paper Fig. S4A")
plt.tight_layout()
plt.savefig(RESULTS / "plot_fundamental_diagram.png", dpi=130)
plt.show()
""")

md(r"""
**Verdict: partly reproduced. The Bottleneck and Evacuation data agree
with the paper's own Fig. S4A; the Hallway does not.**

- *Bottleneck (1 m) and Evacuation.* Both follow Weidmann at low density,
  fall below it between about 1.5 and 3 /m² and end at 0.1-0.2 m/s near
  4 /m², i.e. they sit where the paper's own markers for these scenes
  sit. Their RMS distance to Weidmann is of the same size as the paper's
  (0.18-0.34 vs. 0.17-0.46 m/s per density range, table above). Note
  that the paper's markers deviate from Weidmann as much as ours do: "good
  agreement" in the paper means roughly this level.
- *Hallway.* Up to about 1.2 /m² our speeds match Weidmann and the
  paper's markers. At higher local densities our walkers keep about
  1.1 m/s, while Weidmann and the paper's Hallway markers drop to
  0.2-0.5 m/s at 2-3 /m². Our high-density hallway cells are short-lived
  (dense only for a moment as the counter-flows interpenetrate in lanes),
  and our Hallway never reaches the paper's 3 /m²; the paper's Hallway
  geometry and start layout are not given, and a denser, longer
  counter-flow might slow down more. We did not tune the scene to match.
""")

# ============================================================ 5. Evacuation
md(r"""
## 5. Evacuation: arching at the exit (Fig. 3a)

**Paper**: "150 pedestrians exit a room (10 m wide × 24 m long) through a
narrow doorway. Due to the restricted movement of the pedestrians,
arch-like blockings are formed near the exit, leading to clogging
phenomena similar to the ones observed in granular media." Fig. 3a shows
a semicircular crowd around the door and pedestrians leaving in single
file.

**Ours** (`run_evacuation.py`): the same room and number of agents, starting
at random positions in the room. The door width is not given; the
authors' video shows single-file exiting, so we use 0.8 m (less than two
body widths). A 10 m deep area behind the door ends in the exit zone.

**What we measure**: the flow through the door; the *jam front*, i.e. for
15° sectors around the door, the distance of the farthest agent that is
part of the dense jam (≥ 3 neighbours within 0.8 m); an arch-shaped
(semicircular) jam has the same front distance in every sector, so we
report the front's coefficient of variation across sectors; and the time
gaps between successive agents passing the door, whose long tail is the
signature of temporary clogs in granular flow.
""")

code(r"""
s_ev = summary("evacuation")
df_ev = traj("evacuation")
door = (s_ev["door_x"], s_ev["door_y"])
t_door = passage_times(df_ev, s_ev["door_x"] + s_ev["wall"])
gaps = np.diff(t_door)
fps = 10.0
front_rows = []
for t in [20, 40, 60, 90]:
    snap = df_ev[np.isclose(df_ev["t"], t)]
    fr = jam_front(snap, door)
    front_rows.append({"t [s]": t, "agents in room": int((snap.pos_x < door[0]).sum()),
                       "mean front [m]": fr["front"].mean(),
                       "front CV across sectors": fr["front"].std() / fr["front"].mean(),
                       "sectors with jam": int(fr["front"].notna().sum())})
print(f"crashed: {s_ev['crashed']}; agents left in the simulation at the end: {s_ev['agents_left_in_sim']}")
print(f"agents through the door: {len(t_door)} of {s_ev['n_agents']}; last at t = {t_door[-1]:.1f} s")
print(f"steady flow: {steady_flow(t_door):.2f} /s  ({steady_flow(t_door) / s_ev['door_width']:.2f} /(m s))")
print(f"time gaps between passages: median {np.median(gaps):.2f} s, 90th pct {np.percentile(gaps, 90):.2f} s, "
      f"max {gaps.max():.2f} s; gaps > 3x median: {(gaps > 3 * np.median(gaps)).sum()}")
print(f"minimum clearance between agents: {min_clearance(df_ev[df_ev.frame % 5 == 0], 0.25):.3f} m")
pd.DataFrame(front_rows).round(2)
""")

code(r"""
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.9])
ax = fig.add_subplot(gs[0, 0])
ax.imshow(plt.imread(PAPER_IMG["fig3a_evacuation"])); ax.axis("off")
ax.set_title("Paper, Fig. 3a: arching around the exit", fontsize=9)
for k, t in enumerate([30, 60]):
    ax = fig.add_subplot(gs[0, 1 + k])
    snap = df_ev[np.isclose(df_ev["t"], t)]
    for x, y in zip(snap.pos_x, snap.pos_y):
        ax.add_patch(plt.Circle((x, y), MEAN_AGENT_RADIUS, color="tab:blue", alpha=0.8))
    L, W, dw, wall = s_ev["room_length"], s_ev["room_width"], s_ev["door_width"], s_ev["wall"]
    ax.fill_between([L, L + wall], 0, W / 2 - dw / 2, color="tab:red")
    ax.fill_between([L, L + wall], W / 2 + dw / 2, W, color="tab:red")
    ax.set_xlim(L - 8, L + 6); ax.set_ylim(0, W); ax.set_aspect("equal")
    ax.set_title(f"Ours, t = {t} s", fontsize=9); ax.set_xlabel("x [m]")
ax = fig.add_subplot(gs[1, 0:2])
for t in [20, 40, 60, 90]:
    fr = jam_front(df_ev[np.isclose(df_ev["t"], t)], door)
    ax.plot(fr["angle"], fr["front"], "o-", label=f"t = {t} s")
ax.set_xlabel("sector angle around the door [deg] (0 = straight into the room)")
ax.set_ylabel("jam front distance [m]"); ax.set_ylim(0, None)
ax.set_title("Jam front vs. angle: flat = semicircular arch", fontsize=9); ax.legend(fontsize=8)
ax = fig.add_subplot(gs[1, 2])
ax.plot(t_door, np.arange(1, len(t_door) + 1), color="tab:blue")
ax.set_xlabel("t [s]"); ax.set_ylabel("agents through the door")
ax.set_title(f"Outflow ({steady_flow(t_door):.2f} /s)", fontsize=9)
plt.tight_layout()
plt.savefig(RESULTS / "plot_evacuation.png", dpi=130)
plt.show()
""")

md(r"""
**Verdict: reproduced (qualitatively, as the paper states it).** The crowd
forms the semicircular jam of Fig. 3a within the first 20 s and keeps it
while the room empties: the jam front is the same distance from the door
in every 15° sector to within 4-7% (coefficient of variation), shrinking
from about 5.3 m to 4.2 m between t = 20 s and 90 s, and agents leave in
single file as in the paper's picture. The outflow (0.76 /s through
0.8 m) is intermittent: the median time between two agents passing the
door is about 1 s, but the longest gaps reach 3-6 s (14 gaps longer than
three times the median), the temporary blockages the paper compares to
granular media. The paper gives no numbers for this scenario, and the
door width is our assumption, so only the shape can be compared. Minimum
clearance stayed ≥ 0 (no overlaps).

**Replay.**
""")

code(r"""
traj_ev, area_ev = read_sqlite_file(str(RESULTS / "evacuation.sqlite"))
animate(traj_ev, area_ev, every_nth_frame=40, radius=MEAN_AGENT_RADIUS, title_note="Evacuation, 0.8 m door")
""")

# ============================================================ 6. Hallway lanes
md(r"""
## 6. Hallway: lane formation (Fig. 3b)

**Paper**: "300 pedestrians cross paths while walking from opposite ends of
an open hallway that is 20 m wide. The pedestrians dynamically form lanes
of uniform walking directions to efficiently resolve collisions."

**Ours** (`run_hallway.py`): 20 m wide, 50 m long (length not given),
150 agents spawned at random in a 10 m × 20 m block at each end, walking to
the far end. Same runs as Sections 2-4.

**What we measure**: a lane order parameter. For each agent in the central
section (15 m < x < 35 m), neighbours ahead or behind within a lateral band
|Δy| < 0.4 m and |Δx| < 3 m are counted as same- or opposite-direction;
φᵢ = ((n_same − n_opp)/(n_same + n_opp))², φ = ⟨φᵢ⟩ (1 = perfect lanes).
The reference φ_mixed recomputes φ after shuffling the direction labels
among the same agents (same positions, no lanes). φ is only evaluated
while the two groups actually pass through each other: at least 30
agents of each group inside the other group's extent along x.
""")

code(r"""
def lanes(tag):
    # Only frames where the groups really interpenetrate: at least 30 agents
    # of each group inside the other group's x-extent (10th-90th percentile).
    # Counting agents in the section is not enough: early on, both groups
    # are in it but still far apart, and phi = 1 trivially.
    s = summary(tag)
    df = traj(tag)
    lo = lane_order(df, s["group_east"], np.arange(2.5, 60, 0.5), x_range=(15, 35))
    east = set(s["group_east"])
    frames = df["frame"].unique()
    overlap = []
    for t in lo["t"]:
        g = df[(df["frame"] == frames[np.argmin(np.abs(frames / 10.0 - t))]) & df.pos_x.between(15, 35)]
        is_e = g["id"].isin(east)
        xe, xw = g.loc[is_e, "pos_x"], g.loc[~is_e, "pos_x"]
        n_e_in_w = xe.between(xw.quantile(0.1), xw.quantile(0.9)).sum()
        n_w_in_e = xw.between(xe.quantile(0.1), xe.quantile(0.9)).sum()
        overlap.append(min(n_e_in_w, n_w_in_e))
    lo["n_interpenetrating"] = overlap
    return lo[lo["n_interpenetrating"] >= 30]

lane_rows, lane_ts = [], {}
for force, tags in [("anticipatory", SCENES["Hallway"]), ("distance-based", SCENES["Hallway (distance)"])]:
    for tag in tags:
        lo = lanes(tag)
        lane_ts[tag] = lo
        lane_rows.append({"force": force, "run": tag, "mixing window [s]": f"{lo.t.min():.1f}-{lo.t.max():.1f}",
                          "phi": lo["phi"].mean(), "phi_mixed": lo["phi_mixed"].mean(),
                          "phi / phi_mixed": lo["phi"].mean() / lo["phi_mixed"].mean()})
lane_df = pd.DataFrame(lane_rows)
display(lane_df.round(3))
lane_df.groupby("force")[["phi", "phi_mixed", "phi / phi_mixed"]].agg(["mean", "std"]).round(3)
""")

code(r"""
fig = plt.figure(figsize=(14, 8.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9])
ax = fig.add_subplot(gs[0, 0])
ax.imshow(plt.imread(PAPER_IMG["fig3b_hallway"])); ax.axis("off")
ax.set_title("Paper, Fig. 3b: lanes in the hallway (colour = goal direction)", fontsize=9)
s_h = summary("hallway_ttc"); df_h = traj("hallway_ttc")
t_snap = float(lane_ts["hallway_ttc"]["t"].median())
ax = fig.add_subplot(gs[0, 1])
snap = df_h[np.isclose(df_h["t"], round(t_snap, 1))]
east = snap["id"].isin(set(s_h["group_east"]))
ax.scatter(snap.pos_x[east], snap.pos_y[east], s=14, color="tab:red", label="walking +x")
ax.scatter(snap.pos_x[~east], snap.pos_y[~east], s=14, color="tab:blue", label="walking -x")
ax.set_xlim(0, 50); ax.set_ylim(0, 20); ax.set_aspect("equal")
ax.set_title(f"Ours (anticipatory), t = {t_snap:.1f} s", fontsize=9); ax.legend(fontsize=8, loc="upper right")
ax = fig.add_subplot(gs[1, :])
for tag, col in [("hallway_ttc", "tab:red"), ("hallway_distance", "tab:blue")]:
    lo = lane_ts[tag]
    ax.plot(lo["t"], lo["phi"], color=col, label=f"{tag}: φ")
    ax.plot(lo["t"], lo["phi_mixed"], ":", color=col, label=f"{tag}: φ_mixed (labels shuffled)")
ax.set_xlabel("t [s]"); ax.set_ylabel("lane order φ"); ax.set_ylim(0, 1)
ax.set_title("Lane order while the two groups pass through each other (seed 1)", fontsize=9)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(RESULTS / "plot_hallway_lanes.png", dpi=130)
plt.show()
""")

code(r"""
traj_h, area_h = read_sqlite_file(str(RESULTS / "hallway_ttc.sqlite"))
animate(traj_h, area_h, every_nth_frame=20, radius=MEAN_AGENT_RADIUS, title_note="Hallway, anticipatory force")
""")

md(r"""
**Verdict: reproduced.** While the groups pass through each other, the
lane order is φ = 0.80 on average (3 seeds, ±0.01) against 0.37 for the
same positions with shuffled directions (ratio 2.2): walkers going the
same way line up behind each other, as in Fig. 3b. With the
distance-based force φ is 0.49 against 0.34 (ratio 1.45), so lanes are
much weaker there. The paper shows lanes only as a picture, so this is a
qualitative confirmation; the threshold φ_mixed is our own reference, not
the paper's.
""")

# ============================================================ 7. Bottleneck
md(r"""
## 7. Bottleneck: clogging and "zipping" (Fig. 3c)

**Paper**: "150 pedestrians start in a 5 m-wide waiting area and have to
pass through a 5 m-long bottleneck of variable width (1 m − 3 m). In all
cases, pedestrians exhibit clogging behaviors in the waiting area (Fig.
3c). With wider bottlenecks, 'zipping' patterns emerge inside the
constriction. For example, at a width of 2.5 m, pedestrians tend to walk
diagonally behind each other, dynamically forming 5-6 overlapping layers
that maximize the utility of the bottleneck."

**Ours** (`run_bottleneck.py`): widths 1.0, 1.5, 2.0, 2.5 and 3.0 m, 150
agents placed at random in a 15 m × 5 m waiting area (length not given;
2 /m², the authors' video shows a packed start), a 10 m × 10 m area
behind the bottleneck. Seed 1 for every width.

**What we measure**: flow through the bottleneck exit; the density just in
front of the entrance (Edie cells in the last 2 m of the waiting area)
while agents are queueing, as the clogging measure; and the lateral
position profile inside the bottleneck (1 m < x < 4 m), whose peaks are
the layers. Layers "overlap" (the zipper effect) when they are closer
than one body width (0.5 m).
""")

code(r"""
bn_rows, profiles = [], {}
for w in [1.0, 1.5, 2.0, 2.5, 3.0]:
    tag = f"bottleneck_w{w}"
    s = summary(tag); df = traj(tag)
    pt = passage_times(df, s["neck_length"])
    front = edie_cells(df, region=(-2.0, 0.0, -s["wait_width"] / 2, s["wait_width"] / 2))
    busy = front[front["ti"] * 4 < pt[int(0.9 * len(pt))]]      # while agents are still queueing
    centers, prof, peaks = lateral_layers(df, (1.0, 4.0), w)
    profiles[w] = (centers, prof, peaks)
    bn_rows.append({"width [m]": w, "crashed": s["crashed"], "agents through": len(pt),
                    "flow [1/s]": steady_flow(pt), "specific flow [1/(m s)]": steady_flow(pt) / w,
                    "density in front, median [1/m2]": busy["density"].median(),
                    "speed in front, median [m/s]": busy["speed"].median(),
                    "layers": len(peaks), "mean layer spacing [m]": np.diff(peaks).mean() if len(peaks) > 1 else np.nan,
                    "min clearance [m]": min_clearance(df[df.frame % 5 == 0], 0.25)})
bn_df = pd.DataFrame(bn_rows)
bn_df.round(3)
""")

code(r"""
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.85])
ax = fig.add_subplot(gs[0, 0])
ax.imshow(plt.imread(PAPER_IMG["fig3c_bottleneck"])); ax.axis("off")
ax.set_title("Paper, Fig. 3c: clogging in front of, and zipping inside, the bottleneck", fontsize=9)
ax = fig.add_subplot(gs[0, 1])
s25 = summary("bottleneck_w2.5"); df25 = traj("bottleneck_w2.5")
snap = df25[np.isclose(df25["t"], 15.0)]
for x, y in zip(snap.pos_x, snap.pos_y):
    ax.add_patch(plt.Circle((x, y), MEAN_AGENT_RADIUS, color="tab:blue", alpha=0.8))
W = 2.5
for sgn in (1, -1):
    ax.plot([-15, 0, 0, 5, 5, 15], [sgn * 2.5, sgn * 2.5, sgn * W / 2, sgn * W / 2, sgn * 5, sgn * 5], color="tab:red", lw=3)
ax.set_xlim(-8, 8); ax.set_ylim(-3, 3); ax.set_aspect("equal")
ax.set_title("Ours, 2.5 m wide, t = 15 s", fontsize=9); ax.set_xlabel("x [m]")
ax = fig.add_subplot(gs[1, 0])
for w in [2.0, 2.5, 3.0]:
    c, p, pk = profiles[w]
    ax.plot(c, p, label=f"{w} m: {len(pk)} layers")
    ax.plot(pk, np.interp(pk, c, p), "kx")
ax.set_xlabel("lateral position y inside the bottleneck [m]"); ax.set_ylabel("density of positions")
ax.set_title("Lateral profile inside the bottleneck (x = 1-4 m); x = detected layers", fontsize=9)
ax.legend(fontsize=8)
ax = fig.add_subplot(gs[1, 1])
ax.bar(range(len(bn_df)), bn_df["layers"], color="tab:blue", alpha=0.8, label="ours: layers")
ax.set_xticks(range(len(bn_df)), [f"{w:.1f}" for w in bn_df["width [m]"]])
ax.fill_between([2.6, 3.4], *PAPER_LAYERS_W25, color="black", alpha=0.25, label="paper: 5-6 overlapping layers at 2.5 m")
ax.set_xlabel("bottleneck width [m]"); ax.set_ylabel("number of layers")
ax.set_title("Layers inside the bottleneck vs. width", fontsize=9); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(RESULTS / "plot_bottleneck.png", dpi=130)
plt.show()
""")

md(r"""
**Verdict: clogging reproduced; zipping only partly; the number of layers
is not.**

- *Clogging in the waiting area, "in all cases"*: yes. At every width the
  crowd piles up in front of the entrance at a median 2.4-3.1 /m² (Edie
  cells in the last 2 m of the waiting area), moving at a median
  0.1-0.7 m/s, slowest for the narrowest bottleneck. The snapshot shows
  the same jammed block as Fig. 3c.
- *Zipping*: inside the 2.5 m bottleneck agents walk in lanes that are
  staggered, each agent diagonally behind its neighbours in the adjacent
  lanes, as in the paper's picture.
- *"5-6 overlapping layers" at 2.5 m*: **no.** We find 4 layers, 0.57 m
  apart on average, more than one body width (0.5 m), so the layers do
  not overlap sideways. The number of layers grows by one per 0.5 m of
  width (1 layer at 1 m, 5 at 3 m). Two things we checked: the peak count
  does not depend on the histogram bin width (0.02 or 0.04 m give the
  same peaks), and the outermost layers sit about 0.4 m from the walls,
  i.e. the anticipatory wall force keeps agents off them. The paper's
  agent radius is not given; we took 0.25 m from its own g(r) (Section
  3). A smaller radius would fit more layers, but we did not tune it to
  get this number.

Not a claim of the paper, but worth noting: the specific flow at 2.5-3 m
(2.6-2.7 /(m s)) is higher than typically measured in laboratory
bottleneck experiments (about 1.9 /(m s)).

**Replay** (2.5 m bottleneck).
""")

code(r"""
traj_b, area_b = read_sqlite_file(str(RESULTS / "bottleneck_w2.5.sqlite"))
animate(traj_b, area_b, every_nth_frame=10, radius=MEAN_AGENT_RADIUS, title_note="Bottleneck, 2.5 m wide")
""")

# ============================================================ 8. Crossing
md(r"""
## 8. Crossing: slowing down instead of deviating, and diagonal stripes (Fig. 3d)

**Paper**: "Two groups, of 40 pedestrians each, cross paths
perpendicularly. The pedestrians prefer to slow down and let others pass
rather than deviate from their planned courses. As such, homogeneous
clusters of pedestrians emerge within the two groups, leading to the
formation of diagonal line-shaped patterns."

**Ours** (`run_crossing.py`): two blocks of 5 × 8 agents (1 m spacing,
±0.1 m jitter), group A walking +x and group B walking +y, both blocks
starting 12 m from the crossing point so that they meet there at the same
time. The block layout is read off the authors' video. Three seeds at
dt = 0.01 s (the `dt_crossing_0.01*` runs of the dt study; `crossing` is
seed 1 again for the replay).

**What we measure**: per agent, its delay (time to the exit line minus
the free walking time at its preferred speed) and the part of that delay
explained by walking a longer path, (path length − straight distance) /
preferred speed; the rest comes from slowing down. For the stripes: the
orientation of links between same-group neighbours (< 1.2 m) inside the
central 12 m × 12 m while both groups are there, averaged as a nematic
order (angle of ⟨e^{2iφ}⟩). The two walking directions are 0° and 90°, so
a diagonal stripe is at ±45°. For flows along +x and +y, the relative
velocity is along (1, −1), i.e. −45°.
""")

code(r"""
cross_rows, per_frame = [], {}
for sfx in ["", "_s2", "_s3"]:
    tag = f"dt_crossing_0.01{sfx}"
    s = summary(tag); df = traj(tag)
    m = crossing_agent_metrics(df, s)
    angle, strength, pf = stripe_orientation(df, s)
    per_frame[tag] = pf
    both = df[(df.pos_x.abs() < 6) & (df.pos_y.abs() < 6)]
    a_in = both[both["id"].isin(set(s["group_a"]))].groupby("frame").size()
    b_in = both[~both["id"].isin(set(s["group_a"]))].groupby("frame").size()
    overlap = (a_in.reindex(b_in.index).fillna(0).clip(upper=b_in)).max()
    cross_rows.append({
        "seed": s["seed"], "reached exit": f"{int(m['reached'].sum())}/{len(m)}",
        "max agents of the smaller group in the centre together": int(overlap),
        "mean delay [s]": m["delay"].mean(), "mean detour time [s]": m["detour_time"].mean(),
        "slow-down share of delay": 1 - m["detour_time"].sum() / m["delay"].sum(),
        "mean max lateral offset [m]": m["max_lateral"].mean(),
        "stripe angle [deg]": angle, "stripe order S": strength,
        "min clearance [m]": min_clearance(df, 0.25),
    })
cross_df = pd.DataFrame(cross_rows)
cross_df.round(3)
""")

code(r"""
s_c = summary("dt_crossing_0.01"); df_c = traj("dt_crossing_0.01")
m_c = crossing_agent_metrics(df_c, s_c)
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.9])
ax = fig.add_subplot(gs[0, 0])
ax.imshow(plt.imread(PAPER_IMG["fig3d_crossing"])); ax.axis("off")
ax.set_title("Paper, Fig. 3d: diagonal stripes", fontsize=9)
pf = per_frame["dt_crossing_0.01"]
t_best = pf.loc[pf["S"].idxmax(), "t"]
for k, t in enumerate([t_best - 2, t_best]):
    ax = fig.add_subplot(gs[0, 1 + k])
    snap = df_c[np.isclose(df_c["t"], round(t, 1))]
    a = snap["id"].isin(set(s_c["group_a"]))
    ax.scatter(snap.pos_x[a], snap.pos_y[a], s=30, color="tab:blue", label="A, walking +x")
    ax.scatter(snap.pos_x[~a], snap.pos_y[~a], s=30, color="tab:red", label="B, walking +y")
    ax.plot([-6, 6], [6, -6], "k:", lw=1, label="-45° diagonal")
    ax.set_xlim(-7, 9); ax.set_ylim(-7, 9); ax.set_aspect("equal")
    ax.set_title(f"Ours, t = {t:.1f} s", fontsize=9); ax.legend(fontsize=7, loc="upper left")
ax = fig.add_subplot(gs[1, 0])
ax.scatter(m_c["detour_time"], m_c["delay"], s=15)
lim = max(m_c["delay"].max(), 0.5)
ax.plot([0, lim], [0, lim], "k:", lw=1, label="all delay from detour")
ax.plot([0, lim / 2], [0, lim], lw=1.5, label="paper: slowing ≥ 50% of delay (below this line: detour dominates)", **PAPER_STYLE)
ax.set_xlabel("detour time [s]"); ax.set_ylabel("delay [s]"); ax.legend(fontsize=7)
ax.set_title("Per agent (seed 1): delay vs. detour", fontsize=9)
ax = fig.add_subplot(gs[1, 1])
ax.bar(range(len(cross_df)), cross_df["slow-down share of delay"], color="tab:blue", alpha=0.8, label="ours")
ax.set_xticks(range(len(cross_df)), cross_df["seed"])
ax.axhline(PAPER_SLOWDOWN_SHARE, lw=1.5, label="paper: 'prefer to slow down' (> 50%)", **PAPER_STYLE)
ax.set_ylim(0, 1); ax.set_xlabel("seed"); ax.set_ylabel("share of delay from slowing down")
ax.legend(fontsize=7); ax.set_title("Slow down or deviate?", fontsize=9)
ax = fig.add_subplot(gs[1, 2])
ax.bar(range(len(cross_df)), cross_df["stripe angle [deg]"], color="tab:blue", alpha=0.8, label="ours")
ax.set_xticks(range(len(cross_df)), cross_df["seed"])
ax.axhline(-PAPER_STRIPE_ANGLE, lw=1.5, label="paper: diagonal (−45°)", **PAPER_STYLE)
ax.set_ylim(-90, 90); ax.set_xlabel("seed"); ax.set_ylabel("stripe axis [deg]")
ax.legend(fontsize=7); ax.set_title("Stripe orientation", fontsize=9)
plt.tight_layout()
plt.savefig(RESULTS / "plot_crossing.png", dpi=130)
plt.show()
""")

md(r"""
**Verdict: reproduced.** The groups meet in the centre as intended (up to
35-39 agents of the smaller group in the central 12 m × 12 m at once),
everyone crosses, and nobody overlaps. About 70-75% of the delay comes
from slowing down rather than from a longer path (paper: pedestrians
"prefer to slow down ... rather than deviate"); agents deviate on average
at most 0.6-0.8 m from their straight line. The same-group links inside
the crossing line up along −39° to −54° (−45° ± 7°), i.e. the diagonal
stripes of Fig. 3d, perpendicular to neither walking direction and
along the relative velocity of the two flows. The stripe order is weak
(S = 0.17-0.25; 1 would be perfect lines), which matches the loose,
ragged stripes of the paper's picture.

Some agents have a *negative* delay: the slowest walkers (preferred speed
0.5-0.7 m/s) are pushed forward by faster ones closing in from behind,
since Eq. S2 also acts between followers. The slow-down share uses the
sums over all agents.

**Replay.**
""")

code(r"""
traj_c, area_c = read_sqlite_file(str(RESULTS / "crossing.sqlite"))
animate(traj_c, area_c, every_nth_frame=6, radius=MEAN_AGENT_RADIUS, title_note="Crossing, 2 x 40 agents")
""")

# ============================================================ 9. Collective motion
md(r"""
## 9. Collective motion without goals (Fig. 3e)

**Paper**: "750 pedestrians are placed in an enclosed square area of size
40 × 40 m, and at each time step are propelled forward in the direction
of their current velocity without having a specific goal. Pedestrians are
initially given random orientations, and after a long enough time they
spontaneously form a vortex pattern in which all pedestrians are walking
in unison." The main text adds that the crowd, "initialized to a high
energy state with many imminent collisions settles over time into a low
energy state".

**Ours** (`run_collective.py`, `self_propelled=True`: the preferred
velocity is the preferred speed along the agent's own current velocity).
**Deviation, forced by compute time:** at the paper's size this
pure-Python callback reached only 20 s of simulated time in about 25 min
(every agent reads ~150 neighbours within the 10 m range every step), and
showed no ordering yet by then (below). We therefore keep the paper's
density (750 / 1600 m² = 0.47 /m²) in a 20 m × 20 m box with 188 agents,
2 seeds, 400 s each. The walls, the model, dt and every parameter are
unchanged. A smaller box also shortens the time needed to circulate
around it, so the time to order is not comparable with the paper's.

**What we measure**, every 5 s: the milling (vortex) order
M = |⟨r̂ᵢ × v̂ᵢ⟩| about the box centre (1 = everyone circling the same way),
the polarization |⟨v̂ᵢ⟩| (≈ 0 for a vortex), and the mean interaction
energy per agent, Σⱼ k τᵢⱼ⁻² e^(−τᵢⱼ/τ₀) (the paper's Eq. 2 with the
model's own radii).
""")

code(r"""
coll = {}
for sd in (1, 2):
    tag = f"collective_n188_s{sd}"
    df = traj(tag)
    coll[tag] = collective_order(df, np.arange(0, df["t"].max() + 0.01, 5.0))
    s = summary(tag)
    last = coll[tag][coll[tag]["t"] >= coll[tag]["t"].max() - 100]
    print(f"{tag}: crashed={s['crashed']}, simulated {s['sim_time_completed']:.0f} s; "
          f"last 100 s: milling {last['milling'].mean():.3f} ± {last['milling'].std():.3f}, "
          f"polarization {last['polarization'].mean():.3f}, energy/agent {last['energy_per_agent'].mean():.3f} "
          f"(t=0: {coll[tag]['energy_per_agent'].iloc[0]:.2f}); "
          f"first time milling > 0.8: {coll[tag].loc[coll[tag]['milling'] > 0.8, 't'].min():.0f} s")
full = load_trajectory(RESULTS / "collective_n750_partial.sqlite")[0]
print("\nPaper-size run (750 agents, 40 m box), stopped after 20 s:")
collective_order(full, [0.5, 10.0, 19.5]).round(3)
""")

code(r"""
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[0.9, 1])
ax = fig.add_subplot(gs[0, 0:2])
ax.imshow(plt.imread(PAPER_IMG["fig3e_collective"])); ax.axis("off")
ax.set_title("Paper, Fig. 3e: random initial state (left), vortex steady state (right)", fontsize=9)
ax = fig.add_subplot(gs[0, 2])
for tag, ls in [("collective_n188_s1", "-"), ("collective_n188_s2", "--")]:
    c = coll[tag]
    ax.plot(c["t"], c["milling"], ls, color="tab:blue", label=f"milling ({tag[-2:]})")
    ax.plot(c["t"], c["polarization"], ls, color="tab:orange", label=f"polarization ({tag[-2:]})")
ax.axhline(PAPER_MILLING, lw=1.5, label="paper: all walking in unison (vortex)", **PAPER_STYLE)
ax.set_xlabel("t [s]"); ax.set_ylim(0, 1.05); ax.legend(fontsize=7)
ax.set_title("Order parameters", fontsize=9)
df1 = traj("collective_n188_s1")
half = summary("collective_n188_s1")["arena_half"]
for k, t in enumerate([0.0, 60.0, float(np.floor(df1["t"].max()))]):
    ax = fig.add_subplot(gs[1, k])
    snap = df1[np.isclose(df1["t"], t)]
    ax.quiver(snap.pos_x, snap.pos_y, snap.vx, snap.vy, angles="xy", scale=30, width=0.004)
    ax.plot([-half, half, half, -half, -half], [-half, -half, half, half, -half], color="0.5", lw=3)
    ax.set_aspect("equal"); ax.set_xlim(-half - 1, half + 1); ax.set_ylim(-half - 1, half + 1)
    ax.set_title(f"Ours, t = {t:.0f} s", fontsize=9)
plt.tight_layout()
plt.savefig(RESULTS / "plot_collective.png", dpi=130)
plt.show()

fig, ax = plt.subplots(figsize=(6, 3.5))
for tag, ls in [("collective_n188_s1", "-"), ("collective_n188_s2", "--")]:
    ax.plot(coll[tag]["t"], coll[tag]["energy_per_agent"], ls, color="tab:red", label=tag)
ax.set_yscale("log"); ax.set_xlabel("t [s]"); ax.set_ylabel("interaction energy per agent")
ax.set_title("From a high- to a low-energy state", fontsize=9); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(RESULTS / "plot_collective_energy.png", dpi=130); plt.show()
""")

md(r"""
**Verdict: reproduced, at a quarter of the paper's area.** From random
headings, both seeds organize into the vortex of Fig. 3e: a ring of
walkers circulating along the walls around an empty centre. The milling
order passes 0.8 after 45-55 s and then stays at 0.91-0.92 for the rest
of the 400 s, while the polarization stays low (0.1-0.2, as it must for a
vortex). The mean interaction energy per agent falls by a factor of
20-40 (from 1.3-2.2 to about 0.05), the paper's "high energy state ...
settles over time into a low energy state". Milling saturates below 1
because the square's corners bend the flow. Differences from the paper:
the box is 20 m instead of 40 m (at the same density), and our ring is
thinner than the paper's, whose picture shows a denser crowd. The
paper-size run we started showed no ordering within the 20 s it reached
(milling 0.05), which says nothing about its long-time state.

**Replay** (188 agents, seed 1; one frame every 10 s).
""")

code(r"""
traj_col, area_col = read_sqlite_file(str(RESULTS / "collective_n188_s1.sqlite"))
animate(traj_col, area_col, every_nth_frame=100, radius=MEAN_AGENT_RADIUS, title_note="Goal-less walkers, 188 agents")
""")

# ============================================================ Summary
md(r"""
## 10. Summary

| # | Phenomenon (paper) | Reproduced? | Notes |
|---|---|---|---|
| 1 | Anticipation: early avoidance head-on, no reaction side by side (Fig. 1a/b) | ✅ | head-on: steering starts ~5 m apart, peak < 0.5 m/s² (distance-based: < 1 m apart, ~8 m/s²); side by side: zero force, vs. pushed apart with the distance-based force |
| 2 | Simulations give back E ∝ τ⁻² (Fig. 4, Fig. S4B) | ✅ shape / ⚠️ range | τ⁻² fits with R² = 0.93-0.97 (Evacuation at 0.1 m: 0.74); free exponent 1.8-2.2 (0.1 m radius) or 1.6 (0.25 m), vs. 2.04-2.27 for the authors' data with the same fit; with the paper's 0.1 m analysis radius our agents almost never get below τ ≈ 1.0-1.2 s, where the paper's curves start at 0.75 s |
| 3 | Distance-based force: E shows no τ-dependence (Fig. 4 inset, S6A) | ✅ in the paper's units / ⚠️ normalization | scaled like Fig. S6 (by the anticipatory curve's τ⁻² amplitude), our distance-based E stays within 0.32 of zero at both analysis radii, inside the paper's ±0.3 band; in raw ln(1/g) the 0.1 m curves still show a weak rise at short τ; the paper states neither its normalization for these panels nor its simulation analysis radius |
| 4 | g(r) depends strongly on the approach rate (Fig. S5A vs. Fig. 1c) | ✅ | same ordering, shapes and feature positions as Fig. S5A (slow-pair peak 4.7 at 0.62 m vs. 5.4 at 0.57 m) |
| 5 | Weaker v-dependence with the distance-based force (Fig. S5B) | ❌ | our fast-approaching pairs stay suppressed to 3 m (g ≈ 0.6) where the paper's are ≈ 1; cause not found |
| 6 | Speed-density data agree with Weidmann (Fig. S4A) | ⚠️ | Bottleneck (1 m) and Evacuation sit where the paper's own markers sit (similar RMS to Weidmann); Hallway keeps ~1.1 m/s up to 2.5 /m² where Weidmann and the paper drop to 0.2-0.5 m/s |
| 7 | Arching and clogging at a narrow exit (Fig. 3a) | ✅ | semicircular jam (front varies 4-7% across sectors), single-file exit, intermittent outflow; door width assumed (0.8 m) |
| 8 | Lanes in the hallway (Fig. 3b) | ✅ | lane order 0.80 vs. 0.37 with shuffled directions (distance-based: 0.49 vs. 0.34) |
| 9 | Clogging at every bottleneck width; zipping at 2.5 m (Fig. 3c) | ✅ clogging / ⚠️ zipping / ❌ layer count | 2.4-3.1 /m² jam at every width; staggered lanes, but 4 non-overlapping layers (0.57 m apart) at 2.5 m, not "5-6 overlapping" |
| 10 | Crossing: slow down rather than deviate; diagonal stripes (Fig. 3d) | ✅ | 70-75% of the delay from slowing down; stripes at −45° ± 7° (weak order, S ≈ 0.2) |
| 11 | Goal-less walkers form a vortex, energy decreases (Fig. 3e) | ✅ (reduced size) | milling 0.91-0.92 after ~50 s, energy per agent ÷20-40; 188 agents in 20 m × 20 m at the paper's density (paper size not feasible in pure Python) |

**Overall.** The model's defining property is reproduced exactly:
anticipation instead of distance (row 1), and with it the paper's central
self-consistency check (row 2). Put through the paper's own
statistical-mechanics analysis, our simulated trajectories give back an
inverse-square interaction energy, with exponents within the method's own
scatter of the authors' measurement on real crowds. The qualitative crowd
phenomena of Fig. 3 all appear. Where we differ, it is in details the
paper leaves open or only shows as pictures: the analysis radius used for
simulations and the normalization of the distance-based panels (rows 2-3), the number of layers in the bottleneck (row 9),
the hallway's high-density speeds (row 6), and the distance-based
control's g(r) (row 5). For each of these, the setup choices we had to
make (agent radius 0.25 m, geometry lengths, door width, the
Helbing-2000 constants for the control) are stated in the corresponding
section, and none was tuned to improve agreement.
""")


nb["cells"] = cells
with open("Universal_Power_Law_validation.ipynb", "w") as f:
    nbf.write(nb, f)

print(f"Wrote Universal_Power_Law_validation.ipynb with {len(cells)} cells")
