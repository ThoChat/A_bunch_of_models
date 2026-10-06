# A bunch of models

Reimplementations of classic microscopic crowd-simulation models as Python
operational models for [JuPedSim](https://github.com/PedestrianDynamics/jupedsim),
each with a notebook that tries to reproduce the validation results of the
model's source paper.

The models were chosen from the survey by W. van Toll and J. Pettré,
*Algorithms for Microscopic Crowd Simulation: Advancements in the 2010s*,
Computer Graphics Forum 40(2), 2021 ([10.1111/cgf.142664](https://doi.org/10.1111/cgf.142664)).
[`crowd-simulation-models.md`](crowd-simulation-models.md) lists every model
the survey discusses, with the paper that introduced it.

The aim is not to tune each model until it looks right. Each notebook
compares directly with the paper's own figures and numbers, and it says
plainly where the reimplementation does not reproduce the paper.

## Models

| Folder | Model | Source paper | Type |
|---|---|---|---|
| [`SFM/`](SFM) | Social Force Model with granular/panic extensions | Helbing, Farkas, Vicsek, *Nature* 2000 | force-based |
| [`Universal_Power_Law/`](Universal_Power_Law) | Time-to-collision power-law force | Karamouzas, Skinner, Guy, *PRL* 2014 | force-based |
| [`RVO/`](RVO) | Reciprocal Velocity Obstacles | van den Berg, Lin, Manocha, *ICRA* 2008 | velocity-based (sampling) |
| [`ORCA/`](ORCA) | Optimal Reciprocal Collision Avoidance | van den Berg, Guy, Lin, Manocha, *ISRR* 2011 | velocity-based (linear programming) |
| [`PLEdestrians/`](PLEdestrians) | Least-effort (biomechanical energy) steering | Guy et al., *SCA* 2010 | velocity-based |
| [`SPH_Crowds/`](SPH_Crowds) | Smoothed Particle Hydrodynamics for extreme densities, blended with SF/RVO | van Toll, Chatagnon et al., *Computers & Graphics* 2021 | hybrid (force + density) |

Each folder's `model_description.md` explains how the model works, what it
is good for, and how the original paper validated it. The final cell of each
notebook has a table that rates every phenomenon from the paper as
reproduced ✅, partly reproduced ⚠️ or not reproduced ❌, with the numbers
behind each rating.

## Folder layout

Every model folder has the same structure:

```
<MODEL>/
  model_description.md      # summary of the model and of the paper's validation
  <paper>.pdf               # source paper
  py<model>.py              # the model: frozen state dataclass + CustomOperationalModel subclass
  build_notebook.py         # generates the validation notebook (do not edit the .ipynb by hand)
  <MODEL>_validation.ipynb  # executed validation notebook
  validation/
    analysis.py             # trajectory loading and metrics
    run_<scenario>.py       # one script per experiment in the paper
    results/                # .sqlite trajectories, .json summaries, plots, figures cropped from the paper
```

Each notebook contains, for every experiment in the paper: the paper's setup
next to ours, a table of results, plots with the paper's data overlaid
(digitized by hand from the PDF where the paper has numeric plots), a
trajectory animation, and a verdict.

## Other files

- [`model_validation_instructions.md`](model_validation_instructions.md):
  the step-by-step procedure used to implement and validate each model
  (layout, environment, the JuPedSim API, how to compare against the paper,
  pitfalls found along the way, reporting standard). Use it to add a new
  model; `SFM/` and `RVO/` are the reference examples.
- [`crowd-simulation-models.md`](crowd-simulation-models.md): the catalogue
  of models from the survey.
- `all_papers/`: the survey paper.

## Setup

The models run on a local build of JuPedSim that provides
`jupedsim.models.custom_model.CustomOperationalModel`, which lets an
operational model be written in Python. JuPedSim is not part of this
repository: clone it into `jupedsim/` at the repository root and build it
there, with its Python environment in `.venv-python-model/` (Python 3.12).

The Python packages the models, notebooks and the JuPedSim build need are
listed in [`requirements.txt`](requirements.txt). Building JuPedSim also
needs CMake, Ninja and a C++20 compiler (e.g. `brew install cmake ninja` on
macOS). First-time setup, from the repository root:

```bash
python3.12 -m venv .venv-python-model
source .venv-python-model/bin/activate
pip install -r requirements.txt

git clone https://github.com/PedestrianDynamics/jupedsim.git
cmake -S jupedsim -B jupedsim/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DPython_EXECUTABLE="$PWD/.venv-python-model/bin/python"
cmake --build jupedsim/build
```

Then, in every new shell:

```bash
source .venv-python-model/bin/activate
source jupedsim/build/environment        # puts the local JuPedSim build on PYTHONPATH
```

After pulling a newer JuPedSim, rebuild with `cmake --build jupedsim/build`:
the Python sources and the compiled bindings must come from the same
version, otherwise `import jupedsim` fails.

`SFM/compile_and_run.sh` rebuilds JuPedSim with ninja and then runs a
script; it is only needed when JuPedSim itself has changed.

## Running

Each runner takes its parameters on the command line and writes
`validation/results/<tag>.sqlite` and `<tag>.json`. For example:

```bash
cd RVO/validation
python run_circle.py --n 12 --mode rvo --seed 1 --tag circle_n12_rvo
```

Regenerate and execute a notebook after changing `build_notebook.py`:

```bash
cd RVO
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace RVO_validation.ipynb
```

The first code cell of each notebook adds the JuPedSim build to `sys.path`
itself, so the notebooks also open directly in Jupyter or VS Code once the
venv is selected as the kernel.

## Caveats

- The models are written in pure Python and called once per agent per step,
  so they are much slower than the papers' C++ implementations. Results
  reproduce the shape of the papers' timing curves, not their absolute
  speed, and some large scenarios were run at a reduced size. Each notebook
  states where this happened.
- Values digitized from the papers' figures are approximate. Each notebook
  states its method and the expected error.

## License

MIT, see [`LICENSE`](LICENSE). The papers in this repository remain under
the copyright of their publishers.
