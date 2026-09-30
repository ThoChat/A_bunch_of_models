# SPDX-License-Identifier: LGPL-3.0-or-later
"""Pieces shared by every run_<scenario>.py: imports, agent sampling, the
guarded simulation loop and the JSON summary. Kept separate from
analysis.py, which only reads trajectories and does not need jupedsim."""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import jupedsim as jps  # noqa: E402
from jupedsim.sqlite_serialization import SqliteTrajectoryWriter  # noqa: E402
from pyUniversal_Power_Law import PowerLawState, UniversalPowerLawModel  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# Paper (Supplemental Material): preferred speeds ~ N(1.3, 0.3) m/s. Clipped
# so that no agent gets a negative or implausibly large preferred speed.
PREF_SPEED_MEAN = 1.3
PREF_SPEED_STD = 0.3
PREF_SPEED_CLIP = (0.5, 2.1)
AGENT_RADIUS = 0.25
OUTPUT_FPS = 10.0


def pref_speeds(rng, n):
    return np.clip(
        rng.normal(PREF_SPEED_MEAN, PREF_SPEED_STD, n), *PREF_SPEED_CLIP
    )


def sample_positions(rng, n, xlim, ylim, min_dist, max_tries=200000):
    """Uniform random positions in a box, at least `min_dist` apart."""
    pts = []
    tries = 0
    while len(pts) < n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(f"could only place {len(pts)} of {n} agents")
        p = np.array([rng.uniform(*xlim), rng.uniform(*ylim)])
        if all(np.hypot(*(p - q)) >= min_dist for q in pts):
            pts.append(p)
    return np.array(pts)


def make_writer(tag, dt):
    every = max(1, int(round(1.0 / (OUTPUT_FPS * dt))))
    db_path = RESULTS / f"{tag}.sqlite"
    if db_path.exists():
        db_path.unlink()
    return SqliteTrajectoryWriter(output_file=db_path, every_nth_frame=every), db_path


def run_loop(sim, writer, t_max, stop_when_empty=True, progress_every=10.0, on_step=None):
    """Iterate until `t_max` (or until everyone has left), recording a crash
    instead of losing the run. Always closes the writer."""
    n_steps = int(round(t_max / sim.delta_time()))
    crashed = None
    t_wall = time.perf_counter()
    next_report = progress_every
    try:
        for _ in range(n_steps):
            sim.iterate()
            if on_step is not None:
                on_step(sim)
            if stop_when_empty and sim.agent_count() == 0:
                break
            if sim.elapsed_time() >= next_report:
                print(
                    f"  t={sim.elapsed_time():6.1f}s agents={sim.agent_count():4d} "
                    f"wall={time.perf_counter() - t_wall:7.1f}s",
                    flush=True,
                )
                next_report += progress_every
    except Exception as e:  # noqa: BLE001 - recorded in the summary
        crashed = {
            "step": sim.iteration_count(),
            "time": sim.elapsed_time(),
            "error": repr(e),
        }
        print(f"  CRASHED: {crashed}", flush=True)
    writer.close()
    return {
        "crashed": crashed,
        "sim_time_completed": sim.elapsed_time(),
        "iterations": sim.iteration_count(),
        "agents_left_in_sim": sim.agent_count(),
        "wall_time_s": time.perf_counter() - t_wall,
    }


def write_summary(tag, summary):
    path = RESULTS / f"{tag}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"Wrote {path}")
    return path


__all__ = [
    "jps", "PowerLawState", "UniversalPowerLawModel", "RESULTS", "AGENT_RADIUS",
    "OUTPUT_FPS", "pref_speeds", "sample_positions", "make_writer", "run_loop",
    "write_summary",
]
