# SPDX-License-Identifier: LGPL-3.0-or-later
"""Circle scenario scalability sweep (van den Berg, Lin & Manocha 2008,
Section VI-B): measures per-step wall-clock time and collision-freedom
(minimum inter-agent clearance) as a function of N, the paper's own
scalability check (their Fig. 5: frame time vs. number of agents, tested up
to N=1000).

For N <= 100 a full trajectory is recorded (used elsewhere in the notebook
for replay/inspection); for larger N only timing/clearance are measured
(`--benchmark-only` in run_circle.build_simulation), since a full-resolution
trajectory for 1000 agents over many steps is unnecessary disk churn for a
number we only need once for a timing plot.

Usage:
    python run_circle_scaling.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import jupedsim as jps
from jupedsim.sqlite_serialization import SqliteTrajectoryWriter
from run_circle import build_simulation

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

DT = 0.025
AGENT_RADIUS = 0.25
MAX_SPEED = 1.5
SPACING_FACTOR = 4.0
SEED = 1


def run_one(n: int, record_trajectory: bool, sim_time: float):
    writer = None
    db_path = RESULTS / f"circle_scaling_n{n}.sqlite"
    if record_trajectory:
        writer = SqliteTrajectoryWriter(output_file=db_path, every_nth_frame=1)

    sim, agents, targets, radius = build_simulation(
        n, True, DT, SPACING_FACTOR, AGENT_RADIUS, MAX_SPEED, SEED, writer
    )

    n_steps = int(sim_time / DT)
    step_times = []
    min_clearance = float("inf")
    crashed = None
    try:
        for i in range(n_steps):
            t0 = time.perf_counter()
            sim.iterate()
            step_times.append(time.perf_counter() - t0)
            if i % 10 == 0:
                pos = np.array([sim.agent(a).position for a in agents])
                d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
                np.fill_diagonal(d, np.inf)
                min_clearance = min(min_clearance, d.min() - 2 * AGENT_RADIUS)
    except Exception as e:  # noqa: BLE001
        crashed = {"step": len(step_times), "error": repr(e)}

    if writer is not None:
        writer.close()

    return {
        "n": n,
        "sim_time_requested": sim_time,
        "n_steps_completed": len(step_times),
        "mean_step_time_ms": float(np.mean(step_times) * 1000) if step_times else None,
        "median_step_time_ms": float(np.median(step_times) * 1000) if step_times else None,
        "min_clearance": min_clearance if np.isfinite(min_clearance) else None,
        "recorded_trajectory": record_trajectory,
        "db_path": str(db_path) if record_trajectory else None,
        "crashed": crashed,
    }


def main():
    results = []
    for n, sim_time in [(12, 20.0), (50, 20.0), (100, 20.0)]:
        print(f"Running N={n} (full trajectory)...")
        results.append(run_one(n, record_trajectory=True, sim_time=sim_time))
    for n in [250, 500, 1000]:
        print(f"Running N={n} (timing only)...")
        results.append(run_one(n, record_trajectory=False, sim_time=3.0))

    out_path = RESULTS / "circle_scaling.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")
    for r in results:
        print(
            f"  N={r['n']:>5}  mean={r['mean_step_time_ms']:.2f} ms/step  "
            f"min_clearance={r['min_clearance']}"
        )


if __name__ == "__main__":
    main()
