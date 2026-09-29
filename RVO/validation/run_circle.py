# SPDX-License-Identifier: LGPL-3.0-or-later
"""Circle scenario (van den Berg, Lin & Manocha 2008, Section VI):
N agents placed evenly on a circle, each walking straight to its own
antipodal point, forcing a dense crossing in the middle.

Used for two things in the validation notebook:
  * small N (12), run once with `--mode vo` (plain, non-reciprocal Velocity
    Obstacles) and once with `--mode rvo` (reciprocal averaging), same seed
    and positions -- checks the paper's oscillation-removal claim (Theorem
    8) by comparing path smoothness between the two.
  * larger N, to check collision-freedom (Theorem 6: minimum separation
    should never go below the combined agent radius) and to measure how
    per-step wall-clock time scales with N (the paper's own scalability
    claim, Section VI-B).

Usage:
    python run_circle.py --n 12 --mode rvo --seed 1 --tag circle_n12_rvo
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import jupedsim as jps
from jupedsim.sqlite_serialization import SqliteTrajectoryWriter
from pyrvo import ReciprocalVelocityObstacleModel, RVOState

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


def build_simulation(
    n: int,
    reciprocal: bool,
    dt: float,
    spacing_factor: float,
    agent_radius: float,
    max_speed: float,
    seed: int,
    writer: SqliteTrajectoryWriter | None,
):
    circumference = n * (2 * agent_radius) * spacing_factor
    radius = max(3.0, circumference / (2 * np.pi))
    bound = radius * 1.4
    geometry = [(-bound, -bound), (bound, -bound), (bound, bound), (-bound, bound)]

    model = ReciprocalVelocityObstacleModel()
    sim = jps.Simulation(model=model, geometry=geometry, dt=dt, trajectory_writer=writer)

    rng = np.random.default_rng(seed)
    # A tiny random angular jitter breaks the otherwise-exact antipodal
    # symmetry of every single pair -- without it, every agent's opposite
    # number is on a perfectly collinear reciprocal collision course, which
    # is the most degenerate case the sampling-based selection can face.
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(
        -0.01, 0.01, n
    )
    agents = []
    targets = {}
    for angle in angles:
        pos = (radius * np.cos(angle), radius * np.sin(angle))
        target = (-pos[0], -pos[1])
        wp = sim.add_waypoint_stage(target, 0.15)
        journey = sim.add_journey(jps.JourneyDescription([wp]))
        agent_id = sim.add_agent(
            journey_id=journey,
            stage_id=wp,
            position=pos,
            state=RVOState(
                velocity=(0.0, 0.0),
                radius=agent_radius,
                max_speed=max_speed,
                reciprocal=reciprocal,
            ),
        )
        agents.append(agent_id)
        targets[agent_id] = target
    return sim, agents, targets, radius


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--mode", choices=["rvo", "vo"], default="rvo")
    p.add_argument("--dt", type=float, default=0.025)
    p.add_argument("--sim-time", type=float, default=25.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--agent-radius", type=float, default=0.25)
    p.add_argument("--max-speed", type=float, default=1.5)
    p.add_argument("--spacing-factor", type=float, default=4.0)
    p.add_argument("--tag", required=True)
    p.add_argument("--every-nth-frame", type=int, default=1)
    p.add_argument("--benchmark-only", action="store_true", help="skip trajectory recording, just measure timing")
    args = p.parse_args()

    reciprocal = args.mode == "rvo"
    writer = None
    db_path = RESULTS / f"{args.tag}.sqlite"
    if not args.benchmark_only:
        writer = SqliteTrajectoryWriter(output_file=db_path, every_nth_frame=args.every_nth_frame)

    sim, agents, targets, radius = build_simulation(
        args.n, reciprocal, args.dt, args.spacing_factor, args.agent_radius,
        args.max_speed, args.seed, writer,
    )

    n_steps = int(args.sim_time / args.dt)
    step_times = []
    crashed = None
    min_clearance_running = float("inf")
    try:
        for i in range(n_steps):
            t0 = time.perf_counter()
            sim.iterate()
            step_times.append(time.perf_counter() - t0)
            if i % 20 == 0:
                pos = np.array([sim.agent(a).position for a in agents])
                d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
                np.fill_diagonal(d, np.inf)
                clearance = d.min() - 2 * args.agent_radius
                min_clearance_running = min(min_clearance_running, clearance)
    except Exception as e:  # noqa: BLE001 - recorded, not swallowed silently
        crashed = {"step": len(step_times), "time": sim.elapsed_time(), "error": repr(e)}

    if writer is not None:
        writer.close()

    summary = {
        "n": args.n,
        "mode": args.mode,
        "reciprocal": reciprocal,
        "dt": args.dt,
        "seed": args.seed,
        "agent_radius": args.agent_radius,
        "max_speed": args.max_speed,
        "circle_radius": radius,
        "n_steps_completed": len(step_times),
        "mean_step_time_ms": float(np.mean(step_times) * 1000) if step_times else None,
        "median_step_time_ms": float(np.median(step_times) * 1000) if step_times else None,
        "min_clearance_running": min_clearance_running,
        "db_path": str(db_path) if writer is not None else None,
        "targets": {str(k): v for k, v in targets.items()},
        "agent_ids": agents,
        "crashed": crashed,
    }
    json_path = RESULTS / f"{args.tag}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
