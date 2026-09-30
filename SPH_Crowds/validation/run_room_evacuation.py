# SPDX-License-Identifier: LGPL-3.0-or-later
"""Scenario 1 of van Toll et al. (2021), Section 6.1 / Fig. 4(a) / Table 1:
400 agents leave a 20 x 20 m room through a 0.8 m wide door.

Geometry (read off Fig. 4(a), 1 m grid): room x in [0, 20], y in [-10, 10];
a 1 m thick wall x in [20, 21] with the door at y in [-0.4, 0.4]; a 2 m deep
area outside. Goal 1 m beyond the exit, at (22, 0); an agent is removed
once it is within 0.5 m of it (an exit stage = disk of radius 0.5).
Agents start on a 20 x 20 grid with 1 m spacing.

Measured (as in the paper): number of evacuated agents, the mean (and std)
SPH density of all agents at t = 15 s, and the flow rate from the first to
the 350th evacuee.

Usage:
    python run_room_evacuation.py --profile SPH --rho0-max 5 --tag room_SPH_rho5
    python run_room_evacuation.py --profile SF --k-ag 250 --tag room_SF_k250
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jupedsim as jps
from jupedsim.sqlite_serialization import SqliteTrajectoryWriter
from pySPH_Crowds import SPHCrowdModel, make_state

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

GOAL = (22.0, 0.0)


def build_geometry(door_width: float):
    room = shapely.box(0.0, -10.0, 20.0, 10.0)
    door = shapely.box(20.0 - 1e-3, -door_width / 2, 21.0 + 1e-3, door_width / 2)
    outside = shapely.box(21.0, -10.0, 23.0, 10.0)
    return shapely.union_all([room, door, outside]).simplify(0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="SPH")
    p.add_argument("--k-ag", type=float, default=1000.0, help="K_ag of SF/RVO profiles")
    p.add_argument("--rho0-max", type=float, default=5.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--max-time", type=float, default=400.0)
    p.add_argument("--n-side", type=int, default=20)
    p.add_argument("--door-width", type=float, default=0.8)
    p.add_argument("--rvo-samples", type=int, default=100, help="RVO candidate velocities (paper: 'many')")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--every-nth-frame", type=int, default=5)
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    walkable = build_geometry(args.door_width)
    model = SPHCrowdModel(seed=args.seed)
    db_path = RESULTS / f"{args.tag}.sqlite"
    writer = SqliteTrajectoryWriter(output_file=db_path, every_nth_frame=args.every_nth_frame)
    sim = jps.Simulation(model=model, geometry=walkable, dt=args.dt, trajectory_writer=writer)

    exit_stage = sim.add_exit_stage(shapely.Point(GOAL).buffer(0.5, quad_segs=8))
    journey = sim.add_journey(jps.JourneyDescription([exit_stage]))

    rng = np.random.default_rng(args.seed)
    n = args.n_side
    spacing = 20.0 / n
    ids = []
    for i in range(n):
        for j in range(n):
            pos = (spacing * (i + 0.5), -10.0 + spacing * (j + 0.5))
            st = make_state(rng, args.profile, uid=len(ids), k_ag_nav=args.k_ag, rho0_max=args.rho0_max,
                            rvo_samples=args.rvo_samples)
            ids.append(sim.add_agent(journey_id=journey, stage_id=exit_stage, position=pos, state=st))
    n_total = len(ids)

    exit_times = []
    density_15 = None
    snapshot_15 = []
    density_series = []  # (t, mean, std, n)
    step_times = []
    crashed = None
    n_steps = int(round(args.max_time / args.dt))
    try:
        for it in range(n_steps):
            t0 = time.perf_counter()
            before = sim.agent_count()
            sim.iterate()
            step_times.append(time.perf_counter() - t0)
            t = sim.elapsed_time()
            exit_times.extend([t] * (before - sim.agent_count()))
            if it % int(round(1.0 / args.dt)) == 0 or abs(t - 15.0) < args.dt / 2:
                dens = np.array([a.state.density for a in sim.agents()])
                if dens.size:
                    density_series.append((t, float(dens.mean()), float(dens.std()), int(dens.size)))
                if abs(t - 15.0) < args.dt / 2:
                    density_15 = {"mean": float(dens.mean()), "std": float(dens.std()), "n": int(dens.size)}
                    snapshot_15 = [[a.position[0], a.position[1], a.state.density] for a in sim.agents()]
            if sim.agent_count() == 0:
                break
    except Exception as e:  # noqa: BLE001
        crashed = {"step": len(step_times), "time": sim.elapsed_time(), "error": repr(e)}
    writer.close()

    flow = None
    if len(exit_times) >= 350:
        flow = 349.0 / (exit_times[349] - exit_times[0])
    summary = {
        "scenario": "room_evacuation",
        "profile": args.profile,
        "k_ag_nav": args.k_ag,
        "rho0_max": args.rho0_max,
        "rvo_samples": args.rvo_samples,
        "dt": args.dt,
        "max_time": args.max_time,
        "n_total": n_total,
        "door_width": args.door_width,
        "seed": args.seed,
        "goal": GOAL,
        "n_evacuated": len(exit_times),
        "exit_times": exit_times,
        "density_15s": density_15,
        "density_series": density_series,
        "snapshot_15s": snapshot_15,
        "flow_rate_first_to_350th": flow,
        "sim_time_reached": sim.elapsed_time(),
        "n_steps_completed": len(step_times),
        "mean_step_time_ms": float(np.mean(step_times) * 1000) if step_times else None,
        "wall_clock_s": float(np.sum(step_times)),
        "db_path": str(db_path),
        "crashed": crashed,
    }
    with open(RESULTS / f"{args.tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{args.tag}: evacuated {len(exit_times)}/{n_total}, rho(15s)={density_15}, "
          f"flow={flow}, t={sim.elapsed_time():.1f}s, crashed={crashed}, "
          f"{np.sum(step_times):.0f}s wall")


if __name__ == "__main__":
    main()
