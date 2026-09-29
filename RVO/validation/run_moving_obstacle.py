# SPDX-License-Identifier: LGPL-3.0-or-later
"""Moving Obstacle scenario (van den Berg, Lin & Manocha 2008, Section VI):
a handful of pedestrians cross a street on which a car drives straight
through, non-reactively (the car doesn't avoid anyone -- pedestrians must
avoid it). Tests correct handling of a fast, non-cooperating obstacle,
using this model's `is_reactive=False` agent state (see pyrvo.py): such an
agent ignores every neighbor/wall and drives straight at its own preferred
velocity, while OTHER agents see it via `neighbor.state.is_reactive` and
fall back to plain (non-reciprocal) VO against it, since a non-cooperating
obstacle won't take its half of the avoidance responsibility.

Usage:
    python run_moving_obstacle.py --n-pedestrians 11 --tag moving_obstacle
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-pedestrians", type=int, default=11)
    p.add_argument("--dt", type=float, default=0.025)
    p.add_argument("--sim-time", type=float, default=25.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--agent-radius", type=float, default=0.25)
    p.add_argument("--max-speed", type=float, default=1.4)
    p.add_argument("--street-half-length", type=float, default=16.0)
    p.add_argument("--street-half-width", type=float, default=8.0)
    p.add_argument("--car-radius", type=float, default=0.9)
    p.add_argument("--car-speed", type=float, default=4.5)
    # Timed so the car reaches the first pedestrian as the line arrives at its
    # lane: it then splits the line roughly in half, as in the paper's Fig. 10.
    p.add_argument("--car-start-x", type=float, default=-13.2)
    p.add_argument("--ped-start-y", type=float, default=-2.5)
    p.add_argument("--tag", default="moving_obstacle")
    p.add_argument("--every-nth-frame", type=int, default=1)
    args = p.parse_args()

    geometry = [
        (-args.street_half_length, -args.street_half_width),
        (args.street_half_length, -args.street_half_width),
        (args.street_half_length, args.street_half_width),
        (-args.street_half_length, args.street_half_width),
    ]
    model = ReciprocalVelocityObstacleModel()
    db_path = RESULTS / f"{args.tag}.sqlite"
    writer = SqliteTrajectoryWriter(output_file=db_path, every_nth_frame=args.every_nth_frame)
    sim = jps.Simulation(model=model, geometry=geometry, dt=args.dt, trajectory_writer=writer)

    # The car drives straight along the street's long axis (the "road"),
    # oblivious to everyone -- a fixed waypoint far past the far end.
    car_start = (args.car_start_x, 0.0)
    car_target = (args.street_half_length - 0.5, 0.0)
    car_wp = sim.add_waypoint_stage(car_target, 0.3)
    car_journey = sim.add_journey(jps.JourneyDescription([car_wp]))
    car_id = sim.add_agent(
        journey_id=car_journey,
        stage_id=car_wp,
        position=car_start,
        state=RVOState(
            velocity=(args.car_speed, 0.0),
            radius=args.car_radius,
            max_speed=args.car_speed,
            is_reactive=False,
        ),
    )

    # Pedestrians cross the street perpendicular to the car's path (the
    # "sidewalk to sidewalk" direction), starting staggered along the
    # street's length so they don't all meet the car at once.
    rng = np.random.default_rng(args.seed)
    agents = [car_id]
    targets = {}
    xs = np.linspace(
        -args.street_half_length * 0.6, args.street_half_length * 0.6, args.n_pedestrians
    )
    for x in xs:
        y0 = args.ped_start_y
        y1 = args.street_half_width - 1.0
        start = (x + rng.uniform(-0.3, 0.3), y0)
        target = (x + rng.uniform(-0.3, 0.3), y1)
        wp = sim.add_waypoint_stage(target, 0.2)
        journey = sim.add_journey(jps.JourneyDescription([wp]))
        agent_id = sim.add_agent(
            journey_id=journey,
            stage_id=wp,
            position=start,
            state=RVOState(
                velocity=(0.0, 0.0), radius=args.agent_radius, max_speed=args.max_speed,
            ),
        )
        agents.append(agent_id)
        targets[agent_id] = target

    n_steps = int(args.sim_time / args.dt)
    crashed = None
    step_times = []
    try:
        for i in range(n_steps):
            t0 = time.perf_counter()
            sim.iterate()
            step_times.append(time.perf_counter() - t0)
    except Exception as e:  # noqa: BLE001
        crashed = {"step": len(step_times), "time": sim.elapsed_time(), "error": repr(e)}

    writer.close()

    summary = {
        "n_pedestrians": args.n_pedestrians,
        "dt": args.dt,
        "seed": args.seed,
        "agent_radius": args.agent_radius,
        "max_speed": args.max_speed,
        "car_id": car_id,
        "car_radius": args.car_radius,
        "car_speed": args.car_speed,
        "car_start": car_start,
        "n_steps_completed": len(step_times),
        "mean_step_time_ms": float(np.mean(step_times) * 1000) if step_times else None,
        "db_path": str(db_path),
        "targets": {str(k): v for k, v in targets.items()},
        "agent_ids": agents,
        "pedestrian_ids": agents[1:],
        "waypoint_tolerance": 0.2,
        "crashed": crashed,
    }
    json_path = RESULTS / f"{args.tag}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {json_path}: {len(step_times)} steps completed")


if __name__ == "__main__":
    main()
