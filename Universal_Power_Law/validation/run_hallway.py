# SPDX-License-Identifier: LGPL-3.0-or-later
"""Hallway scenario (Supplemental Material, "Hallway"; paper Fig. 3b and
the source of Fig. 4, Fig. S4, Fig. S5): 300 pedestrians cross paths
while walking from opposite ends of an open hallway that is 20 m wide.

The paper does not give the hallway's length or the start layout. We use
a 50 m long hallway, 150 agents spawned at random in a 10 m x 20 m block
at each end. Each agent's goal is straight ahead: its own 1 m wide exit box
in the last 4 m of the far end, at its starting y (the authors' reference
code gives every agent a goal point and removes it there). One exit
polygon spanning the whole end would not do: JuPedSim steers agents
towards a single point of the polygon, which made both groups funnel
towards the hallway's centre line.

`--interaction distance` runs the same scene with the distance-based
force of Helbing et al. (2000), the paper's control (Fig. 4 inset, S5B, S6A).

Usage:
    python run_hallway.py --tag hallway_ttc
    python run_hallway.py --interaction distance --tag hallway_distance
"""

import argparse

import numpy as np
import shapely

from common import (
    AGENT_RADIUS, PowerLawState, UniversalPowerLawModel, jps, make_writer,
    pref_speeds, run_loop, sample_positions, write_summary,
)

LENGTH, WIDTH = 50.0, 20.0
N_PER_SIDE = 150
SPAWN = {"east": (4.0, 14.0), "west": (36.0, 46.0)}   # x-ranges of the spawn blocks
EXIT_DEPTH = 4.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--sim-time", type=float, default=90.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--interaction", choices=["ttc", "distance"], default="ttc")
    p.add_argument("--n-per-side", type=int, default=N_PER_SIDE)
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    geometry = shapely.box(0.0, 0.0, LENGTH, WIDTH)
    writer, db_path = make_writer(args.tag, args.dt)
    sim = jps.Simulation(model=UniversalPowerLawModel(), geometry=geometry, dt=args.dt, trajectory_writer=writer)

    exit_x = {"east": (LENGTH - EXIT_DEPTH, LENGTH), "west": (0.0, EXIT_DEPTH)}
    sign = {"east": 1.0, "west": -1.0}

    groups = {"east": [], "west": []}
    for name, (x0, x1) in SPAWN.items():
        pos = sample_positions(rng, args.n_per_side, (x0, x1), (0.4, WIDTH - 0.4), 2 * AGENT_RADIUS + 0.1)
        for q, v0 in zip(pos, pref_speeds(rng, args.n_per_side)):
            y_lo, y_hi = max(q[1] - 0.5, 0.0), min(q[1] + 0.5, WIDTH)
            stage = sim.add_exit_stage(shapely.box(exit_x[name][0], y_lo, exit_x[name][1], y_hi))
            journey = sim.add_journey(jps.JourneyDescription([stage]))
            aid = sim.add_agent(
                journey_id=journey, stage_id=stage, position=tuple(q),
                state=PowerLawState(
                    velocity=(sign[name] * v0, 0.0), pref_speed=v0,
                    radius=AGENT_RADIUS, interaction=args.interaction,
                ),
            )
            groups[name].append(aid)

    result = run_loop(sim, writer, args.sim_time)
    write_summary(args.tag, {
        "scenario": "hallway", "dt": args.dt, "seed": args.seed,
        "interaction": args.interaction, "agent_radius": AGENT_RADIUS,
        "length": LENGTH, "width": WIDTH, "n_per_side": args.n_per_side,
        "spawn_x": SPAWN, "exit_depth": EXIT_DEPTH,
        "group_east": groups["east"], "group_west": groups["west"],
        "db_path": str(db_path), **result,
    })


if __name__ == "__main__":
    main()
