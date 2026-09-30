# SPDX-License-Identifier: LGPL-3.0-or-later
"""Evacuation scenario (Supplemental Material, "Evacuation"; paper Fig. 3a):
150 pedestrians exit a room (10 m wide x 24 m long) through a narrow
doorway; arch-like blockings form near the exit.

The paper does not give the door width. The authors' video shows
pedestrians leaving in single file, so we use 0.8 m (less than two agent
diameters of 2 x 0.5 m). The door is in the middle of one 10 m wall; a
10 m x 10 m area outside it ends in a 3 m deep exit zone.

Usage:
    python run_evacuation.py --tag evacuation
"""

import argparse

import numpy as np
import shapely

from common import (
    AGENT_RADIUS, PowerLawState, UniversalPowerLawModel, jps, make_writer,
    pref_speeds, run_loop, sample_positions, write_summary,
)

ROOM_L, ROOM_W = 24.0, 10.0
WALL = 0.2
OUTSIDE_L = 10.0
N_AGENTS = 150


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--sim-time", type=float, default=240.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--door-width", type=float, default=0.8)
    p.add_argument("--interaction", choices=["ttc", "distance"], default="ttc")
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    yc = ROOM_W / 2
    x_out = ROOM_L + WALL
    geometry = shapely.union_all([
        shapely.box(0.0, 0.0, ROOM_L, ROOM_W),
        shapely.box(ROOM_L - 0.01, yc - args.door_width / 2, x_out + 0.01, yc + args.door_width / 2),
        shapely.box(x_out, 0.0, x_out + OUTSIDE_L, ROOM_W),
    ])
    writer, db_path = make_writer(args.tag, args.dt)
    sim = jps.Simulation(model=UniversalPowerLawModel(), geometry=geometry, dt=args.dt, trajectory_writer=writer)
    exit_stage = sim.add_exit_stage(shapely.box(x_out + OUTSIDE_L - 3.0, 0.0, x_out + OUTSIDE_L, ROOM_W))
    journey = sim.add_journey(jps.JourneyDescription([exit_stage]))

    pos = sample_positions(rng, N_AGENTS, (0.4, ROOM_L - 0.4), (0.4, ROOM_W - 0.4), 2 * AGENT_RADIUS + 0.1)
    ids = []
    for q, v0 in zip(pos, pref_speeds(rng, N_AGENTS)):
        ids.append(sim.add_agent(
            journey_id=journey, stage_id=exit_stage, position=tuple(q),
            state=PowerLawState(velocity=(0.0, 0.0), pref_speed=v0,
                                radius=AGENT_RADIUS, interaction=args.interaction),
        ))

    result = run_loop(sim, writer, args.sim_time)
    write_summary(args.tag, {
        "scenario": "evacuation", "dt": args.dt, "seed": args.seed,
        "interaction": args.interaction, "agent_radius": AGENT_RADIUS,
        "room_length": ROOM_L, "room_width": ROOM_W, "wall": WALL,
        "door_width": args.door_width, "door_x": ROOM_L, "door_y": yc,
        "n_agents": N_AGENTS, "agent_ids": ids, "db_path": str(db_path), **result,
    })


if __name__ == "__main__":
    main()
