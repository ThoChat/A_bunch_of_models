# SPDX-License-Identifier: LGPL-3.0-or-later
"""Crossing scenario (Supplemental Material, "Crossing"; paper Fig. 3d):
two groups of 40 pedestrians cross paths perpendicularly. The paper
reports that pedestrians "prefer to slow down and let others pass rather
than deviate from their planned courses", and that diagonal line-shaped
patterns (stripes) form.

Group A (ids in summary["group_a"]) walks +x, group B walks +y. Both
blocks start the same distance from the centre, so they meet there at the
same time (checked afterwards in the notebook from the trajectories).

Usage:
    python run_crossing.py --tag crossing
    python run_crossing.py --dt 0.05 --tag dt_crossing_0.05
"""

import argparse

import numpy as np
import shapely

from common import (
    AGENT_RADIUS, PowerLawState, UniversalPowerLawModel, jps, make_writer,
    pref_speeds, run_loop, write_summary,
)

HALF = 20.0          # open square arena, 40 m x 40 m
ROWS, COLS = 5, 8    # 40 agents per group: 5 deep x 8 abreast
SPACING = 1.0
START = 12.0         # distance of each block's centre from the crossing centre


def block(rng, center, along, across):
    pts = []
    for i in range(ROWS):
        for j in range(COLS):
            offset = (i - (ROWS - 1) / 2) * SPACING * along + (j - (COLS - 1) / 2) * SPACING * across
            pts.append(center + offset + rng.uniform(-0.1, 0.1, 2))
    return np.array(pts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--sim-time", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--interaction", choices=["ttc", "distance"], default="ttc")
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    geometry = shapely.box(-HALF, -HALF, HALF, HALF)
    writer, db_path = make_writer(args.tag, args.dt)
    sim = jps.Simulation(model=UniversalPowerLawModel(), geometry=geometry, dt=args.dt, trajectory_writer=writer)

    ex, ey = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    pos_a = block(rng, np.array([-START, 0.0]), ex, ey)
    pos_b = block(rng, np.array([0.0, -START]), ey, ex)
    speeds = pref_speeds(rng, 2 * ROWS * COLS)

    group_a, group_b, pref = [], [], {}
    for idx, (pos, direction, group) in enumerate(
        [(q, ex, group_a) for q in pos_a] + [(q, ey, group_b) for q in pos_b]
    ):
        v0 = speeds[idx]
        # Each agent's goal is straight ahead: a 1 m wide exit box in the
        # last 3 m of the arena, in its own lane. A single exit spanning the
        # whole side would make JuPedSim steer everyone to one point of it.
        if direction is ex:
            box = shapely.box(HALF - 3, pos[1] - 0.5, HALF, pos[1] + 0.5)
        else:
            box = shapely.box(pos[0] - 0.5, HALF - 3, pos[0] + 0.5, HALF)
        stage = sim.add_exit_stage(box)
        journey = sim.add_journey(jps.JourneyDescription([stage]))
        aid = sim.add_agent(
            journey_id=journey, stage_id=stage, position=tuple(pos),
            state=PowerLawState(
                velocity=tuple(v0 * direction), pref_speed=v0,
                radius=AGENT_RADIUS, interaction=args.interaction,
            ),
        )
        group.append(aid)
        pref[aid] = float(v0)

    result = run_loop(sim, writer, args.sim_time)
    write_summary(args.tag, {
        "scenario": "crossing",
        "dt": args.dt, "seed": args.seed, "interaction": args.interaction,
        "agent_radius": AGENT_RADIUS, "rows": ROWS, "cols": COLS,
        "spacing": SPACING, "start_distance": START, "arena_half": HALF,
        "group_a": group_a, "group_b": group_b,
        "pref_speed": {str(k): v for k, v in pref.items()},
        "db_path": str(db_path), **result,
    })


if __name__ == "__main__":
    main()
