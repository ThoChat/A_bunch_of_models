# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bottleneck scenario (Supplemental Material, "Bottleneck"; paper Fig. 3c,
and a source of Fig. 4 / S4): 150 pedestrians start in a 5 m wide waiting
area and pass through a 5 m long bottleneck of variable width (1 m - 3 m).
The paper reports clogging in the waiting area for every width and, at
2.5 m, "zipping": 5-6 overlapping layers inside the constriction.

The paper does not give the waiting area's length. We use 15 m, which
holds the 150 agents at 2 /m^2 (the authors' video shows a packed start).
Behind the bottleneck is a 10 m x 10 m area with a 3 m deep exit zone.

Usage:
    python run_bottleneck.py --width 2.5 --tag bottleneck_w2.5
"""

import argparse

import numpy as np
import shapely

from common import (
    AGENT_RADIUS, PowerLawState, UniversalPowerLawModel, jps, make_writer,
    pref_speeds, run_loop, sample_positions, write_summary,
)

WAIT_L, WAIT_W = 15.0, 5.0
NECK_L = 5.0
OUT_L, OUT_W = 10.0, 10.0
N_AGENTS = 150


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=float, required=True)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--sim-time", type=float, default=240.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--interaction", choices=["ttc", "distance"], default="ttc")
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    w = args.width
    geometry = shapely.union_all([
        shapely.box(-WAIT_L, -WAIT_W / 2, 0.0, WAIT_W / 2),
        shapely.box(-0.01, -w / 2, NECK_L + 0.01, w / 2),
        shapely.box(NECK_L, -OUT_W / 2, NECK_L + OUT_L, OUT_W / 2),
    ])
    writer, db_path = make_writer(args.tag, args.dt)
    sim = jps.Simulation(model=UniversalPowerLawModel(), geometry=geometry, dt=args.dt, trajectory_writer=writer)
    exit_stage = sim.add_exit_stage(shapely.box(NECK_L + OUT_L - 3.0, -OUT_W / 2, NECK_L + OUT_L, OUT_W / 2))
    journey = sim.add_journey(jps.JourneyDescription([exit_stage]))

    pos = sample_positions(rng, N_AGENTS, (-WAIT_L + 0.3, -0.3), (-WAIT_W / 2 + 0.3, WAIT_W / 2 - 0.3), 2 * AGENT_RADIUS + 0.02)
    ids = []
    for q, v0 in zip(pos, pref_speeds(rng, N_AGENTS)):
        ids.append(sim.add_agent(
            journey_id=journey, stage_id=exit_stage, position=tuple(q),
            state=PowerLawState(velocity=(0.0, 0.0), pref_speed=v0,
                                radius=AGENT_RADIUS, interaction=args.interaction),
        ))

    result = run_loop(sim, writer, args.sim_time)
    write_summary(args.tag, {
        "scenario": "bottleneck", "dt": args.dt, "seed": args.seed,
        "interaction": args.interaction, "agent_radius": AGENT_RADIUS,
        "width": w, "wait_length": WAIT_L, "wait_width": WAIT_W,
        "neck_length": NECK_L, "out_length": OUT_L, "out_width": OUT_W,
        "n_agents": N_AGENTS, "agent_ids": ids, "db_path": str(db_path), **result,
    })


if __name__ == "__main__":
    main()
