# SPDX-License-Identifier: LGPL-3.0-or-later
"""Two-pedestrian anticipation test (paper Fig. 1a/1b and the main-text
claim of "anticipatory collision avoidance"): real pedestrians on a
collision course accelerate strongly to avoid each other "even though they
are far from each other" (Fig. 1a), while two pedestrians walking side by
side, close together, show no relative acceleration (Fig. 1b).

Two configurations, each run with the anticipatory ("ttc") and the
distance-based ("distance") force:
  * headon:   start 16 m apart, walking at each other at 1.3 m/s with a
              0.2 m lateral offset (an exact head-on pair has no sideways
              force by symmetry);
  * parallel: side by side 0.7 m apart (0.2 m gap), same velocity.

Usage:
    python run_two_agents.py --config headon --interaction ttc --tag two_headon_ttc
"""

import argparse

import numpy as np
import shapely

from common import (
    AGENT_RADIUS, PowerLawState, UniversalPowerLawModel, jps, make_writer,
    run_loop, write_summary,
)

SPEED = 1.3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=["headon", "parallel"], required=True)
    p.add_argument("--interaction", choices=["ttc", "distance"], default="ttc")
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--sim-time", type=float, default=14.0)
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    geometry = shapely.box(-15.0, -10.0, 15.0, 10.0)
    writer, db_path = make_writer(args.tag, args.dt)
    sim = jps.Simulation(model=UniversalPowerLawModel(), geometry=geometry, dt=args.dt, trajectory_writer=writer)

    if args.config == "headon":
        agents = [((-8.0, -0.1), (14.0, -0.1)), ((8.0, 0.1), (-14.0, 0.1))]
    else:
        agents = [((-8.0, -0.35), (14.0, -0.35)), ((-8.0, 0.35), (14.0, 0.35))]

    ids = []
    for start, goal in agents:
        wp = sim.add_waypoint_stage(goal, 0.3)
        journey = sim.add_journey(jps.JourneyDescription([wp]))
        direction = np.sign(goal[0] - start[0])
        ids.append(sim.add_agent(
            journey_id=journey, stage_id=wp, position=start,
            state=PowerLawState(velocity=(direction * SPEED, 0.0), pref_speed=SPEED,
                                radius=AGENT_RADIUS, interaction=args.interaction),
        ))

    result = run_loop(sim, writer, args.sim_time, stop_when_empty=False, progress_every=100.0)
    write_summary(args.tag, {
        "scenario": "two_agents", "config": args.config, "interaction": args.interaction,
        "dt": args.dt, "speed": SPEED, "agent_radius": AGENT_RADIUS,
        "agents": agents, "agent_ids": ids, "db_path": str(db_path), **result,
    })


if __name__ == "__main__":
    main()
