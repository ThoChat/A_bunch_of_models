# SPDX-License-Identifier: LGPL-3.0-or-later
"""Collective-motion scenario (Supplemental Material, "Collective motion";
paper Fig. 3e): 750 pedestrians in an enclosed 40 m x 40 m square, "at each
time step propelled forward in the direction of their current velocity
without having a specific goal". Initially random orientations; "after a
long enough time they spontaneously form a vortex pattern in which all
pedestrians are walking in unison", settling from a high- to a low-energy
state (main text).

Compute time forced a smaller box: at the paper's size (750 agents,
40 m x 40 m) this pure-Python callback reached only 20 s of simulated
time in ~25 min (each agent reads ~150 neighbours within the 10 m range
every step), with no ordering yet. `--half` and `--n` shrink the box at
the paper's density (750 / 1600 m^2 = 0.47 /m^2): 188 agents in 20 m x 20 m.

Agents use `self_propelled=True` (preferred velocity = preferred speed
along the current velocity). JuPedSim still needs a journey, so each agent
gets a waypoint; the model ignores its direction.

Usage:
    python run_collective.py --n 188 --half 10 --sim-time 400 --tag collective_n188
"""

import argparse

import numpy as np
import shapely

from common import (
    AGENT_RADIUS, PowerLawState, UniversalPowerLawModel, jps, make_writer,
    pref_speeds, run_loop, sample_positions, write_summary,
)

HALF = 20.0
N_AGENTS = 750


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--sim-time", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n", type=int, default=N_AGENTS)
    p.add_argument("--half", type=float, default=HALF, help="half side of the square [m]")
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    half = args.half
    geometry = shapely.box(-half, -half, half, half)
    writer, db_path = make_writer(args.tag, args.dt)
    sim = jps.Simulation(model=UniversalPowerLawModel(), geometry=geometry, dt=args.dt, trajectory_writer=writer)
    wp = sim.add_waypoint_stage((0.0, 0.0), 0.1)
    journey = sim.add_journey(jps.JourneyDescription([wp]))

    pos = sample_positions(rng, args.n, (-half + 0.4, half - 0.4), (-half + 0.4, half - 0.4), 2 * AGENT_RADIUS + 0.1)
    angles = rng.uniform(0.0, 2 * np.pi, args.n)
    ids = []
    for q, v0, ang in zip(pos, pref_speeds(rng, args.n), angles):
        ids.append(sim.add_agent(
            journey_id=journey, stage_id=wp, position=tuple(q),
            state=PowerLawState(
                velocity=(v0 * np.cos(ang), v0 * np.sin(ang)), pref_speed=v0,
                radius=AGENT_RADIUS, self_propelled=True,
            ),
        ))

    result = run_loop(sim, writer, args.sim_time, stop_when_empty=False, progress_every=20.0)
    write_summary(args.tag, {
        "scenario": "collective", "dt": args.dt, "seed": args.seed,
        "agent_radius": AGENT_RADIUS, "arena_half": half, "n_agents": args.n,
        "density": args.n / (2 * half) ** 2,
        "agent_ids": ids, "db_path": str(db_path), **result,
    })


if __name__ == "__main__":
    main()
