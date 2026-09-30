# SPDX-License-Identifier: LGPL-3.0-or-later
"""Narrow Passage benchmark of Guy et al. (2010), Sec. 5.1 / Fig. 6b:
"100 agents must pass through a narrow passage to reach their goals", used
to show jamming/bottlenecks, arching at the entrance and the wake effect
("people slowly spread out after the narrow passage rather than filling the
available space immediately").

The paper gives no dimensions. Our reading of Fig. 6b: a crowd in an open
area squeezes between two crate walls into open space. Geometry: an
upstream room (x in [-14, 0]) and a downstream room (x in [L, L + 14]),
both 16 m wide, joined by a passage of width --gap and length L. The 100
agents start on a jittered lattice upstream; each has its own goal on the
downstream room's far wall (random y), so any lack of spreading after the
exit comes from the dynamics, not from a shared goal point.

Usage:
    python run_narrow_passage.py --tag narrow_passage
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import shapely

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import jupedsim as jps
from jupedsim.sqlite_serialization import SqliteTrajectoryWriter
from pyPLEdestrians import ES, EW, PLEdestriansModel, PLEState
from run_corridor import hex_lattice

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
V_DES = float(np.sqrt(ES / EW))


def build_geometry(gap, length, room=14.0, half_width=8.0):
    g = gap / 2
    outline = [
        (-room, -half_width), (0, -half_width), (0, -g), (length, -g),
        (length, -half_width), (length + room, -half_width),
        (length + room, half_width), (length, half_width), (length, g),
        (0, g), (0, half_width), (-room, half_width),
    ]
    return shapely.Polygon(outline)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--gap", type=float, default=2.4)
    p.add_argument("--length", type=float, default=3.0)
    p.add_argument("--density", type=float, default=1.5)
    p.add_argument("--speed-spread", type=float, default=0.1)
    p.add_argument("--radius", type=float, default=0.3)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--sim-time", type=float, default=120.0)
    p.add_argument("--frame-interval", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tag", default="narrow_passage")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    area = build_geometry(args.gap, args.length)
    goal_x0, goal_x1 = args.length + 6.0, args.length + 13.0

    # Block of the requested density, centred on the passage axis, 2 m
    # upstream of the entrance, just deep enough for n agents.
    block_w = 9.0
    lattice = hex_lattice(block_w, 40.0, args.density, args.radius, rng)
    lattice = lattice[np.argsort(-lattice[:, 1], kind="stable")][: args.n]
    starts = np.column_stack([lattice[:, 1] - 2.0, lattice[:, 0] - block_w / 2])
    n = len(starts)
    goals = []
    while len(goals) < n:  # random sequential placement, >= 0.8 m apart
        g = rng.uniform([goal_x0, -7.2], [goal_x1, 7.2])
        if all(np.hypot(*(g - h)) >= 0.8 for h in goals):
            goals.append(g)
    goals = np.array(goals)
    v_des = V_DES * (1 + rng.uniform(-args.speed_spread, args.speed_spread, n))

    every = max(1, int(round(args.frame_interval / args.dt)))
    db_path = RESULTS / f"{args.tag}.sqlite"
    writer = SqliteTrajectoryWriter(output_file=db_path, every_nth_frame=every)
    model = PLEdestriansModel()
    sim = jps.Simulation(model=model, geometry=area, dt=args.dt, trajectory_writer=writer)

    c = np.sqrt(ES * EW)
    ids, tol = [], 0.3
    for s, g, vd in zip(starts, goals, v_des):
        wp = sim.add_waypoint_stage(tuple(g), tol)
        j = sim.add_journey(jps.JourneyDescription([wp]))
        ids.append(sim.add_agent(
            journey_id=j, stage_id=wp, position=tuple(s),
            state=PLEState(position=tuple(s), goal=tuple(g), radius=args.radius,
                           tau=args.tau, es=c * vd, ew=c / vd),
        ))

    crashed = None
    t_all = None
    wall0 = time.perf_counter()
    try:
        for i in range(int(args.sim_time / args.dt)):
            sim.iterate()
            if i % 10 == 0:
                d = [np.hypot(*(np.array(sim.agent(a).position) - g)) for a, g in zip(ids, goals)]
                if max(d) <= tol:
                    t_all = sim.elapsed_time()
                    break
    except Exception as e:  # noqa: BLE001 - recorded in the summary
        crashed = {"step": sim.iteration_count(), "time": sim.elapsed_time(), "error": repr(e)}
    wall = time.perf_counter() - wall0
    writer.close()

    drift = max(float(np.hypot(*(np.array(sim.agent(a).position) - sim.agent(a).state.position)))
                for a in ids)
    summary = {
        "n_agents": n, "gap": args.gap, "length": args.length, "density_initial": args.density,
        "speed_spread": args.speed_spread, "radius": args.radius, "tau": args.tau,
        "dt": args.dt, "seed": args.seed, "frame_interval": every * args.dt,
        "goal_region_x": [goal_x0, goal_x1], "waypoint_tolerance": tol,
        "goals": {str(a): tuple(map(float, g)) for a, g in zip(ids, goals)},
        "geometry_wkt": area.wkt,
        "t_all_arrived_check": t_all, "sim_time_run": sim.elapsed_time(),
        "wall_seconds": wall, "n_pv_empty": model.n_pv_empty,
        "max_dead_reckoning_drift": drift,
        "db_path": str(db_path), "crashed": crashed,
    }
    (RESULTS / f"{args.tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"{args.tag}: n={n}, all arrived at {t_all}, wall {wall:.0f}s, crashed={crashed}, drift={drift:.1e}")


if __name__ == "__main__":
    main()
