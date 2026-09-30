# SPDX-License-Identifier: LGPL-3.0-or-later
"""Office evacuation (van den Berg, Guy, Lin & Manocha 2011, Section 6,
Fig. 9 and the "Office" curves of Fig. 10): 1,000 agents evacuate an office
floor, each following a globally planned path to an exit, with ORCA doing
the local collision avoidance against other agents and against walls.

The paper does not publish its floor plan (it reuses the office of Guy et
al. 2009 [10]), so this is a stand-in with the same ingredients visible in
Fig. 9: rows of closed offices with single doors, an open-plan area with
cubicle partitions, and a few exits in the outer wall.

  * 60 m x 40 m floor, 10 offices (6 m x 8 m, 1.2 m door) along the top
    wall, 9 along the bottom wall plus a 6 m wide corridor to the bottom
    exit, open-plan area in between with 5 m cubicle partitions (plus a
    2 m stub each) in three rows.
  * three 2.4 m exits (left, right, bottom), each opening into a short
    vestibule whose far end is the exit stage (agents are removed there:
    the paper's agents leave the building).
  * the global path is JuPedSim's navigation-mesh shortest path; each agent
    is assigned the exit with the shortest such path.

`--scale s` multiplies every floor-plan coordinate by s (doors and wall
thickness unchanged) so that the timing sweep can put 5,000 agents in at
the same density as 1,000.

Usage:
    python run_office.py --n 1000 --tag office_n1000
    python run_office.py --n 3000 --scale 1.732 --benchmark-steps 20 --tag timing_office_n3000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import shapely
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jupedsim as jps
from jupedsim.sqlite_serialization import SqliteTrajectoryWriter
from pyORCA import ORCAModel, ORCAState

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

WALL = 0.2  # interior wall thickness [m]
DOOR = 1.2  # office door width [m]
EXIT = 2.4  # exit width [m]
VESTIBULE = 2.0  # depth of the space behind each exit [m]


def wall_with_doors(p0, p1, door_centres, door_width):
    """Axis-aligned wall from p0 to p1 with gaps; returns LineStrings."""
    (x0, y0), (x1, y1) = p0, p1
    horizontal = abs(y1 - y0) < 1e-9
    a0, a1 = (x0, x1) if horizontal else (y0, y1)
    cuts = sorted(door_centres)
    pieces, start = [], a0
    for c in cuts:
        pieces.append((start, c - door_width / 2))
        start = c + door_width / 2
    pieces.append((start, a1))
    lines = []
    for s, e in pieces:
        if e - s <= 1e-6:
            continue
        if horizontal:
            lines.append(shapely.LineString([(s, y0), (e, y0)]))
        else:
            lines.append(shapely.LineString([(x0, s), (x0, e)]))
    return lines


def build_office(scale=1.0):
    W, H = 60.0 * scale, 40.0 * scale
    room_w, room_d = 6.0 * scale, 8.0 * scale
    outer = shapely.box(0, 0, W, H)

    walls = []
    # Top row of offices: y in [H - room_d, H], one door each at the bottom.
    n_rooms = int(round(W / room_w))
    centres = [room_w * (k + 0.5) for k in range(n_rooms)]
    walls += wall_with_doors((0, H - room_d), (W, H - room_d), centres, DOOR)
    for k in range(1, n_rooms):
        walls.append(shapely.LineString([(room_w * k, H - room_d), (room_w * k, H)]))
    # Bottom row: same, but the middle bay is a corridor to the bottom exit.
    mid = n_rooms // 2
    bottom_centres = [c for k, c in enumerate(centres) if k != mid]
    corridor = (room_w * mid, room_w * (mid + 1))
    walls += wall_with_doors(
        (0, room_d), (W, room_d),
        bottom_centres + [0.5 * (corridor[0] + corridor[1])],
        # the corridor bay is fully open to the hall
        DOOR,
    )
    # widen the corridor opening to its full width
    walls = [
        w.difference(shapely.box(corridor[0] + WALL, room_d - 0.5, corridor[1] - WALL, room_d + 0.5))
        for w in walls
    ]
    for k in range(1, n_rooms):
        walls.append(shapely.LineString([(room_w * k, 0), (room_w * k, room_d)]))

    # Open plan: cubicle partitions (5 m) with a 2 m stub in the middle.
    part_len, stub = 5.0 * scale, 2.0 * scale
    hall_lo, hall_hi = room_d, H - room_d
    rows_y = np.linspace(hall_lo, hall_hi, 5)[1:-1]
    x_starts = [8, 15, 22, 33, 40, 47]
    for y in rows_y:
        for xs in x_starts:
            x = xs * scale
            walls.append(shapely.LineString([(x, y), (x + part_len, y)]))
            xm = x + part_len / 2
            walls.append(shapely.LineString([(xm, y - stub / 2), (xm, y + stub / 2)]))

    wall_poly = unary_union([w.buffer(WALL / 2, cap_style="flat") for w in walls if not w.is_empty])

    yc = H / 2
    xc = 0.5 * (corridor[0] + corridor[1])
    vestibules = {
        "left": shapely.box(-VESTIBULE, yc - EXIT / 2, 0, yc + EXIT / 2),
        "right": shapely.box(W, yc - EXIT / 2, W + VESTIBULE, yc + EXIT / 2),
        "bottom": shapely.box(xc - EXIT / 2, -VESTIBULE, xc + EXIT / 2, 0),
    }
    exits = {
        "left": shapely.box(-VESTIBULE, yc - EXIT / 2, -VESTIBULE + 0.5, yc + EXIT / 2),
        "right": shapely.box(W + VESTIBULE - 0.5, yc - EXIT / 2, W + VESTIBULE, yc + EXIT / 2),
        "bottom": shapely.box(xc - EXIT / 2, -VESTIBULE, xc + EXIT / 2, -VESTIBULE + 0.5),
    }
    walkable = unary_union([outer, *vestibules.values()]).difference(wall_poly)
    walkable = shapely.set_precision(walkable, 1e-6)
    if walkable.geom_type != "Polygon":
        raise RuntimeError(f"office geometry is a {walkable.geom_type}, expected one Polygon")
    spawn_area = outer.difference(wall_poly)
    return walkable, spawn_area, exits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.025)
    p.add_argument("--sim-time", type=float, default=150.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--agent-radius", type=float, default=0.25)
    p.add_argument("--speed", type=float, default=1.5)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--tau-obst", type=float, default=0.5)
    p.add_argument("--max-neighbors", type=int, default=None)
    p.add_argument("--pref-noise", type=float, default=1e-4)
    p.add_argument("--tag", default="office_n1000")
    p.add_argument("--every-nth-frame", type=int, default=8)
    p.add_argument("--clearance-every", type=int, default=4)
    p.add_argument("--benchmark-steps", type=int, default=0)
    args = p.parse_args()

    walkable, spawn_area, exits = build_office(args.scale)
    benchmark = args.benchmark_steps > 0
    db_path = RESULTS / f"{args.tag}.sqlite"
    writer = None if benchmark else SqliteTrajectoryWriter(
        output_file=db_path, every_nth_frame=args.every_nth_frame
    )
    model = ORCAModel(pref_noise=args.pref_noise, seed=args.seed)
    sim = jps.Simulation(model=model, geometry=walkable, dt=args.dt, trajectory_writer=writer)

    exit_ids, journeys = {}, {}
    for name, poly in exits.items():
        exit_ids[name] = sim.add_exit_stage(poly)
        journeys[name] = sim.add_journey(jps.JourneyDescription([exit_ids[name]]))

    positions = jps.distribute_by_number(
        polygon=spawn_area,
        number_of_agents=args.n,
        distance_to_agents=2 * args.agent_radius + 0.1,
        distance_to_polygon=args.agent_radius + 0.1,
        seed=args.seed,
        max_iterations=20000,
    )
    router = jps.RoutingEngine(walkable)
    exit_centres = {k: tuple(v.centroid.coords[0]) for k, v in exits.items()}

    def path_length(a, b):
        wps = np.array(router.compute_waypoints(a, b))
        return float(np.linalg.norm(np.diff(wps, axis=0), axis=1).sum())

    agents, assigned, shortest = [], {}, {}
    for pos in positions:
        lengths = {k: path_length(pos, c) for k, c in exit_centres.items()}
        best = min(lengths, key=lengths.get)
        agent_id = sim.add_agent(
            journey_id=journeys[best],
            stage_id=exit_ids[best],
            position=pos,
            state=ORCAState(
                radius=args.agent_radius,
                pref_speed=args.speed,
                max_speed=args.speed,
                time_horizon=args.tau,
                time_horizon_obst=args.tau_obst,
                max_neighbors=args.max_neighbors,
                stop_at_goal=False,  # route corners reverse the orientation too
            ),
        )
        agents.append(agent_id)
        assigned[agent_id] = best
        shortest[agent_id] = lengths[best]

    n_steps = args.benchmark_steps if benchmark else int(round(args.sim_time / args.dt))
    step_times, callback_times, remaining, frac_3d = [], [], [], []
    min_clearance = np.inf
    crashed = None
    try:
        for i in range(n_steps):
            cb0, lp0, n_before = model.callback_seconds, model.n_3d_lp, sim.agent_count()
            t0 = time.perf_counter()
            sim.iterate()
            step_times.append(time.perf_counter() - t0)
            callback_times.append(model.callback_seconds - cb0)
            frac_3d.append((model.n_3d_lp - lp0) / max(n_before, 1))
            remaining.append(sim.agent_count())
            if i % args.clearance_every == 0 and sim.agent_count() > 1:
                pos = np.array([a.position for a in sim.agents()])
                order = np.argsort(pos[:, 0])
                pos = pos[order]
                for s in range(1, len(pos)):
                    dx = pos[s:, 0] - pos[:-s, 0]
                    if dx.min() > 1.0:
                        break
                    min_clearance = min(
                        min_clearance,
                        np.hypot(dx, pos[s:, 1] - pos[:-s, 1]).min() - 2 * args.agent_radius,
                    )
            if sim.agent_count() == 0:
                break
    except Exception as e:  # noqa: BLE001 - recorded, not swallowed silently
        crashed = {"step": len(step_times), "time": sim.elapsed_time(), "error": repr(e)}

    if writer is not None:
        writer.close()

    summary = {
        "scenario": "office",
        "n": args.n,
        "n_spawned": len(agents),
        "scale": args.scale,
        "dt": args.dt,
        "seed": args.seed,
        "agent_radius": args.agent_radius,
        "speed": args.speed,
        "tau": args.tau,
        "tau_obst": args.tau_obst,
        "max_neighbors": args.max_neighbors,
        "pref_noise": args.pref_noise,
        "benchmark_only": benchmark,
        "n_steps_completed": len(step_times),
        "sim_time_completed": len(step_times) * args.dt,
        "mean_step_time_ms": float(np.mean(step_times) * 1000) if step_times else None,
        "mean_callback_time_ms": float(np.mean(callback_times) * 1000) if callback_times else None,
        "step_times_ms": [round(x * 1000, 3) for x in step_times],
        "fraction_3d_lp_per_step": [round(x, 4) for x in frac_3d],
        "remaining_per_step": remaining,
        "min_clearance_running": float(min_clearance),
        "exit_assignment": {str(k): v for k, v in assigned.items()},
        "shortest_path_length": {str(k): v for k, v in shortest.items()},
        "exits": {k: list(v.exterior.coords) for k, v in exits.items()},
        "db_path": None if benchmark else str(db_path),
        "agent_ids": agents,
        "crashed": crashed,
    }
    json_path = RESULTS / f"{args.tag}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=1)
    print(
        f"Wrote {json_path}: {len(step_times)} steps ({len(step_times) * args.dt:.1f} s), "
        f"{summary['mean_step_time_ms']:.1f} ms/step, remaining {remaining[-1] if remaining else None}/"
        f"{len(agents)}, min clearance {min_clearance:+.4f} m, crashed={crashed}"
    )


if __name__ == "__main__":
    main()
