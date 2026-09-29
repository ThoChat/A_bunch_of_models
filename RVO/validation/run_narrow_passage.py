# SPDX-License-Identifier: LGPL-3.0-or-later
"""Narrow Passage scenario (van den Berg, Lin & Manocha 2008, Section VI):
four groups of agents start in the four corners of a square arena, each
heading to the diagonally opposite corner, with square obstacles placed so
the streams must funnel through shared narrow gaps in the middle -- rather
than just crossing an empty room diagonally, as the Circle-scenario-like
"four corners" geometry alone would allow.

Geometry: four square pillars placed at the N/E/S/W points of a small
central diamond (NOT a U-shape / dead end, which the paper notes as a known
RVO failure case). The diagonal gaps between adjacent pillars are the only
way through for a diagonally-travelling stream, and are sized to comfortably
fit only two or three agents abreast at once.

Usage:
    python run_narrow_passage.py --n-per-group 25 --tag narrow_passage
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import jupedsim as jps
from jupedsim.sqlite_serialization import SqliteTrajectoryWriter
from pyrvo import ReciprocalVelocityObstacleModel, RVOState

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


def square(cx, cy, half):
    return [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    ]


def build_geometry(arena_half: float, pillar_offset: float, pillar_half: float):
    outer = [
        (-arena_half, -arena_half),
        (arena_half, -arena_half),
        (arena_half, arena_half),
        (-arena_half, arena_half),
    ]
    pillars = [
        square(pillar_offset, 0.0, pillar_half),
        square(-pillar_offset, 0.0, pillar_half),
        square(0.0, pillar_offset, pillar_half),
        square(0.0, -pillar_offset, pillar_half),
    ]
    return outer, pillars


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-group", type=int, default=25)
    p.add_argument("--dt", type=float, default=0.025)
    p.add_argument("--sim-time", type=float, default=40.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--agent-radius", type=float, default=0.22)
    p.add_argument("--max-speed", type=float, default=1.3)
    p.add_argument("--arena-half", type=float, default=14.0)
    p.add_argument("--pillar-offset", type=float, default=5.0)
    p.add_argument("--pillar-half", type=float, default=1.8)
    p.add_argument("--tag", default="narrow_passage")
    p.add_argument("--every-nth-frame", type=int, default=2)
    args = p.parse_args()

    outer, pillars = build_geometry(args.arena_half, args.pillar_offset, args.pillar_half)
    # `Simulation(geometry=..., excluded_areas=...)` silently ignores
    # `excluded_areas` in this jupedsim build (the constructor accepts
    # **kwargs but never forwards them to geometry construction) -- holes
    # have to be baked into a shapely Polygon and passed as `geometry`
    # directly instead.
    walkable = shapely.Polygon(outer, holes=pillars)
    model = ReciprocalVelocityObstacleModel()
    db_path = RESULTS / f"{args.tag}.sqlite"
    writer = SqliteTrajectoryWriter(output_file=db_path, every_nth_frame=args.every_nth_frame)
    sim = jps.Simulation(
        model=model, geometry=walkable, dt=args.dt, trajectory_writer=writer,
    )

    rng = np.random.default_rng(args.seed)
    corner_dist = args.arena_half - 1.5
    corners = {
        "NE": (corner_dist, corner_dist),
        "NW": (-corner_dist, corner_dist),
        "SE": (corner_dist, -corner_dist),
        "SW": (-corner_dist, -corner_dist),
    }
    opposite = {"NE": "SW", "NW": "SE", "SE": "NW", "SW": "NE"}

    agents = []
    targets = {}
    cluster_spread = 1.8
    for name, (cx, cy) in corners.items():
        target = corners[opposite[name]]
        wp = sim.add_waypoint_stage(target, 0.3)
        journey = sim.add_journey(jps.JourneyDescription([wp]))
        placed = 0
        attempts = 0
        while placed < args.n_per_group and attempts < args.n_per_group * 50:
            attempts += 1
            offset = rng.uniform(-cluster_spread, cluster_spread, size=2)
            pos = (cx + offset[0], cy + offset[1])
            try:
                agent_id = sim.add_agent(
                    journey_id=journey,
                    stage_id=wp,
                    position=pos,
                    state=RVOState(
                        velocity=(0.0, 0.0),
                        radius=args.agent_radius,
                        max_speed=args.max_speed,
                    ),
                )
            except Exception:
                continue
            agents.append(agent_id)
            targets[agent_id] = target
            placed += 1

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
        "n_per_group": args.n_per_group,
        "n_total_spawned": len(agents),
        "dt": args.dt,
        "seed": args.seed,
        "agent_radius": args.agent_radius,
        "max_speed": args.max_speed,
        "arena_half": args.arena_half,
        "pillar_offset": args.pillar_offset,
        "pillar_half": args.pillar_half,
        "n_steps_completed": len(step_times),
        "mean_step_time_ms": float(np.mean(step_times) * 1000) if step_times else None,
        "db_path": str(db_path),
        "targets": {str(k): v for k, v in targets.items()},
        "agent_ids": agents,
        "waypoint_tolerance": 0.3,
        "crashed": crashed,
    }
    json_path = RESULTS / f"{args.tag}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {json_path}: spawned {len(agents)} agents, {len(step_times)} steps completed")


if __name__ == "__main__":
    main()
