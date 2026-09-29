# SPDX-License-Identifier: LGPL-3.0-or-later
"""Figure 1 of Helbing, Farkas & Vicsek (2000): room evacuation through a
1 m-wide door, for a single desired velocity v0. Identical v0 for every
agent (as in the paper's Fig. 1 caption).

Writes:
  results/fig1_v{v0}_seed{seed}.sqlite   -- trajectory
  results/fig1_v{v0}_seed{seed}.json     -- summary metrics

Usage:
  python run_fig1_room_evacuation.py --v0 1.5 --n-agents 50 --seed 1 \
      --max-time 90 --out-dir results
"""
import argparse
import json
import pathlib
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0", type=float, required=True, help="desired speed [m/s], identical for all agents")
    ap.add_argument("--n-agents", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-time", type=float, default=90.0, help="simulated seconds cap")
    ap.add_argument("--room-size", type=float, default=15.0)
    ap.add_argument("--exit-width", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.0001)
    ap.add_argument(
        "--immune-distance", type=float, default=0.0,
        help="exploratory, not in the paper: agents within this many metres of the "
             "door cannot freeze from injury (default 0.0 = paper's literal rule)",
    )
    ap.add_argument("--out-dir", type=str, default="results")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import jupedsim as jps
    from shapely import Polygon
    from sfm_injury import InjuryTrackingSFM, InjuryTrackingState

    out_dir = pathlib.Path(__file__).resolve().parent / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_immuneR{args.immune_distance:g}" if args.immune_distance > 0 else ""
    tag = f"fig1_v{args.v0:g}_seed{args.seed}{suffix}"
    db_path = out_dir / f"{tag}.sqlite"
    json_path = out_dir / f"{tag}.json"

    L = args.room_size
    ew = args.exit_width
    room = Polygon([(0, 0), (L, 0), (L, L), (0, L)])
    ey0, ey1 = (L - ew) / 2.0, (L + ew) / 2.0
    exit_capture = Polygon([(L, ey0), (L + 1.0, ey0), (L + 1.0, ey1), (L, ey1)])
    geometry = room.union(exit_capture)

    writer = jps.SqliteTrajectoryWriter(
        output_file=pathlib.Path(db_path), every_nth_frame=1000  # 10 fps output at dt=1e-4
    )
    sim = jps.Simulation(
        model=InjuryTrackingSFM(door_p0=(L, ey0), door_p1=(L, ey1), immune_distance=args.immune_distance),
        geometry=geometry,
        dt=args.dt,
        trajectory_writer=writer,
    )
    exit_id = sim.add_exit_stage(exit_capture.exterior.coords[:-1])
    journey = jps.JourneyDescription([exit_id])
    journey_id = sim.add_journey(journey)

    rng = np.random.default_rng(args.seed)
    spawn = Polygon([(0.5, 0.5), (L - 0.5, 0.5), (L - 0.5, L - 0.5), (0.5, L - 0.5)])
    positions = jps.distributions.distribute_by_number(
        polygon=spawn,
        number_of_agents=args.n_agents,
        distance_to_agents=0.55,
        distance_to_polygon=0.2,
        seed=args.seed,
    )
    radii = rng.uniform(0.25, 0.35, size=args.n_agents)  # 0.5-0.7 m shoulder width, per paper

    for pos, r in zip(positions, radii):
        sim.add_agent(
            journey_id=journey_id,
            stage_id=exit_id,
            position=pos,
            state=InjuryTrackingState(
                velocity=(0.0, 0.0), desired_speed=args.v0, radius=float(r),
                position=(float(pos[0]), float(pos[1])),
            ),
        )

    max_steps = int(args.max_time / args.dt)
    t0 = time.time()
    step = 0
    crashed_at = None
    try:
        for step in range(max_steps):
            sim.iterate()
            if sim.agent_count() == 0:
                break
    except Exception as e:
        crashed_at = {"step": step, "time": step * args.dt, "error": str(e)}
    wallclock = time.time() - t0
    writer.close()

    remaining = list(sim.agents())
    still_inside_ids = {a.id for a in remaining}
    n_injured_remaining = sum(1 for a in remaining if a.state.injured)

    summary = {
        "v0": args.v0,
        "n_agents": args.n_agents,
        "seed": args.seed,
        "max_time": args.max_time,
        "dt": args.dt,
        "immune_distance": args.immune_distance,
        "steps_run": step + 1,
        "wallclock_seconds": wallclock,
        "n_remaining_at_cutoff": len(still_inside_ids),
        "still_inside_ids": sorted(still_inside_ids),
        "n_injured_at_cutoff": n_injured_remaining,
        "fully_evacuated": len(still_inside_ids) == 0,
        "crashed": crashed_at,
        "db_path": str(db_path),
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[fig1] v0={args.v0} done: {summary}")


if __name__ == "__main__":
    main()
