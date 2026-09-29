# SPDX-License-Identifier: LGPL-3.0-or-later
"""Figure 3 of Helbing, Farkas & Vicsek (2000): pedestrians in a smoky room
must find one of two "invisible" exits, using a mixture of individualistic
searching and herding (panic parameter p).

Exits are invisible in the sense that HerdingSFM never uses JuPedSim's
routed direction-to-target at all (see sfm_herding.py): each agent walks
according to Eq. 4 (blend of its own fixed random guess and the herd's
average direction) and only "finds" an exit by physically wandering within
its capture zone. Agents are pre-assigned to a dummy waypoint stage (never
actually used for steering) purely to satisfy JuPedSim's add_agent API;
removal is driven manually every step by checking agents_in_polygon() on
each exit's ~2 m capture zone (the paper: "If one of the exits is closer
than 2 m, the room is left").

Usage:
  python run_fig3_smoky_room.py --p 0.3 --seed 1 --duration 40 --out-dir results
"""
import argparse
import json
import math
import pathlib
import time


def circle_polygon(cx: float, cy: float, r: float, n: int = 20):
    return [
        (cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n))
        for k in range(n)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, required=True, help="panic parameter (0=individualist, 1=pure herding)")
    ap.add_argument("--herd-radius", type=float, default=5.0)
    ap.add_argument("--n-agents", type=int, default=90)
    ap.add_argument("--room-size", type=float, default=15.0)
    ap.add_argument("--door-width", type=float, default=1.5)
    ap.add_argument("--capture-radius", type=float, default=2.0, help="'invisible until closer than this' distance")
    ap.add_argument(
        "--v0", type=float, default=3.0,
        help="paper's Fig. 3a snapshot uses 5.0, but at that speed, multi-directional "
             "head-on encounters between individualistically-searching agents regularly "
             "exceed this stiff model's dt=1e-4 stability margin within ~1-2s (confirmed "
             "by direct reproduction: two colliding agents reach velocities >1e4 m/s in a "
             "single step). 3.0 m/s survives the full run cleanly; see notebook methodology.",
    )
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--dt", type=float, default=0.0001)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", type=str, default="results")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import numpy as np
    import jupedsim as jps
    from shapely import Polygon
    from sfm_herding import HerdingSFM, HerdingState

    out_dir = pathlib.Path(__file__).resolve().parent / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"fig3_p{args.p:g}_seed{args.seed}"
    db_path = out_dir / f"{tag}.sqlite"
    json_path = out_dir / f"{tag}.json"

    L = args.room_size
    room = Polygon([(0, 0), (L, 0), (L, L), (0, L)])

    # Two doors on opposite walls, centered -- symmetric placement so any
    # imbalance in door usage emerges from herding dynamics, not geometry.
    door1_center = (0.0, L / 2.0)
    door2_center = (L, L / 2.0)
    door1_zone = Polygon(circle_polygon(*door1_center, args.capture_radius))
    door2_zone = Polygon(circle_polygon(*door2_center, args.capture_radius))

    writer = jps.SqliteTrajectoryWriter(
        output_file=pathlib.Path(db_path), every_nth_frame=1000
    )
    sim = jps.Simulation(
        model=HerdingSFM(),
        geometry=room,
        dt=args.dt,
        trajectory_writer=writer,
    )
    dummy_stage = sim.add_waypoint_stage(position=(L / 2.0, L / 2.0), distance=1.0)
    journey = jps.JourneyDescription([dummy_stage])
    journey_id = sim.add_journey(journey)

    rng = np.random.default_rng(args.seed)
    spawn = Polygon([(1.0, 1.0), (L - 1.0, 1.0), (L - 1.0, L - 1.0), (1.0, L - 1.0)])
    positions = jps.distributions.distribute_by_number(
        polygon=spawn,
        number_of_agents=args.n_agents,
        distance_to_agents=0.55,
        distance_to_polygon=0.2,
        seed=args.seed,
    )
    angles = rng.uniform(0, 2 * math.pi, size=args.n_agents)
    radii = rng.uniform(0.25, 0.35, size=args.n_agents)

    for pos, ang, r in zip(positions, angles, radii):
        indiv_dir = (math.cos(ang), math.sin(ang))
        sim.add_agent(
            journey_id=journey_id,
            stage_id=dummy_stage,
            position=pos,
            state=HerdingState(
                velocity=(0.0, 0.0),
                desired_speed=args.v0,
                radius=float(r),
                individual_direction=indiv_dir,
                current_direction=indiv_dir,
                panic_parameter=args.p,
                herd_radius=args.herd_radius,
            ),
        )

    max_steps = int(args.duration / args.dt)
    departures = []  # (agent_id, door(1 or 2), time)
    already_removed = set()
    t0 = time.time()
    crashed_at = None
    step = 0
    try:
        for step in range(max_steps):
            t = step * args.dt
            sim.iterate()

            for a in sim.agents_in_polygon(door1_zone):
                if a.id not in already_removed:
                    already_removed.add(a.id)
                    departures.append((a.id, 1, t))
                    sim.mark_agent_for_removal(a.id)
            for a in sim.agents_in_polygon(door2_zone):
                if a.id not in already_removed:
                    already_removed.add(a.id)
                    departures.append((a.id, 2, t))
                    sim.mark_agent_for_removal(a.id)

            if sim.agent_count() == 0:
                break
    except Exception as e:
        crashed_at = {"step": step, "time": step * args.dt, "error": str(e)}
    wallclock = time.time() - t0
    writer.close()

    departures.sort(key=lambda d: d[2])
    n1 = sum(1 for d in departures if d[1] == 1)
    n2 = sum(1 for d in departures if d[1] == 2)
    escaped_within_30s = sum(1 for d in departures if d[2] <= 30.0)
    time_for_80 = departures[79][2] if len(departures) >= 80 else None

    summary = {
        "p": args.p,
        "herd_radius": args.herd_radius,
        "n_agents": args.n_agents,
        "seed": args.seed,
        "duration": args.duration,
        "dt": args.dt,
        "door1_center": door1_center,
        "door2_center": door2_center,
        "capture_radius": args.capture_radius,
        "n_escaped_total": len(departures),
        "n_escaped_within_30s": escaped_within_30s,
        "time_for_80_to_leave": time_for_80,
        "n1_door1": n1,
        "n2_door2": n2,
        "abs_door_usage_diff": abs(n1 - n2),
        "wallclock_seconds": wallclock,
        "crashed": crashed_at,
        "departures": departures,
        "db_path": str(db_path),
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[fig3] p={args.p} done: escaped_30s={escaped_within_30s}, "
        f"n1={n1}, n2={n2}, wallclock={wallclock:.0f}s"
    )


if __name__ == "__main__":
    main()
