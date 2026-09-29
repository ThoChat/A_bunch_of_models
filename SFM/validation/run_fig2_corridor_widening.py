# SPDX-License-Identifier: LGPL-3.0-or-later
"""Figure 2 of Helbing, Farkas & Vicsek (2000): a corridor with a widening
in the middle, for a single widening angle phi. Pedestrians enter with a
continuous inflow J on the left and flee towards the right; efficiency
E = <v . e_corridor_axis> / v0 is measured in the second half of the
corridor, in the run's steady-state window.

Usage:
  python run_fig2_corridor_widening.py --phi 30 --duration 60 --out-dir results
"""
import argparse
import json
import pathlib
import time

import numpy as np


def build_corridor_geometry(length: float, width: float, phi_deg: float, bulge_len: float):
    """3 m-wide, `length` m-long corridor with a diamond-shaped widening of
    half-angle phi in its middle third, matching Fig. 2's shape (triangular
    pieces forming a widened hexagonal area), phi=0 -> a plain straight
    corridor.
    """
    from shapely import Polygon

    hw = width / 2.0
    x_mid = length / 2.0
    x0 = x_mid - bulge_len / 2.0
    x1 = x_mid + bulge_len / 2.0
    extra = bulge_len / 2.0 * np.tan(np.radians(phi_deg))
    pts = [
        (0.0, -hw),
        (x0, -hw),
        (x_mid, -hw - extra),
        (x1, -hw),
        (length, -hw),
        (length, hw),
        (x1, hw),
        (x_mid, hw + extra),
        (x0, hw),
        (0.0, hw),
    ]
    return Polygon(pts), x0, x1, x_mid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi", type=float, required=True, help="widening half-angle [deg], paper's Fig. 2 x-axis")
    ap.add_argument("--v0", type=float, default=2.0)
    ap.add_argument("--corridor-length", type=float, default=15.0)
    ap.add_argument("--corridor-width", type=float, default=3.0)
    ap.add_argument("--bulge-length", type=float, default=6.0)
    ap.add_argument("--inflow-rate", type=float, default=16.5, help="agents/s entering on the left (J*width)")
    ap.add_argument("--duration", type=float, default=60.0, help="simulated seconds")
    ap.add_argument("--warmup", type=float, default=15.0, help="seconds excluded from steady-state stats")
    ap.add_argument("--dt", type=float, default=0.0001)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", type=str, default="results")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    import jupedsim as jps
    from shapely import Polygon
    from pysocial_force import PythonSocialForceModel, PythonSocialForceModelState

    out_dir = pathlib.Path(__file__).resolve().parent / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"fig2_phi{args.phi:g}"
    db_path = out_dir / f"{tag}.sqlite"
    json_path = out_dir / f"{tag}.json"

    corridor, x0, x1, x_mid = build_corridor_geometry(
        args.corridor_length, args.corridor_width, args.phi, args.bulge_length
    )
    hw = args.corridor_width / 2.0
    exit_capture = Polygon(
        [
            (args.corridor_length, -hw),
            (args.corridor_length + 1.0, -hw),
            (args.corridor_length + 1.0, hw),
            (args.corridor_length, hw),
        ]
    )
    geometry = corridor.union(exit_capture)

    writer = jps.SqliteTrajectoryWriter(
        output_file=pathlib.Path(db_path), every_nth_frame=1000
    )
    sim = jps.Simulation(
        model=PythonSocialForceModel(),
        geometry=geometry,
        dt=args.dt,
        trajectory_writer=writer,
    )
    exit_id = sim.add_exit_stage(exit_capture.exterior.coords[:-1])
    journey = jps.JourneyDescription([exit_id])
    journey_id = sim.add_journey(journey)

    rng = np.random.default_rng(args.seed)
    spawn_interval = 1.0 / args.inflow_rate
    max_steps = int(args.duration / args.dt)
    next_spawn_t = 0.0
    max_radius = 0.35
    # Margin from any wall (the corridor's side walls, and -- easy to miss --
    # its entrance wall at x=0 too) must exceed the largest possible agent
    # radius, or an agent spawns already overlapping the wall: an immediate
    # body_force*(radius-dist) contact-force spike of ~1e4 N that blows up
    # its velocity in a single dt=1e-4 step and trips the tunneling guard a
    # few hundred steps later once it has been flung across the corridor.
    spawn_x = max_radius + 0.3  # clearance from the x=0 entrance wall
    y_lo, y_hi = -hw + max_radius + 0.15, hw - max_radius - 0.15
    # Same guard, but against every agent currently near the entrance, not
    # just the previous spawn: a naive "check only the last spawn" version
    # of this still let a new agent land next to an *existing* agent that
    # had since drifted there under social-force pushes from others, which
    # produced the same wall-tunneling crash, just delayed rather than
    # immediate (confirmed by direct step-by-step reproduction).
    spawn_check_zone = Polygon(
        [
            (spawn_x - max_radius - 0.1, y_lo - max_radius - 0.1),
            (spawn_x + 0.15 + max_radius + 0.1, y_lo - max_radius - 0.1),
            (spawn_x + 0.15 + max_radius + 0.1, y_hi + max_radius + 0.1),
            (spawn_x - max_radius - 0.1, y_hi + max_radius + 0.1),
        ]
    )

    t0 = time.time()
    crashed_at = None
    n_spawned = 0
    try:
        for step in range(max_steps):
            t = step * args.dt
            if t >= next_spawn_t:
                next_spawn_t += spawn_interval
                if not sim.agents_in_polygon(spawn_check_zone):
                    y = rng.uniform(y_lo, y_hi)
                    x_jitter = rng.uniform(0.0, 0.15)
                    sim.add_agent(
                        journey_id=journey_id,
                        stage_id=exit_id,
                        position=(spawn_x + x_jitter, float(y)),
                        state=PythonSocialForceModelState(
                            velocity=(args.v0, 0.0), desired_speed=args.v0, radius=float(rng.uniform(0.25, max_radius))
                        ),
                    )
                    n_spawned += 1
                # else: entrance currently occupied -- skip this tick, retry
                # at the next scheduled spawn time (self-throttling inflow).
            sim.iterate()
    except Exception as e:
        crashed_at = {"step": step, "time": step * args.dt, "error": str(e)}
    wallclock = time.time() - t0
    writer.close()

    still_inside_ids = {a.id for a in sim.agents()}
    summary = {
        "phi": args.phi,
        "v0": args.v0,
        "duration": args.duration,
        "warmup": args.warmup,
        "dt": args.dt,
        "corridor_length": args.corridor_length,
        "corridor_width": args.corridor_width,
        "x0": x0,
        "x1": x1,
        "x_mid": x_mid,
        "n_spawned": n_spawned,
        "n_remaining_at_end": len(still_inside_ids),
        "still_inside_ids": sorted(still_inside_ids),
        "wallclock_seconds": wallclock,
        "crashed": crashed_at,
        "db_path": str(db_path),
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[fig2] phi={args.phi} done: n_remaining={len(still_inside_ids)}, wallclock={wallclock:.0f}s")


if __name__ == "__main__":
    main()
