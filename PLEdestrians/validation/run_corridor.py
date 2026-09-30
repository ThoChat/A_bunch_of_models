# SPDX-License-Identifier: LGPL-3.0-or-later
"""Long Corridor benchmark of Guy et al. (2010), Sec. 5.1 / Figs. 6a, 7, 9:
"Ten thousand agents that fill a corridor that is 300m long. The agents all
have a random goal that is located 100m or more south of their initial
position."

The corridor is 25 m wide (the x-range of Fig. 9). The paper's 10,000
agents over 300 m x 25 m is 1.33 agents/m^2. We keep the width, the
density and the goal rule, but fill a shorter block (--block-length, south
= -y) so a run takes minutes, not hours, in this pure-Python callback. The
corridor continues far enough south that no agent reaches its goal during
the run.

Agents start on a hexagonal lattice at the requested density with a small
random jitter; random sequential placement cannot reach the densities
Fig. 7 needs (above ~1.6 agents/m^2 for 0.3 m disks).

Usage:
    python run_corridor.py --density 1.33 --block-length 40 --tag corridor_rho1.33
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

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
V_DES = float(np.sqrt(ES / EW))


def hex_lattice(width, length, density, radius, rng):
    """Hexagonal lattice with `density` points/m^2 inside
    [radius+0.05, width-radius-0.05] x [-length, 0], jittered so that no
    two points come closer than 2*radius + 0.02."""
    a = np.sqrt(2.0 / (np.sqrt(3.0) * density))  # lattice constant
    if a < 2 * radius + 0.02:
        raise ValueError(f"density {density} too high for radius {radius}")
    jit = 0.5 * (a - 2 * radius - 0.02)
    dy = a * np.sqrt(3.0) / 2
    margin = radius + 0.05
    pts = []
    for k, y in enumerate(np.arange(-dy / 2, -length, -dy)):
        x0 = margin + (a / 2 if k % 2 else 0.0)
        for x in np.arange(x0, width - margin, a):
            pts.append((x, y))
    pts = np.array(pts)
    ang = rng.uniform(0, 2 * np.pi, len(pts))
    mag = rng.uniform(0, jit, len(pts))
    pts = pts + np.column_stack([np.cos(ang), np.sin(ang)]) * mag[:, None]
    pts[:, 0] = np.clip(pts[:, 0], margin, width - margin)
    return pts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--density", type=float, default=10000 / (300 * 25))
    p.add_argument("--width", type=float, default=25.0)
    p.add_argument("--block-length", type=float, default=40.0)
    p.add_argument("--goal-min", type=float, default=100.0)
    p.add_argument("--goal-extra", type=float, default=20.0)
    p.add_argument("--speed-spread", type=float, default=0.1)
    p.add_argument("--radius", type=float, default=0.3)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--sim-time", type=float, default=40.0)
    p.add_argument("--frame-interval", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    W = args.width
    south = -(args.block_length + args.goal_min + args.goal_extra + 5.0)
    corridor = shapely.Polygon([(0, south), (W, south), (W, 2.0), (0, 2.0)])
    starts = hex_lattice(W, args.block_length, args.density, args.radius, rng)
    n = len(starts)
    v_des = V_DES * (1 + rng.uniform(-args.speed_spread, args.speed_spread, n))
    goals = np.column_stack([
        rng.uniform(0.5, W - 0.5, n),
        starts[:, 1] - rng.uniform(args.goal_min, args.goal_min + args.goal_extra, n),
    ])

    every = max(1, int(round(args.frame_interval / args.dt)))
    db_path = RESULTS / f"{args.tag}.sqlite"
    writer = SqliteTrajectoryWriter(output_file=db_path, every_nth_frame=every)
    model = PLEdestriansModel()
    sim = jps.Simulation(model=model, geometry=corridor, dt=args.dt, trajectory_writer=writer)

    c = np.sqrt(ES * EW)
    ids = []
    for s, g, vd in zip(starts, goals, v_des):
        wp = sim.add_waypoint_stage(tuple(g), 0.2)
        j = sim.add_journey(jps.JourneyDescription([wp]))
        ids.append(sim.add_agent(
            journey_id=j, stage_id=wp, position=tuple(s),
            state=PLEState(position=tuple(s), goal=tuple(g), radius=args.radius,
                           tau=args.tau, es=c * vd, ew=c / vd),
        ))

    n_steps = int(args.sim_time / args.dt)
    crashed = None
    step_times = []
    try:
        for _ in range(n_steps):
            t0 = time.perf_counter()
            sim.iterate()
            step_times.append(time.perf_counter() - t0)
    except Exception as e:  # noqa: BLE001 - recorded in the summary
        crashed = {"step": sim.iteration_count(), "time": sim.elapsed_time(), "error": repr(e)}
    writer.close()

    drift = max(float(np.hypot(*(np.array(sim.agent(a).position) - sim.agent(a).state.position)))
                for a in ids)
    summary = {
        "density_initial": args.density, "n_agents": n, "width": W,
        "block_length": args.block_length, "goal_min": args.goal_min,
        "goal_extra": args.goal_extra, "speed_spread": args.speed_spread,
        "radius": args.radius, "tau": args.tau, "dt": args.dt, "seed": args.seed,
        "sim_time": args.sim_time, "frame_interval": every * args.dt,
        "south_end": south,
        "v_des_agents": {str(a): float(v) for a, v in zip(ids, v_des)},
        "n_steps_completed": len(step_times),
        "mean_step_time_s": float(np.mean(step_times)) if step_times else None,
        "n_pv_empty": model.n_pv_empty,
        "max_dead_reckoning_drift": drift,
        "db_path": str(db_path), "crashed": crashed,
    }
    (RESULTS / f"{args.tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"{args.tag}: n={n}, {len(step_times)} steps, "
          f"{summary['mean_step_time_s']} s/step, crashed={crashed}, drift={drift:.1e}")


if __name__ == "__main__":
    main()
