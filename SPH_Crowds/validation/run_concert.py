# SPDX-License-Identifier: LGPL-3.0-or-later
"""Scenario 3 of van Toll et al. (2021), Section 6.3 / Fig. 4(c) / Figs. 7-10:
a dense concert crowd in which a push triggers a shockwave.

Geometry (read off Fig. 4(c), 1 m fine grid, 10 m coarse grid): venue
x in [0, 69], y in [-34, 34]; the stage is the region right of x = 69 plus
a circular bulge into the venue (circle centre (102.5, 0), radius 36.4:
chord 28.5 m, 2.9 m deep). The push region is the red 1 x 20 m rectangle
x in [43.8, 45.0], y in [-10, 10].

Paper: 10,000 agents are inserted in rows of 100 per second for 100 s at
the left of the venue, walk towards an unreachable goal on the stage with
K_goal = 0.1 and converge; at t = 150 s every agent inside the rectangle
pushes forward for 0.5 s (K_goal = 1, ignoring contact and SPH forces).

Ours (see notebook): the same 10,000 agents are placed directly on a
hexagonal lattice filling the region the paper's crowd occupies at
t = 150 s (Fig. 7: the crowd's back edge is an arc of radius ~40.4 m
around the goal, which at 10,000 agents is 5.0 P/m^2), then settle for
`--settle` seconds before the push. This replaces 150 s of insertion and
convergence, which in pure Python would cost ~7x more compute per run.

The goal must lie in the walkable area in JuPedSim, so the waypoint sits
at the tip of the stage bulge, (65.9, 0), instead of on the stage (73.8, 0).

Usage:
    python run_concert.py --tag concert_baseline
    python run_concert.py --k-gas 50 --tag concert_k50
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jupedsim as jps
from jupedsim.sqlite_serialization import SqliteTrajectoryWriter
from pySPH_Crowds import SPHCrowdModel, make_state

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

STAGE_C = (102.5, 0.0)
STAGE_R = 36.4
GOAL_PAPER = (73.8, 0.0)
WAYPOINT = (65.9, 0.0)
PUSH_RECT = (43.8, -10.0, 45.0, 10.0)
CROWD_R = 40.4


def build_geometry(half_height: float):
    venue = shapely.box(0.0, -half_height, 69.0, half_height)
    stage = shapely.Point(STAGE_C).buffer(STAGE_R, quad_segs=32)
    return venue.difference(stage)


def lattice_positions(walkable, n_agents, spacing_density, rng):
    """Hexagonal lattice at the given density inside the crowd region,
    keeping the n_agents points closest to the goal."""
    a = np.sqrt(2.0 / (np.sqrt(3.0) * spacing_density))
    region = shapely.Point(GOAL_PAPER).buffer(CROWD_R + 12.0, quad_segs=64).intersection(
        walkable.buffer(-0.3))
    minx, miny, maxx, maxy = region.bounds
    xs, ys = [], []
    j = 0
    y = miny
    while y <= maxy:
        x0 = minx + (a / 2 if j % 2 else 0.0)
        row = np.arange(x0, maxx, a)
        xs.append(row); ys.append(np.full(row.size, y))
        y += a * np.sqrt(3.0) / 2
        j += 1
    pts = np.stack([np.concatenate(xs), np.concatenate(ys)], axis=1)
    inside = shapely.contains_xy(region, pts[:, 0], pts[:, 1])
    pts = pts[inside]
    d = np.hypot(pts[:, 0] - GOAL_PAPER[0], pts[:, 1] - GOAL_PAPER[1])
    pts = pts[np.argsort(d)[:n_agents]]
    return pts + rng.uniform(-0.02, 0.02, pts.shape)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-agents", type=int, default=10000)
    p.add_argument("--half-height", type=float, default=34.0)
    p.add_argument("--init-density", type=float, default=5.0)
    p.add_argument("--k-gas", type=float, default=200.0)
    p.add_argument("--mu", type=float, default=5.0)
    p.add_argument("--rho0-max", type=float, default=5.0)
    p.add_argument("--rho0-min", type=float, default=0.0)
    p.add_argument("--k-ag", type=float, default=50.0)
    p.add_argument("--k-obs", type=float, default=200.0)
    p.add_argument("--k-goal", type=float, default=0.1)
    p.add_argument("--settle", type=float, default=30.0)
    p.add_argument("--push-duration", type=float, default=0.5)
    p.add_argument("--after", type=float, default=6.5)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--every-nth-frame", type=int, default=5)
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    walkable = build_geometry(args.half_height)
    model = SPHCrowdModel(seed=args.seed)
    db_path = RESULTS / f"{args.tag}.sqlite"
    writer = SqliteTrajectoryWriter(output_file=db_path, every_nth_frame=args.every_nth_frame)
    sim = jps.Simulation(model=model, geometry=walkable, dt=args.dt, trajectory_writer=writer)
    wp = sim.add_waypoint_stage(WAYPOINT, 0.0)
    journey = sim.add_journey(jps.JourneyDescription([wp]))

    rng = np.random.default_rng(args.seed)
    pts = lattice_positions(walkable, args.n_agents, args.init_density, rng)
    uid_of = {}
    for k, (x, y) in enumerate(pts):
        st = make_state(rng, "SPH", uid=k, k_gas=args.k_gas, mu=args.mu, rho0_max=args.rho0_max,
                        rho0_min=args.rho0_min, k_ag_sph=args.k_ag, k_obs_sph=args.k_obs,
                        k_goal=args.k_goal)
        aid = sim.add_agent(journey_id=journey, stage_id=wp, position=(float(x), float(y)), state=st)
        uid_of[aid] = k

    t_push = args.settle
    n_steps = int(round((args.settle + args.after) / args.dt))
    push_start = int(round(t_push / args.dt))
    push_end = push_start + int(round(args.push_duration / args.dt))
    # Snapshots (x, y, vx, vy, rho, uid) at the paper's Fig. 7 times relative to
    # the push (150.3, 151, 152, 153, 154, 156 s = push + 0.3 ... 6), plus
    # 5 s and 0 s before it.
    rel = [-5.0, 0.0, 0.3, 1.0, 2.0, 3.0, 4.0, 6.0]
    snap_steps = {int(round((t_push + r) / args.dt)): r for r in rel if 0 <= t_push + r <= args.settle + args.after}
    snaps = {}
    density_series = []
    pushed = []
    step_times = []
    crashed = None
    try:
        for it in range(n_steps):
            if it == push_start:
                pushed = [uid_of[a.id] for a in sim.agents()
                          if PUSH_RECT[0] <= a.position[0] <= PUSH_RECT[2]
                          and PUSH_RECT[1] <= a.position[1] <= PUSH_RECT[3]]
                model.push_uids = frozenset(pushed)
            if it == push_end:
                model.push_uids = frozenset()
            if it % int(round(1.0 / args.dt)) == 0:
                dens = np.array([a.state.density for a in sim.agents()])
                density_series.append((sim.elapsed_time(), float(dens.mean()), float(dens.std()),
                                       float(np.percentile(dens, 90))))
            if it in snap_steps:
                snaps[f"t{snap_steps[it]:.2f}"] = np.array(
                    [[a.position[0], a.position[1], a.state.velocity[0], a.state.velocity[1],
                      a.state.density, a.state.uid] for a in sim.agents()])
            t0 = time.perf_counter()
            sim.iterate()
            step_times.append(time.perf_counter() - t0)
            if it % 250 == 0:
                print(f"t={sim.elapsed_time():.1f}s  step {np.mean(step_times[-250:]) * 1000:.0f} ms", flush=True)
    except Exception as e:  # noqa: BLE001
        crashed = {"step": len(step_times), "time": sim.elapsed_time(), "error": repr(e)}
    writer.close()
    if snaps:
        np.savez_compressed(RESULTS / f"{args.tag}_snapshots.npz", **snaps)

    summary = {
        "scenario": "concert",
        "n_agents": len(pts),
        "half_height": args.half_height,
        "init_density": args.init_density,
        "k_gas": args.k_gas,
        "mu": args.mu,
        "rho0_max": args.rho0_max,
        "rho0_min": args.rho0_min,
        "k_ag": args.k_ag,
        "k_obs": args.k_obs,
        "k_goal": args.k_goal,
        "t_push": t_push,
        "push_duration": args.push_duration,
        "push_rect": PUSH_RECT,
        "n_pushed": len(pushed),
        "pushed_uids": pushed,
        "uid_of": {str(k): v for k, v in uid_of.items()},
        "dt": args.dt,
        "seed": args.seed,
        "density_series": density_series,
        "sim_time_reached": sim.elapsed_time(),
        "mean_step_time_ms": float(np.mean(step_times) * 1000) if step_times else None,
        "wave_step_time_ms": float(np.mean(step_times[push_start:]) * 1000) if len(step_times) > push_start else None,
        "wall_clock_s": float(np.sum(step_times)),
        "db_path": str(db_path),
        "crashed": crashed,
    }
    with open(RESULTS / f"{args.tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{args.tag}: n={len(pts)}, pushed={len(pushed)}, crashed={crashed}, {np.sum(step_times):.0f}s wall")


if __name__ == "__main__":
    main()
