# SPDX-License-Identifier: LGPL-3.0-or-later
"""Scenario 2 of van Toll et al. (2021), Section 6.2 / Fig. 4(b) / Table 2:
crossing flow plus a bottleneck.

Geometry (read off Fig. 4(b), 1 m grid): a horizontal hall x in [-6, 20],
y in [-6, 6]; a vertical corridor x in [4, 10], y in [-10, 10] crossing it;
the right end is closed by a 1 m thick wall (x in [20, 21]) with a 0.8 m
door at y = 0, exactly as in Scenario 1, and a small area outside.

Agents:
- 224 "brown" agents start in the green rectangle x in [10, 20], y in
  [-6, 6] (14 x 16 grid with a little jitter) and leave through the door.
- Every 2 s, 8 "blue" agents appear at x = -5, y = -3.5 ... 3.5 with the
  same goal (1 m beyond the door, removed within 0.5 m of it).
- Every 1 s, 4 "pink" agents appear at y = 9, x = 4.75 ... 9.25 and walk
  down the corridor; they leave at its lower end (y < -9).

Measured (as in the paper): the mean (std) SPH density at t = 45 s of the
agents inside the green rectangle, and the mean number of evacuees per
second during the first 60 s.

Usage:
    python run_crossing_bottleneck.py --profile RVO->SPH --tag crossing_RVO-SPH
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

GOAL = (22.0, 0.0)
GREEN = (10.0, -6.0, 20.0, 6.0)


def build_geometry(door_width: float):
    hall = shapely.box(-6.0, -6.0, 20.0, 6.0)
    corridor = shapely.box(4.0, -10.0, 10.0, 10.0)
    door = shapely.box(20.0 - 1e-3, -door_width / 2, 21.0 + 1e-3, door_width / 2)
    outside = shapely.box(21.0, -2.0, 23.0, 2.0)
    return shapely.union_all([hall, corridor, door, outside]).simplify(0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="SPH")
    p.add_argument("--rho0-max", type=float, default=5.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--sim-time", type=float, default=60.0)
    p.add_argument("--door-width", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--every-nth-frame", type=int, default=5)
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    model = SPHCrowdModel(seed=args.seed)
    db_path = RESULTS / f"{args.tag}.sqlite"
    writer = SqliteTrajectoryWriter(output_file=db_path, every_nth_frame=args.every_nth_frame)
    sim = jps.Simulation(model=model, geometry=build_geometry(args.door_width), dt=args.dt,
                         trajectory_writer=writer)

    exit_door = sim.add_exit_stage(shapely.Point(GOAL).buffer(0.5, quad_segs=8))
    j_door = sim.add_journey(jps.JourneyDescription([exit_door]))
    exit_down = sim.add_exit_stage(shapely.box(4.0, -10.0, 10.0, -9.0))
    j_down = sim.add_journey(jps.JourneyDescription([exit_down]))

    rng = np.random.default_rng(args.seed)
    uid = [0]
    groups = {}  # jupedsim id -> "brown" / "blue" / "pink"

    def add(pos, journey, stage, group):
        st = make_state(rng, args.profile, uid=uid[0], rho0_max=args.rho0_max)
        uid[0] += 1
        aid = sim.add_agent(journey_id=journey, stage_id=stage, position=pos, state=st)
        groups[aid] = group

    xs = np.linspace(10.0, 20.0, 14, endpoint=False) + 10.0 / 28
    ys = np.linspace(-6.0, 6.0, 16, endpoint=False) + 12.0 / 32
    for x in xs:
        for y in ys:
            jit = rng.uniform(-0.1, 0.1, 2)
            add((float(x + jit[0]), float(y + jit[1])), j_door, exit_door, "brown")

    blue_y = np.arange(8) - 3.5
    pink_x = np.array([4.75, 6.25, 7.75, 9.25])
    spawn_blue = int(round(2.0 / args.dt))
    spawn_pink = int(round(1.0 / args.dt))

    exits = []  # (t, group) of agents that left through the door
    pink_out = 0
    density_45 = None
    snapshot_45 = []
    step_times = []
    crashed = None
    n_steps = int(round(args.sim_time / args.dt))
    try:
        for it in range(n_steps):
            if it % spawn_blue == 0:
                for y in blue_y:
                    add((-5.0, float(y)), j_door, exit_door, "blue")
            if it % spawn_pink == 0:
                for x in pink_x:
                    add((float(x), 9.0), j_down, exit_down, "pink")
            before = {a.id for a in sim.agents()}
            t0 = time.perf_counter()
            sim.iterate()
            step_times.append(time.perf_counter() - t0)
            t = sim.elapsed_time()
            gone = before - {a.id for a in sim.agents()}
            for aid in gone:
                if groups[aid] == "pink":
                    pink_out += 1
                else:
                    exits.append((t, groups[aid]))
            if abs(t - 45.0) < args.dt / 2:
                dens = [a.state.density for a in sim.agents()
                        if GREEN[0] <= a.position[0] <= GREEN[2] and GREEN[1] <= a.position[1] <= GREEN[3]]
                density_45 = {"mean": float(np.mean(dens)), "std": float(np.std(dens)), "n": len(dens)}
                snapshot_45 = [[a.position[0], a.position[1], a.state.velocity[0], a.state.velocity[1],
                                a.state.density, groups[a.id]] for a in sim.agents()]
    except Exception as e:  # noqa: BLE001
        crashed = {"step": len(step_times), "time": sim.elapsed_time(), "error": repr(e)}
    writer.close()

    t_end = sim.elapsed_time()
    n_first60 = sum(1 for t, _ in exits if t <= 60.0)
    final_groups = {}
    for a in sim.agents():
        final_groups.setdefault(groups[a.id], []).append(a.position)
    summary = {
        "scenario": "crossing_bottleneck",
        "profile": args.profile,
        "rho0_max": args.rho0_max,
        "dt": args.dt,
        "sim_time": args.sim_time,
        "seed": args.seed,
        "door_width": args.door_width,
        "goal": GOAL,
        "green_rect": GREEN,
        "n_spawned": uid[0],
        "groups": {str(k): v for k, v in groups.items()},
        "exits": exits,
        "n_pink_out": pink_out,
        "density_45s": density_45,
        "snapshot_45s": snapshot_45,
        "flow_rate_first_60s": n_first60 / min(60.0, t_end) if t_end > 0 else None,
        "final_positions": {g: v for g, v in final_groups.items()},
        "sim_time_reached": t_end,
        "mean_step_time_ms": float(np.mean(step_times) * 1000) if step_times else None,
        "wall_clock_s": float(np.sum(step_times)),
        "db_path": str(db_path),
        "crashed": crashed,
    }
    with open(RESULTS / f"{args.tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{args.tag}: exits(60s)={n_first60}, flow={summary['flow_rate_first_60s']}, "
          f"rho(45s)={density_45}, pink_out={pink_out}, crashed={crashed}, {np.sum(step_times):.0f}s wall")


if __name__ == "__main__":
    main()
