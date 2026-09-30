# SPDX-License-Identifier: LGPL-3.0-or-later
"""Circle scenarios (van den Berg, Guy, Lin & Manocha 2011, Section 6):
agents start on a circle and each walks to its own antipodal point.

Covers three of the paper's experiments:
  * Fig. 7(a): two robots exchanging positions (N=2 is a head-on swap),
  * Fig. 7(b): five robots crossing to antipodal points,
  * Fig. 8 / Fig. 10(b) "Circle": 1,000 agents (and the timing sweep up to
    5,000) on a large circle, which congests in the centre.

For large N the agents are placed on `--rows` concentric rings (the paper's
Fig. 8 shows a ring several agents thick, not a single file), each row
rotated by half a spacing against the previous one.

Usage:
    python run_circle.py --n 5 --circle-radius 1.1 --tag circle_n5
    python run_circle.py --n 1000 --rows 4 --tag circle_n1000
    python run_circle.py --n 2000 --rows 4 --benchmark-steps 20 --tag timing_circle_n2000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jupedsim as jps
from jupedsim.sqlite_serialization import SqliteTrajectoryWriter
from pyORCA import ORCAModel, ORCAState

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


def circle_layout(n, rows, spacing, circle_radius=None):
    """Start positions on `rows` concentric rings; returns (positions, R_inner)."""
    per_row = int(np.ceil(n / rows))
    if circle_radius is None:
        circle_radius = per_row * spacing / (2 * np.pi)
    positions = []
    for k in range(rows):
        m = min(per_row, n - len(positions))
        r = circle_radius + k * spacing
        phase = 0.5 * (2 * np.pi / per_row) * (k % 2)
        for a in np.linspace(0, 2 * np.pi, m, endpoint=False) + phase:
            positions.append((r * np.cos(a), r * np.sin(a)))
    return positions, circle_radius


def build_simulation(args, writer):
    positions, circle_radius = circle_layout(
        args.n, args.rows, args.spacing, args.circle_radius
    )
    r_max = max(np.hypot(*p) for p in positions)
    bound = r_max + 3.0
    geometry = [(-bound, -bound), (bound, -bound), (bound, bound), (-bound, bound)]

    model = ORCAModel(pref_noise=args.pref_noise, seed=args.seed)
    sim = jps.Simulation(model=model, geometry=geometry, dt=args.dt, trajectory_writer=writer)

    agents, targets = [], {}
    for pos in positions:
        target = (-pos[0], -pos[1])
        wp = sim.add_waypoint_stage(target, 0.1)
        journey = sim.add_journey(jps.JourneyDescription([wp]))
        heading = np.array(target) - np.array(pos)
        v0 = args.initial_speed * heading / np.linalg.norm(heading)
        agent_id = sim.add_agent(
            journey_id=journey,
            stage_id=wp,
            position=pos,
            state=ORCAState(
                velocity=(float(v0[0]), float(v0[1])),
                radius=args.agent_radius,
                pref_speed=args.speed,
                max_speed=args.speed,
                time_horizon=args.tau,
                time_horizon_obst=args.tau_obst,
                max_neighbors=args.max_neighbors,
            ),
        )
        agents.append(agent_id)
        targets[agent_id] = target
    return sim, model, agents, targets, circle_radius


def running_min_clearance(sim, agents, radius):
    pos = np.array([sim.agent(a).position for a in agents])
    order = np.argsort(pos[:, 0])
    pos = pos[order]
    best = np.inf
    for s in range(1, len(pos)):
        dx = pos[s:, 0] - pos[:-s, 0]
        if dx.min() > 2 * radius + 0.5:
            break
        best = min(best, np.hypot(dx, pos[s:, 1] - pos[:-s, 1]).min())
    return best - 2 * radius


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--rows", type=int, default=1)
    p.add_argument("--spacing", type=float, default=0.8, help="along-ring and row spacing [m]")
    p.add_argument("--circle-radius", type=float, default=None, help="override inner ring radius [m]")
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--sim-time", type=float, default=30.0)
    p.add_argument("--agent-radius", type=float, default=0.25)
    p.add_argument("--speed", type=float, default=1.5)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument(
        "--initial-speed", type=float, default=0.0,
        help="start already walking toward the goal at this speed [m/s]",
    )
    p.add_argument("--tau-obst", type=float, default=0.5)
    p.add_argument("--max-neighbors", type=int, default=None)
    p.add_argument(
        "--pref-noise", type=float, default=1e-4,
        help="RVO2-style symmetry-breaking perturbation of v_pref [m/s]; 0 = off",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tag", required=True)
    p.add_argument("--every-nth-frame", type=int, default=1)
    p.add_argument("--clearance-every", type=int, default=1, help="check running clearance every k steps")
    p.add_argument(
        "--benchmark-steps", type=int, default=0,
        help="timing only: no trajectory, stop after this many steps",
    )
    p.add_argument("--stop-when-arrived", action="store_true")
    args = p.parse_args()

    benchmark = args.benchmark_steps > 0
    db_path = RESULTS / f"{args.tag}.sqlite"
    writer = None if benchmark else SqliteTrajectoryWriter(
        output_file=db_path, every_nth_frame=args.every_nth_frame
    )
    sim, model, agents, targets, circle_radius = build_simulation(args, writer)

    n_steps = args.benchmark_steps if benchmark else int(round(args.sim_time / args.dt))
    step_times, callback_times, frac_3d = [], [], []
    min_clearance = np.inf
    crashed = None
    try:
        for i in range(n_steps):
            cb0 = model.callback_seconds
            lp3_0 = model.n_3d_lp
            t0 = time.perf_counter()
            sim.iterate()
            step_times.append(time.perf_counter() - t0)
            callback_times.append(model.callback_seconds - cb0)
            frac_3d.append((model.n_3d_lp - lp3_0) / len(agents))
            if i % args.clearance_every == 0:
                min_clearance = min(min_clearance, running_min_clearance(sim, agents, args.agent_radius))
            if args.stop_when_arrived and all(sim.agent(a).state.arrived for a in agents):
                break
    except Exception as e:  # noqa: BLE001 - recorded, not swallowed silently
        crashed = {"step": len(step_times), "time": sim.elapsed_time(), "error": repr(e)}

    if writer is not None:
        writer.close()

    summary = {
        "scenario": "circle",
        "n": args.n,
        "rows": args.rows,
        "spacing": args.spacing,
        "circle_radius": circle_radius,
        "dt": args.dt,
        "agent_radius": args.agent_radius,
        "speed": args.speed,
        "tau": args.tau,
        "tau_obst": args.tau_obst,
        "initial_speed": args.initial_speed,
        "max_neighbors": args.max_neighbors,
        "pref_noise": args.pref_noise,
        "seed": args.seed,
        "benchmark_only": benchmark,
        "n_steps_completed": len(step_times),
        "sim_time_completed": len(step_times) * args.dt,
        "mean_step_time_ms": float(np.mean(step_times) * 1000) if step_times else None,
        "mean_callback_time_ms": float(np.mean(callback_times) * 1000) if callback_times else None,
        "step_times_ms": [round(x * 1000, 3) for x in step_times],
        "fraction_3d_lp_per_step": [round(x, 4) for x in frac_3d],
        "min_clearance_running": float(min_clearance),
        "n_arrived_flag": int(sum(sim.agent(a).state.arrived for a in agents)),
        "db_path": None if benchmark else str(db_path),
        "targets": {str(k): v for k, v in targets.items()},
        "agent_ids": agents,
        "crashed": crashed,
    }
    json_path = RESULTS / f"{args.tag}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=1)
    print(
        f"Wrote {json_path}: {len(step_times)} steps, "
        f"{summary['mean_step_time_ms']:.1f} ms/step, min clearance {min_clearance:+.4f} m, "
        f"arrived {summary['n_arrived_flag']}/{args.n}, crashed={crashed}"
    )


if __name__ == "__main__":
    main()
