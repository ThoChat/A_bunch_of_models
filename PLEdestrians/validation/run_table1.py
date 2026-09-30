# SPDX-License-Identifier: LGPL-3.0-or-later
"""Table 1 / Fig. 8 benchmarks of Guy et al. (2010), Sec. 5.1-5.2:

  swap2      "2-Agent swapping" (#1): two agents swap ends of a 10 m line,
             with the small lateral offset read off Fig. 8.
  circle10   "10-Agent Circle" (#2): agents walk to antipodal points.
  concentric "Concentric Circles" (#3): 34 agents on an inner and 66 on an
             outer circle, each walking to its antipodal point.

Each runs with PLEdestrians or, for the paper's method comparison, with the
repo's finished RVO (RVO/pyrvo.py) and Helbing social-force
(SFM/pysocial_force.py) models, all at the same radius and desired speed.

Usage:
    python run_table1.py --scenario swap2 --model ple --tag t1_swap2_ple
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "RVO"))
sys.path.insert(0, str(HERE.parent.parent / "SFM"))

import jupedsim as jps
from jupedsim.sqlite_serialization import SqliteTrajectoryWriter
from pyPLEdestrians import ES, EW, PLEdestriansModel, PLEState

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
V_DES = float(np.sqrt(ES / EW))  # 1.33 m/s, Lemma 1

# Swap distance of #1/#2. The text says "a 10m radius circle", but Table 1
# rules that out: PLE's 33.3 J/kg for #1 equals Corollary 1's minimum
# 2 L sqrt(es ew) for L = 9.9 m, and 7.5 s = 10 m / 1.33 m/s. So agents
# travel 10 m, i.e. a circle of 5 m radius.
SWAP_LENGTH = 10.0
# Fig. 8, hand-digitized from the 300 dpi crop: the two agents' start/end
# markers are 36 px apart vertically over 856 px horizontally, i.e. 0.042 of
# the path length, 0.42 m for a 10 m swap. (Assumes equal axis scales,
# which the figure does not state.)
FIG8_LATERAL_OFFSET = 0.42
# Concentric Circles: radii are not given. Outer = the text's "10m radius
# circle"; inner chosen so both circles have the same spacing between
# neighbours (66 vs 34 agents -> 0.95 m arc spacing on both).
R_OUTER = 10.0
R_INNER = R_OUTER * 34 / 66


def layout(scenario, rng, jitter=0.01):
    """Return list of (start, goal) pairs."""
    if scenario == "swap2":
        h = SWAP_LENGTH / 2
        o = FIG8_LATERAL_OFFSET / 2
        return [((-h, o), (h, o)), ((h, -o), (-h, -o))]
    rings = {"circle10": [(10, SWAP_LENGTH / 2)],
             "concentric": [(34, R_INNER), (66, R_OUTER)]}[scenario]
    pairs = []
    for n, R in rings:
        # Tiny angular jitter (+-0.01 rad), identical for every model:
        # exactly antipodal pairs are a measure-zero degenerate case.
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(-jitter, jitter, n)
        for a in ang:
            s = (R * np.cos(a), R * np.sin(a))
            pairs.append((s, (-s[0], -s[1])))
    return pairs


def make_model_and_state(model, args, start, goal, v_des=V_DES):
    if model == "ple":
        m = PLEdestriansModel(pref_noise=args.pref_noise, seed=args.seed)
        # Heterogeneous agents keep sqrt(es ew) (hence Corollary 1's minimum
        # energy per metre) and vary v_des = sqrt(es / ew).
        c = np.sqrt(ES * EW)
        st = PLEState(position=start, goal=goal, radius=args.radius, tau=args.tau,
                      time_horizon=args.time_horizon, es=c * v_des, ew=c / v_des)
    elif model == "rvo":
        from pyrvo import ReciprocalVelocityObstacleModel, RVOState
        m = ReciprocalVelocityObstacleModel()
        st = RVOState(velocity=(0.0, 0.0), radius=args.radius, max_speed=v_des)
    elif model == "sfm":
        from pysocial_force import PythonSocialForceModel, PythonSocialForceModelState
        m = PythonSocialForceModel()
        st = PythonSocialForceModelState(velocity=(0.0, 0.0), radius=args.radius,
                                         desired_speed=v_des)
    return m, st


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=["swap2", "circle10", "concentric"], required=True)
    p.add_argument("--model", choices=["ple", "rvo", "sfm"], default="ple")
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--pref-noise", type=float, default=0.0,
                   help="symmetry-breaking noise [m/s]; tested, not needed (see notebook)")
    p.add_argument("--time-horizon", type=float, default=2.0)
    p.add_argument("--radius", type=float, default=0.3)
    p.add_argument("--sim-time", type=float, default=None)
    p.add_argument("--arrive-tol", type=float, default=0.2)
    p.add_argument("--frame-interval", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--jitter", type=float, default=0.01, help="start angle jitter [rad]")
    p.add_argument("--speed-spread", type=float, default=0.0,
                   help="v_des drawn uniformly in V_DES*(1 +- spread)")
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    if args.dt is None:
        args.dt = {"ple": 0.05, "rvo": 0.025, "sfm": 1e-4}[args.model]
    if args.sim_time is None:
        args.sim_time = {"swap2": 25.0, "circle10": 40.0, "concentric": 200.0}[args.scenario]
    every = max(1, int(round(args.frame_interval / args.dt)))

    rng = np.random.default_rng(args.seed)
    pairs = layout(args.scenario, rng, args.jitter)
    v_des_all = V_DES * (1 + rng.uniform(-args.speed_spread, args.speed_spread, len(pairs)))
    bound = R_OUTER + 3 if args.scenario == "concentric" else SWAP_LENGTH / 2 + 3
    geometry = [(-bound, -bound), (bound, -bound), (bound, bound), (-bound, bound)]

    db_path = RESULTS / f"{args.tag}.sqlite"
    writer = SqliteTrajectoryWriter(output_file=db_path, every_nth_frame=every)
    model, _ = make_model_and_state(args.model, args, (0, 0), (0, 0))
    sim = jps.Simulation(model=model, geometry=geometry, dt=args.dt, trajectory_writer=writer)

    ids, goals = [], {}
    for (start, goal), vd in zip(pairs, v_des_all):
        _, st = make_model_and_state(args.model, args, start, goal, vd)
        wp = sim.add_waypoint_stage(goal, args.arrive_tol)
        j = sim.add_journey(jps.JourneyDescription([wp]))
        aid = sim.add_agent(journey_id=j, stage_id=wp, position=start, state=st)
        ids.append(aid)
        goals[aid] = goal

    n_steps = int(args.sim_time / args.dt)
    check_every = max(1, int(0.5 / args.dt))
    arrived = set()
    crashed = None
    t_all_arrived = None
    max_drift = 0.0
    wall0 = time.perf_counter()
    try:
        for i in range(n_steps):
            sim.iterate()
            if i % check_every == 0:
                for aid in ids:
                    a = sim.agent(aid)
                    pos = np.array(a.position)
                    if np.hypot(*(pos - goals[aid])) <= args.arrive_tol:
                        arrived.add(aid)
                    if args.model == "ple":
                        max_drift = max(max_drift, float(np.hypot(*(pos - a.state.position))))
                if len(arrived) == len(ids):
                    t_all_arrived = sim.elapsed_time()
                    # 1 s of settling after the last arrival, then stop.
                    for _ in range(int(1.0 / args.dt)):
                        sim.iterate()
                    break
    except Exception as e:  # noqa: BLE001 - recorded in the summary
        crashed = {"step": sim.iteration_count(), "time": sim.elapsed_time(), "error": repr(e)}
    wall = time.perf_counter() - wall0
    writer.close()

    summary = {
        "scenario": args.scenario, "model": args.model, "dt": args.dt, "tau": args.tau, "pref_noise": args.pref_noise,
        "time_horizon": args.time_horizon, "radius": args.radius, "v_des": V_DES,
        "arrive_tol": args.arrive_tol, "seed": args.seed, "n_agents": len(ids), "jitter": args.jitter,
        "speed_spread": args.speed_spread,
        "v_des_agents": {str(a): float(v) for a, v in zip(ids, v_des_all)},
        "frame_interval": every * args.dt,
        "swap_length": SWAP_LENGTH, "fig8_lateral_offset": FIG8_LATERAL_OFFSET,
        "r_inner": R_INNER, "r_outer": R_OUTER,
        "goals": {str(k): v for k, v in goals.items()},
        "starts": {str(a): s for a, (s, _) in zip(ids, pairs)},
        "t_all_arrived_check": t_all_arrived,
        "n_arrived_check": len(arrived),
        "sim_time_run": sim.elapsed_time(),
        "wall_seconds": wall,
        "n_pv_empty": getattr(model, "n_pv_empty", None),
        "max_dead_reckoning_drift": max_drift if args.model == "ple" else None,
        "db_path": str(db_path),
        "crashed": crashed,
    }
    (RESULTS / f"{args.tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"{args.tag}: arrived {len(arrived)}/{len(ids)} by t={t_all_arrived}, "
          f"wall {wall:.1f}s, crashed={crashed}, drift={max_drift:.2e}")


if __name__ == "__main__":
    main()
