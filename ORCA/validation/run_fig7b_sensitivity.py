# SPDX-License-Identifier: LGPL-3.0-or-later
"""Five-robot circle (Fig. 7b): does the exactly symmetric setup resolve or
deadlock? Sweeps the time horizon tau, the start speed (from rest vs.
already walking at v_pref), the start-circle radius and the RVO2-style
symmetry-breaking noise; the paper states none of these for Fig. 7.

Each point is its own `run_circle.py` process. Writes
results/fig7b_sensitivity.json.

Usage:
    python run_fig7b_sensitivity.py
"""

import itertools
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SWEEP_DIR = RESULTS / "fig7b_sweep"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)

DT = 0.025
TAUS = [0.5, 1.0, 2.0, 4.0]
START_SPEEDS = [0.0, 1.5]
RADII = [1.1, 2.0, 5.0]
NOISES = [0.0, 1e-4]


def main():
    rows = []
    for tau, v0, radius, noise in itertools.product(TAUS, START_SPEEDS, RADII, NOISES):
        tag = f"fig7b_sweep/tau{tau}_v{v0}_R{radius}_noise{noise}"
        sim_time = 2 * radius / 1.5 + 12.0
        subprocess.run(
            [
                sys.executable, str(HERE / "run_circle.py"),
                "--n", "5", "--circle-radius", str(radius), "--dt", str(DT),
                "--sim-time", str(sim_time), "--tau", str(tau),
                "--initial-speed", str(v0), "--pref-noise", str(noise),
                "--stop-when-arrived", "--every-nth-frame", "4", "--tag", tag,
            ],
            check=True,
            capture_output=True,
        )
        with open(RESULTS / f"{tag}.json") as f:
            s = json.load(f)
        rows.append(
            {
                "tau": tau,
                "initial_speed": v0,
                "circle_radius": radius,
                "pref_noise": noise,
                "n_arrived": s["n_arrived_flag"],
                "sim_time_completed": s["sim_time_completed"],
                "sim_time_limit": sim_time,
                "min_clearance": s["min_clearance_running"],
                "crashed": s["crashed"],
            }
        )
        print(rows[-1])
    with open(RESULTS / "fig7b_sensitivity.json", "w") as f:
        json.dump({"dt": DT, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
