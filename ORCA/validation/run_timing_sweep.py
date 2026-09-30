# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fig. 10(b) timing sweep: wall-clock time per step vs. number of agents,
for the Circle and the Office scenarios, N = 500 ... 5,000 as in the paper.

Each point is its own OS process (the Python callback holds the GIL), run
one after another so that points never compete for the CPU. Each process
times `--benchmark-steps` steps from the start of the scenario (no
trajectory is written). Density is kept constant as N grows: the circle
gets proportionally longer rows (4 rows, 0.8 m spacing), the office floor
plan is scaled by sqrt(N / 1000).

Writes results/timing_sweep.json.

Usage:
    python run_timing_sweep.py
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SWEEP_DIR = RESULTS / "timing_sweep"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)

NS = [500, 1000, 2000, 3000, 4000, 5000]
STEPS = 40
DT = 0.025


def run(cmd):
    subprocess.run([sys.executable, *cmd], check=True, capture_output=True)


def main():
    rows = []
    for n in NS:
        for scenario in ["circle", "office"]:
            tag = f"timing_sweep/{scenario}_n{n}"
            if scenario == "circle":
                run([
                    str(HERE / "run_circle.py"), "--n", str(n), "--rows", "4", "--spacing", "0.8",
                    "--dt", str(DT), "--benchmark-steps", str(STEPS), "--clearance-every", "1000",
                    "--tag", tag,
                ])
            else:
                run([
                    str(HERE / "run_office.py"), "--n", str(n), "--scale", f"{np.sqrt(n / 1000):.4f}",
                    "--dt", str(DT), "--benchmark-steps", str(STEPS), "--clearance-every", "1000",
                    "--tag", tag,
                ])
            with open(RESULTS / f"{tag}.json") as f:
                s = json.load(f)
            st = np.array(s["step_times_ms"][5:])  # drop warm-up steps
            rows.append({
                "scenario": scenario,
                "n": n,
                "n_spawned": s.get("n_spawned", n),
                "steps_timed": len(st),
                "mean_step_time_ms": float(st.mean()),
                "median_step_time_ms": float(np.median(st)),
                "mean_callback_time_ms": s["mean_callback_time_ms"],
                "crashed": s["crashed"],
            })
            print(rows[-1], flush=True)
    with open(RESULTS / "timing_sweep.json", "w") as f:
        json.dump({"dt": DT, "benchmark_steps": STEPS, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
