# SPDX-License-Identifier: LGPL-3.0-or-later
"""Runs every validation data point as its own OS process (each run holds
the GIL inside jupedsim's per-agent Python callback, so threads would not
help), with a concurrency cap.

Usage:
    python sweep_all.py --only room crossing concert
    python sweep_all.py --only perf --max-concurrent 1   # timing runs, alone
"""
import argparse
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
PYTHON = sys.executable

SAFE = {"SF": "SF", "RVO": "RVO", "SPH": "SPH", "SF+SPH": "SFplusSPH", "RVO+SPH": "RVOplusSPH",
        "SF->SPH": "SFtoSPH", "RVO->SPH": "RVOtoSPH"}


def fmt(x):
    return f"{x:g}"


def room_jobs():
    jobs = []
    sweep = [("SF", "k", [50, 100, 250, 500, 1000]), ("RVO", "k", [1000])]
    sweep += [(p, "rho", [3, 4, 5, 6, 7, 8]) for p in ("SPH", "SF+SPH", "RVO+SPH")]
    sweep += [(p, "rho", [4, 5, 6]) for p in ("SF->SPH", "RVO->SPH")]
    for profile, kind, values in sweep:
        for v in values:
            tag = f"room_{SAFE[profile]}_{kind}{fmt(v)}"
            flag = ["--k-ag", fmt(v)] if kind == "k" else ["--rho0-max", fmt(v)]
            jobs.append((tag, [PYTHON, str(HERE / "run_room_evacuation.py"), "--profile", profile,
                               *flag, "--max-time", "500", "--tag", tag]))
    return jobs


def crossing_jobs():
    return [(f"crossing_{SAFE[p]}", [PYTHON, str(HERE / "run_crossing_bottleneck.py"), "--profile", p,
                                     "--tag", f"crossing_{SAFE[p]}"])
            for p in ("SF", "RVO", "SPH", "SF+SPH", "RVO+SPH", "SF->SPH", "RVO->SPH")]


CONCERT = {
    "concert_baseline": [],
    "concert_k50": ["--k-gas", "50"],
    "concert_k500": ["--k-gas", "500"],
    "concert_rho4": ["--rho0-max", "4", "--init-density", "4"],
    "concert_rho6": ["--rho0-max", "6"],
    "concert_mu0": ["--mu", "0"],
    "concert_static_rho4": ["--rho0-min", "4", "--rho0-max", "4", "--init-density", "4"],
    "concert_noSPH_kag200": ["--k-gas", "0", "--mu", "0", "--k-ag", "200", "--k-obs", "1000"],
    "concert_noSPH_kag1000": ["--k-gas", "0", "--mu", "0", "--k-ag", "1000", "--k-obs", "1000"],
    "concert_kag0": ["--k-ag", "0"],
}


def concert_jobs(skip=()):
    return [(tag, [PYTHON, str(HERE / "run_concert.py"), "--settle", "20", *extra, "--tag", tag])
            for tag, extra in CONCERT.items() if tag not in skip]


def perf_jobs():
    jobs = []
    for n in (1000, 2500, 5000, 10000):
        tag = f"perf_n{n}"
        jobs.append((tag, [PYTHON, str(HERE / "run_concert.py"), "--n-agents", str(n), "--settle", "1.0",
                           "--after", "0.02", "--every-nth-frame", "1000", "--tag", tag]))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=["room", "crossing", "concert"])
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--max-concurrent", type=int, default=11)
    args = ap.parse_args()

    jobs = []
    for group in args.only:
        jobs += {"room": room_jobs, "crossing": crossing_jobs, "concert": concert_jobs,
                 "perf": perf_jobs}[group]()
    jobs = [j for j in jobs if j[0] not in args.skip]
    # Longest jobs first so they don't end up as the tail.
    order = {"crossing": 0, "concert": 1, "room": 2, "perf": 3}
    jobs.sort(key=lambda j: order[j[0].split("_")[0]])

    log_dir = HERE / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    running, pending, completed = {}, list(jobs), []
    t_start = time.time()
    print(f"Total jobs: {len(jobs)}, max concurrent: {args.max_concurrent}", flush=True)
    while pending or running:
        while pending and len(running) < args.max_concurrent:
            tag, cmd = pending.pop(0)
            lf = open(log_dir / f"{tag}.log", "w")
            running[subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)] = (tag, lf, time.time())
            print(f"[{time.time() - t_start:6.0f}s] started {tag}", flush=True)
        time.sleep(5)
        for proc in [p for p in running if p.poll() is not None]:
            tag, lf, t0 = running.pop(proc)
            lf.close()
            completed.append((tag, proc.returncode, time.time() - t0))
            print(f"[{time.time() - t_start:6.0f}s] finished {tag} rc={proc.returncode} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    n_failed = sum(1 for _, rc, _ in completed if rc != 0)
    print(f"All {len(completed)} jobs done in {time.time() - t_start:.0f}s. Failed: {n_failed}", flush=True)


if __name__ == "__main__":
    main()
