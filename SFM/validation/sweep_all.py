# SPDX-License-Identifier: LGPL-3.0-or-later
"""Orchestrates all validation runs for the three paper figures as separate
OS processes (needed for real parallelism: each run holds the Python GIL
almost continuously inside jupedsim's per-agent Python callback), with a
concurrency cap so we don't oversubscribe the machine's cores.

Usage: python sweep_all.py
"""
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
PYTHON = sys.executable
MAX_CONCURRENT = 10

FIG1_V0 = [0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
FIG2_PHI = [0, 10, 20, 30, 40, 50]
FIG3_P = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

JOBS = []
for v0 in FIG1_V0:
    JOBS.append(
        [
            PYTHON, str(HERE / "run_fig1_room_evacuation.py"),
            "--v0", str(v0), "--n-agents", "50", "--seed", "1",
            "--max-time", "90", "--out-dir", "results",
        ]
    )
for phi in FIG2_PHI:
    JOBS.append(
        [
            PYTHON, str(HERE / "run_fig2_corridor_widening.py"),
            "--phi", str(phi), "--duration", "60", "--warmup", "15",
            "--out-dir", "results",
        ]
    )
for p in FIG3_P:
    JOBS.append(
        [
            PYTHON, str(HERE / "run_fig3_smoky_room.py"),
            "--p", str(p), "--seed", "1", "--duration", "40",
            "--out-dir", "results",
        ]
    )


def main():
    (HERE / "results").mkdir(exist_ok=True)
    log_dir = HERE / "results" / "logs"
    log_dir.mkdir(exist_ok=True)

    running = {}  # proc -> (job, log_file_handle)
    pending = list(JOBS)
    completed = []
    t_start = time.time()

    print(f"Total jobs: {len(JOBS)}, max concurrent: {MAX_CONCURRENT}")

    while pending or running:
        while pending and len(running) < MAX_CONCURRENT:
            job = pending.pop(0)
            name = "_".join(job[1:2] + job[3:5]).replace("/", "_")
            log_path = log_dir / f"{pathlib.Path(job[1]).stem}_{job[3]}_{job[4]}.log"
            lf = open(log_path, "w")
            proc = subprocess.Popen(job, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
            running[proc] = (job, lf, time.time())
            print(f"[{time.time()-t_start:6.0f}s] started: {' '.join(job[1:])}")

        time.sleep(5)
        done_procs = [p for p in running if p.poll() is not None]
        for p in done_procs:
            job, lf, t0 = running.pop(p)
            lf.close()
            rc = p.returncode
            dur = time.time() - t0
            completed.append((job, rc, dur))
            status = "OK" if rc == 0 else f"FAILED(rc={rc})"
            print(f"[{time.time()-t_start:6.0f}s] finished ({dur:.0f}s, {status}): {' '.join(job[1:])}")

    n_failed = sum(1 for _, rc, _ in completed if rc != 0)
    print(f"\nAll {len(completed)} jobs done in {time.time()-t_start:.0f}s. Failed: {n_failed}")
    with open(HERE / "results" / "sweep_done.flag", "w") as f:
        f.write(f"completed={len(completed)} failed={n_failed} total_seconds={time.time()-t_start:.0f}\n")
        for job, rc, dur in completed:
            f.write(f"{' '.join(job[1:])} rc={rc} dur={dur:.0f}s\n")


if __name__ == "__main__":
    main()
