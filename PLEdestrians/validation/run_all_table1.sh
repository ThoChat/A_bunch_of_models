#!/bin/bash
# Final Table 1 runs: every model, every benchmark; circle10 over 5 seeds.
# Usage: ./run_all_table1.sh [models...]   (default: ple rvo sfm)
# Each point runs as its own OS process (the Python callback holds the GIL).
cd "$(dirname "$0")"
L=results/logs
for m in ${@:-ple rvo sfm}; do
  python run_table1.py --scenario swap2 --model $m --tag t1_swap2_$m > $L/t1_swap2_$m.log 2>&1 &
  for s in 1 2 3 4 5; do
    python run_table1.py --scenario circle10 --model $m --speed-spread 0.1 --seed $s --tag t1_circle10_${m}_s$s > $L/t1_circle10_${m}_s$s.log 2>&1 &
  done
  python run_table1.py --scenario concentric --model $m --speed-spread 0.1 --seed 1 --sim-time 400 --tag t1_concentric_$m > $L/t1_concentric_$m.log 2>&1 &
done
wait
