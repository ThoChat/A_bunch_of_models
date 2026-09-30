#!/bin/bash
# Parameter scans for the notebook's implementation notes:
#   dt  in {0.2, 0.1, 0.05, 0.025, 0.0125} s  (PLE, swap2 + circle10 seed 1)
#   tau in {0.05, 0.25, 0.5, 1, 2} s          (PLE, swap2 + circle10 seeds 1-3)
#   SFM dt in {2e-3, 1e-3, 5e-4, 1e-4} s      (swap2 + circle10 seed 1)
cd "$(dirname "$0")"
L=results/logs
for dt in 0.2 0.1 0.05 0.025 0.0125; do
  (python run_table1.py --scenario swap2 --dt $dt --tag dtscan_swap2_dt$dt
   python run_table1.py --scenario circle10 --speed-spread 0.1 --dt $dt --tag dtscan_circle10_dt$dt) > $L/dtscan_$dt.log 2>&1 &
done
for tau in 0.05 0.25 0.5 1.0 2.0; do
  (python run_table1.py --scenario swap2 --tau $tau --tag tauscan_swap2_tau$tau
   for s in 1 2 3; do
     python run_table1.py --scenario circle10 --speed-spread 0.1 --tau $tau --seed $s --tag tauscan_circle10_tau${tau}_s$s
   done) > $L/tauscan_$tau.log 2>&1 &
done
for dt in 0.002 0.001 0.0005 0.0001; do
  (python run_table1.py --scenario swap2 --model sfm --dt $dt --tag sfmdt_swap2_dt$dt
   python run_table1.py --scenario circle10 --model sfm --speed-spread 0.1 --dt $dt --tag sfmdt_circle10_dt$dt) > $L/sfmdt_$dt.log 2>&1 &
done
wait
