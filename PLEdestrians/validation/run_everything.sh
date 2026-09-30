#!/bin/bash
# Regenerate every result used by the notebook (a few hours of CPU; the SFM
# concentric run at dt=1e-4 dominates). Needs the venv and
# jupedsim/build/environment sourced.
cd "$(dirname "$0")"
mkdir -p results/logs
./run_all_table1.sh ple rvo sfm &
./run_all_corridor.sh &
./run_scans.sh &
python run_narrow_passage.py --tag narrow_passage > results/logs/narrow_passage.log 2>&1 &
wait
echo "all done"
