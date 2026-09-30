#!/bin/bash
# Long Corridor: the main run (paper's 1.33 agents/m^2, Fig. 9) and a
# density sweep for Fig. 7, each point in its own OS process.
cd "$(dirname "$0")"
L=results/logs
python run_corridor.py --density 1.3333 --block-length 40 --sim-time 40 --tag corridor_main > $L/corridor_main.log 2>&1 &
for rho in 0.3 0.6 1.0 1.5 2.0 2.5 2.8; do
  python run_corridor.py --density $rho --block-length 20 --sim-time 30 --tag corridor_rho$rho > $L/corridor_rho$rho.log 2>&1 &
done
wait
