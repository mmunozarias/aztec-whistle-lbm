#!/bin/bash
# Blowing-pressure sweep on the 0.5 mm grid: 0.5, 1, 2 and 4 kPa, original geometry (case O).
# A SLURM job array runs the four pressures in parallel, each in its own folder.
# Each run is 40 ms long, so the settled signal (after about 5 ms) gives a frequency
# resolution of about 30 Hz instead of 143 Hz in the paper.
#
# Size per task:  121 x 249 x 153 = 4.6 M cells, dt = 0.84 us, 40 ms = 47 528 steps (2.2e11 cell updates)
# Memory per task: about 3 GB (solver 1.2 GB plus stored slices)  -> request 8 GB
# Cores per task:  16
# Time per task:   1 to 1.5 h expected at 40-55 M cell updates/s on 16 cores. Requested: 6 h.
# Total:           4 tasks x 16 cores; they can run at the same time.
# Notes:           at 0.5 kPa the whistle may not reach self-oscillation; that is a result, not a failure.
#                  At 4 kPa the jet reaches about 80 m/s (Mach 0.24). If a run stops with 'diverged',
#                  rerun that pressure at DX=0.35 or report the highest stable pressure.
# Output:          runs/dx0.5_dp<P>/res_orig.npz for P in 500 1000 2000 4000
#
# Submit from the repository root (after the one-time setup described in habrok_convergence_0p25mm.sh):
#   mkdir -p runs                                # SLURM writes its logs there
#   sbatch code/hpc/habrok_pressure_sweep.sh
# When all four tasks have finished:
#   python code/src/07_compare_runs.py runs/dx0.5_dp* --t0 5e-3 --csv runs/sweep_summary.csv --plot runs/sweep_regime.png
# Grid convergence (after habrok_convergence_0p25mm.sh has also finished):
#   python code/src/07_compare_runs.py runs/dx0.5_dp2000 runs/dx0.25_dp2000 --t0 5e-3 --t1 20e-3
#
#SBATCH --job-name=whistle-sweep
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=you@example.org   # set your address and remove one #
#SBATCH --array=0-3
#SBATCH --output=runs/%x-%A_%a.out

set -euo pipefail
module purge
module load Python/3.11.3-GCCcore-12.3.0        # adjust to the module you found with 'module avail Python'
source "$HOME/venvs/whistle/bin/activate"
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

PRESSURES=(500 1000 2000 4000)
DP=${PRESSURES[$SLURM_ARRAY_TASK_ID]}
STL=${STL:-code/input/_BothDeathwhistlefile98.stl}
DX=${DX:-0.5}; T=40e-3
W=runs/dx${DX}_dp${DP}
mkdir -p "$W"
[ -f "$W/geom.npz" ] || python code/src/01_geometry.py --stl "$STL" --dx $DX --workdir "$W"
python code/src/02_simulate.py --case orig --dp $DP --time $T --workdir "$W"
