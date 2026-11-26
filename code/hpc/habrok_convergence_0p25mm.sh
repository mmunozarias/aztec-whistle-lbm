#!/bin/bash
# Grid-convergence run: 0.25 mm grid, 2 kPa blowing pressure, original geometry (case O).
# Compare with the 0.5 mm, 2 kPa run of habrok_pressure_sweep.sh (same 20 ms window or longer).
#
# Size:      241 x 497 x 305 = 36.5 M cells, dt = 0.42 us, 20 ms = 47 528 steps (1.7e12 cell updates)
# Memory:    about 3 GB for the geometry step, about 11 GB for the solver  -> request 32 GB
# Cores:     64 (NUMBA_NUM_THREADS = 64)
# Time:      2.5 to 4 h expected at 150-200 M cell updates/s on 64 cores; up to ~10 h if
#            scaling is poor. Requested: 24 h (margin; there is no restart file, so a timeout loses the run).
# Sponge:    02_simulate.py keeps the absorbing layer at 6 mm with the same damping rate and added
#            viscosity on every grid (24 cells here, 12 at 0.5 mm), so only the resolution changes.
#            The 0.5 mm output is bit-identical to the code used for the paper.
# Output:    runs/dx0.25_dp2000/  res_orig.npz (probes every step, slices every 21 us),
#            snap_orig.npy (final 3D field, about 290 MB), geom.npz
#
# Before submitting (once):
#   module avail Python                         # pick a recent Python 3.11+ module
#   python -m venv ~/venvs/whistle && source ~/venvs/whistle/bin/activate
#   pip install -r code/requirements.txt
#   copy the STL to code/input/_BothDeathwhistlefile98.stl (Thingiverse thing 5148690, not in the repo)
# Submit from the repository root:
#   mkdir -p runs                                # SLURM writes its log there
#   sbatch code/hpc/habrok_convergence_0p25mm.sh
#
#SBATCH --job-name=whistle-dx025
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=you@example.org   # set your address and remove one #
#SBATCH --output=runs/%x-%j.out

set -euo pipefail
module purge
module load Python/3.11.3-GCCcore-12.3.0        # adjust to the module you found with 'module avail Python'
source "$HOME/venvs/whistle/bin/activate"
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK      # libigl winding numbers in the geometry step

STL=${STL:-code/input/_BothDeathwhistlefile98.stl}
DX=0.25; DP=2000; T=20e-3
W=runs/dx${DX}_dp${DP}
mkdir -p "$W"
[ -f "$W/geom.npz" ] || python code/src/01_geometry.py --stl "$STL" --dx $DX --workdir "$W"
python code/src/02_simulate.py --case ${CASE:-orig} --dp $DP --time $T --workdir "$W"
# Same 5-20 ms window on both grids; includes the 0.5 mm, 2 kPa run of the sweep if it has finished.
python code/src/07_compare_runs.py $(ls -d runs/dx0.5_dp2000 2>/dev/null) "$W" --t0 5e-3 --t1 20e-3 --csv "$W/convergence_summary.csv" || true
