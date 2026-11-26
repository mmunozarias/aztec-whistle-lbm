#!/bin/bash
# General run: set CASE, DX [mm], DP [Pa], T [s] via --export. Example:
#   sbatch -c 16 --mem=8G -t 03:00:00 --export=ALL,CASE=sealed,DX=0.5,DP=2000,T=40e-3 code/hpc/habrok_run.sh
#SBATCH --job-name=whistle-run
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=you@example.org   # set your address and remove one #
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
module purge
module load Python/3.11.3-GCCcore-12.3.0
source "$HOME/venvs/whistle/bin/activate"
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
CASE=${CASE:?}; DX=${DX:?}; DP=${DP:?}; T=${T:?}
STL=${STL:-code/input/_BothDeathwhistlefile98.stl}
W=runs/dx${DX}_dp${DP}${TAG:-}; mkdir -p "$W"
echo "case=$CASE dx=$DX dp=$DP T=$T cores=$SLURM_CPUS_PER_TASK commit=$(git rev-parse --short HEAD) start=$(date)"
[ -f "$W/geom.npz" ] || python code/src/01_geometry.py --stl "$STL" --dx $DX --workdir "$W"
python code/src/02_simulate.py --case $CASE --dp $DP --time $T --workdir "$W"
echo "done $(date)"