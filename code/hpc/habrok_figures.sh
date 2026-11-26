#!/bin/bash
# Regenerate the paper figures, the key numbers, the sound files and the summary tables from the new runs.
# Everything is collected in results/ (small files, committed to Git).
#   sbatch code/hpc/habrok_figures.sh
#SBATCH --job-name=whistle-figs
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=you@example.org   # set your address and remove one #
#SBATCH --output=runs/%x-%j.out
set -uo pipefail
module purge
module load Python/3.11.3-GCCcore-12.3.0
source "$HOME/venvs/whistle/bin/activate"
export PATH=$HOME/bin:$PATH PYTHONUNBUFFERED=1 NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "commit=$(git rev-parse --short HEAD) start=$(date)"
step() { echo "=== $*"; "$@" || echo "!!! FAILED: $*"; }

OUT=results; mkdir -p $OUT/tables $OUT/sound $OUT/logs
MAIN=runs/dx0.25_dp2000_long      # finest grid, longest signal: the headline case
PAIR=runs/dx0.25_dp2000           # finest grid, original and sealed (20 ms each)

# 1. Paper figures (red / blue / black) and the numbers quoted in the paper
step python code/paper/paper_figures.py --runs runs --out $OUT/paper_figures --stl code/input/_BothDeathwhistlefile98.stl

# 2. Sound of the headline case and of original versus sealed
step python code/src/06_audio.py --workdir $MAIN
cp $MAIN/death_whistle_simulated_sound.wav $OUT/sound/sound_0p25mm_2kPa_long.wav 2>/dev/null
step python code/src/06_audio.py --workdir $PAIR
cp $PAIR/death_whistle_simulated_sound.wav $OUT/sound/sound_0p25mm_2kPa_orig_vs_sealed.wav 2>/dev/null

# 3. Summary tables
step python code/src/07_compare_runs.py runs/dx{1,0.5,0.25}_dp2000/res_orig.npz   --t0 5e-3 --t1 20e-3 --csv $OUT/tables/convergence_orig.csv
step python code/src/07_compare_runs.py runs/dx{1,0.5,0.25}_dp2000/res_sealed.npz --t0 5e-3 --t1 20e-3 --csv $OUT/tables/convergence_sealed.csv
step python code/src/07_compare_runs.py runs/dx0.5_dp{500,1000,1500,2000,3000,4000}/res_orig.npz --t0 5e-3 --csv $OUT/tables/sweep_0p5mm.csv
step python code/src/07_compare_runs.py runs/dx0.25_dp{2000,4000}/res_orig.npz --t0 5e-3 --t1 20e-3 --csv $OUT/tables/check_0p25mm.csv
step python code/src/08_roughness.py runs/dx*_dp*_long/res_orig.npz --t0 20e-3 --csv $OUT/tables/roughness.csv

# 4. Run logs
cp runs/w-*.out $OUT/logs/ 2>/dev/null
echo "done $(date)"; ls -la $OUT $OUT/*/
