#!/bin/bash
# Render the explainer video from the fine-grid runs into results/video/.
# The .mp4 stays on Hábrók (too large for Git); the thumbnail, chapter list and preview sheet are small and can be committed.
#   sbatch code/hpc/habrok_video.sh
# Quick low-resolution test into its own folder (any option of make_video.py can be added after the script name):
#   sbatch --time=02:00:00 code/hpc/habrok_video.sh --scale 0.5 --fps 10 --out results/video_test
#SBATCH --job-name=whistle-video
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=runs/%x-%j.out
set -uo pipefail
module purge
module load Python/3.11.3-GCCcore-12.3.0
source "$HOME/venvs/whistle/bin/activate"
export PATH=$HOME/bin:$PATH PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "commit=$(git rev-parse --short HEAD) start=$(date)"
python code/video/make_video.py --runs runs --out results/video --stl code/input/_BothDeathwhistlefile98.stl \
       --numbers results/paper_figures/numbers.json --recordings experiments "$@"
echo "done $(date)"; ls -la results/video*
