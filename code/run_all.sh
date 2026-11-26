#!/usr/bin/env bash
# Full pipeline. Usage:  bash run_all.sh path/to/_BothDeathwhistlefile98.stl [dx_mm] [sim_time_s]
# Defaults reproduce the published run: dx = 0.5 mm, 10 ms per case (~1.5 h per case on 2 cores).
set -euo pipefail
STL=${1:?give the STL path}; DX=${2:-0.5}; T=${3:-10e-3}; W=${WORKDIR:-out}
python src/01_geometry.py --stl "$STL" --dx "$DX" --workdir "$W"
python src/02_simulate.py --case orig   --time "$T" --workdir "$W"
python src/02_simulate.py --case sealed --time "$T" --workdir "$W"
python src/04_figures.py  --workdir "$W"
python src/05_vortices.py --workdir "$W" --case orig
python src/06_audio.py    --workdir "$W"
python src/03_video.py    --workdir "$W"        # needs ffmpeg on PATH
echo "Results in $W/ (figs/, death_whistle_3d_flow.mp4, death_whistle_simulated_sound.wav)"
