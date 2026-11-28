# Simulation and analysis pipeline

Run the scripts from the repository root; each one lists its options with `--help`.

| Script | What it does |
|---|---|
| `src/01_geometry.py` | Voxelizes the STL with the generalized winding number on a grid of `--dx` mm; writes `geom.npz` (solid for the original and sealed cases, inlet, coordinates) |
| `src/02_simulate.py` | D3Q19 regularized lattice Boltzmann LES with a Smagorinsky model, physical speed of sound and viscosity of air; `--case orig|sealed`, `--dp` blowing pressure [Pa], `--time` [s]. Writes `res_<case>.npz` (probe pressures P1-P5 and duct flow at every step, mid-plane and cross-plane fields every 21 µs) and `snap_<case>.npy` (final 3D field) |
| `src/03_video.py` | Animation of speed and pressure on the mid-plane (needs `ffmpeg`) |
| `src/04_figures.py` | Quick-look figures: spectrum, mean flow, sound sources |
| `src/05_vortices.py` | 3D vortex structures (Q-criterion) from the final field |
| `src/06_audio.py` | Audio clip of the signal at the near-mouth probe P5, band-passed and resampled to 48 kHz |
| `src/07_compare_runs.py` | Table across runs: tone, mean and amplitude of the sphere pressure, phase gap vs sphere, level at P5, flow, Strouhal number; `--t0/--t1` for a common window |
| `src/08_roughness.py` | Tonal fraction, envelope fluctuation and roughness (envelope modulation at 30-150 Hz) of the signal at P5 |
| `paper/paper_figures.py` | All figures of the paper and `numbers.json` |
| `hpc/habrok_run.sh` | General SLURM job: one case, grid, pressure and duration (`CASE`, `DX`, `DP`, `T`, optional `TAG`) |
| `hpc/habrok_figures.sh` | SLURM job that regenerates `results/` |
| `hpc/habrok_video.sh` | SLURM job that renders the explainer video with `video/make_video.py` |
| `run_all.sh` | Whole chain for one grid on a laptop or workstation |

Probes: P1 sphere centre, P2 duct, P3 top of the gap, P4 exit toward the mouth, P5 near-mouth probe (less than 1 cm in front of the mouth).

The video uses the Latin Modern Sans font (`lmsans10-regular.otf` and `lmsans10-bold.otf`, part of TeX Live and MiKTeX). Put them in `~/.local/share/fonts` or point `LMSANS_DIR` to them; without them the script falls back to `cmss10`, which lacks accented letters.

The absorbing layer at the domain boundary is 6 mm thick on every grid. The 1 mm grid is too coarse for the whistle to oscillate; use it only to test the installation.

Expected values for the reference runs are in [`../results/tables/`](../results/tables/) and [`../results/paper_figures/numbers.json`](../results/paper_figures/numbers.json).
