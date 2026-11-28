# Aztec skull whistle: 3D lattice Boltzmann simulation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24629264.svg)](https://doi.org/10.5281/zenodo.24629264)

Code, results and recordings for a study of how the Aztec skull ("death") whistle makes its sound.

- **Paper:** M. Muñoz-Arias and J. Barradas-Berglind, *Why the Aztec Death Whistle Screams: A Three-Dimensional Lattice Boltzmann Simulation of a Skull-Whistle Replica* (manuscript, 2026; link to be added).
- **Code:** a 3D lattice Boltzmann large-eddy simulation (D3Q19, regularized BGK with a Smagorinsky model) that resolves the airflow and the sound together, with the real speed of sound and viscosity of air. In [`code/`](code/); archived on Zenodo, [doi:10.5281/zenodo.24629264](https://doi.org/10.5281/zenodo.24629264).
- **Results:** every figure, number and table in the paper, in [`results/`](results/).
- **Recordings:** audio of a 3D-printed model at 100 % and 125 % scale, with the analysis script, in [`experiments/`](experiments/).
- **Raw simulation output, simulated sound and video:** archived on Zenodo (DOI to be added).
- **Video:** an explainer of the results (link to be added).

## Main results

The jet from the duct grazes the lip of a small opening in a hollow sphere inside the skull. The pressures in the sphere and in the gap around it oscillate in anti-phase, and the strongest pressure fluctuations (about 150 dB) sit at the lip, where vortices are shed in step with the oscillation.

| | Result |
|---|---|
| Tone at 2 kPa | **2.3 kHz**, self-excited: 2.28 kHz on the 0.5 mm grid and 2.33 kHz on the 0.25 mm grid, 2 % apart. A 1 mm grid does not whistle at all. |
| Blowing pressure | No sustained tone at 0.5 kPa. From 1 to 2 kPa the pitch rises from 1.90 to 2.31 kHz, as the jet speed to the power 0.57, consistent with a jet locked to a resonator. The flow follows the orifice law. |
| Quality of the sound | Pitched noise. At 2 kPa the tone and its harmonics carry 37 % (0.5 mm grid) and 22 % (0.25 mm grid) of the power near the mouth, and the envelope varies by 31 to 62 % of its mean. On the 0.25 mm grid a second band appears one octave below the tone. |
| Base slots sealed | Little change on the 0.5 mm grid; on the 0.25 mm grid the whistle locks onto a different mode near 1.44 kHz. |
| Printed model | Blown by mouth (pressure not measured), the 100 % print shows two bands at 1.24 and 2.46 kHz, ratio 1.99: the octave pair of the simulation, about 7 % higher. The 125 % print did not drop in pitch as expected. Qualitative only. |
| Not yet converged | Amplitudes, flow rate, broadband level, the form of the envelope fluctuations and the sealed case. The 0.5 mm grid is reliable only up to 2 kPa. |

See the paper for the full analysis and its limitations.

## Repository layout

| Folder | Contents |
|---|---|
| `code/src/` | Simulation and analysis pipeline, `01_geometry.py` to `08_roughness.py` (see [`code/README.md`](code/README.md)) |
| `code/paper/` | `paper_figures.py`: all figures of the paper and `numbers.json`, from the simulation output |
| `code/hpc/` | SLURM job scripts as used on Hábrók, the cluster of the University of Groningen (general run, convergence, pressure sweep, figures, video) |
| `code/video/` | `make_video.py`: the explainer video, rendered from the simulation output |
| `results/` | `paper_figures/` (PDF and PNG figures, `numbers.json`), `tables/` (CSV summaries), `video/` (thumbnail, preview, chapters) |
| `experiments/` | Recordings of the 3D-printed model (WAV), `recordings.csv`, `analyse_recordings.py`, and its figure and numbers |

## Input geometry

The simulations use the openly licensed model **"Working Death Whistle from Ghostbusters: Afterlife" by HawkWorley**, [Thingiverse thing 5148690](https://www.thingiverse.com/thing:5148690), licensed CC BY. It is not stored in this repository. Download it and use the file `_BothDeathwhistlefile98.stl` in its original orientation. The scripts expect it at `code/input/_BothDeathwhistlefile98.stl`; the inlet, seal region and probe positions refer to that file's coordinates in millimeters.

## Reproducing the results

Requirements: Python 3.10 or newer (3.11 was used on Hábrók).

```bash
python -m venv ~/venvs/whistle && source ~/venvs/whistle/bin/activate
pip install -r code/requirements.txt
mkdir -p code/input runs          # put the STL in code/input/
```

**Quick check on a laptop (1 mm grid, a few minutes).** Runs the whole chain to confirm the installation. The 1 mm grid is too coarse to whistle.

```bash
cd code && bash run_all.sh input/_BothDeathwhistlefile98.stl 1.0 4e-3
```

**The runs of the paper.** One run is one geometry (`orig` or `sealed`), one grid, one blowing pressure and one duration. On a SLURM cluster, from the repository root:

```bash
sbatch -c 16 --mem=8G  -t 03:00:00 --export=ALL,CASE=orig,DX=0.5,DP=2000,T=40e-3 code/hpc/habrok_run.sh
sbatch -c 64 --mem=32G -t 24:00:00 --export=ALL,CASE=orig,DX=0.25,DP=2000,T=20e-3 code/hpc/habrok_run.sh
sbatch -c 64 --mem=128G -t 24:00:00 --export=ALL,CASE=orig,DX=0.25,DP=2000,T=105e-3,TAG=_long code/hpc/habrok_run.sh
```

Each run writes `runs/dx<DX>_dp<DP><TAG>/res_<case>.npz`. The paper uses grids of 1, 0.5 and 0.25 mm at 2 kPa (20 ms, both cases), pressures of 0.5 to 4 kPa on 0.5 mm (40 ms), 4 kPa on 0.25 mm (20 ms), and long runs of 105 ms at 1.5 and 2 kPa on 0.5 mm and at 2 kPa on 0.25 mm. Typical cost: 3 min on 8 cores (1 mm, 20 ms), 25 min on 16 cores (0.5 mm, 20 ms), 2 h on 64 cores (0.25 mm, 20 ms), about 9 h on 64 cores (0.25 mm, 105 ms).

The job scripts load the Hábrók modules and a virtual environment at `~/venvs/whistle`; adapt those lines to your cluster.

**Stability tests (Section IV-B of the paper).** `02_simulate.py` takes `--collision bgk` (plain BGK instead of the regularized collision) and `--cs` (Smagorinsky constant; `--cs 0` switches the eddy viscosity off). The defaults are the settings of every run in the paper. The three test runs:

```bash
for v in "_bgk|--collision bgk --cs 0" "_bgk_smag|--collision bgk --cs 0.17" "_reg_nosmag|--collision reg --cs 0"; do
  sbatch -c 16 --mem=8G -t 01:30:00 --export=ALL,CASE=orig,DX=0.5,DP=2000,T=20e-3,TAG="${v%%|*}",SIMARGS="${v#*|}" code/hpc/habrok_run.sh
done
```

Without the eddy viscosity both collisions diverge within 2 ms; plain BGK with the eddy viscosity stays stable but does not produce the self-sustained tone.

**Start-up and reflection checks (Section IV-C of the paper).** `02_simulate.py` also takes `--ramp` (time over which the blowing pressure rises, default 0.5 ms) and `--sponge` (thickness of the absorbing layer, default 6 mm); `01_geometry.py` takes `--margin` (extra air on every side of the box, default 0 mm). Passed through `SIMARGS` and `GEOMARGS`:

```bash
sbatch -c 16 --mem=8G  -t 01:30:00 --export=ALL,CASE=orig,DX=0.5,DP=2000,T=20e-3,TAG=_ramp2ms,SIMARGS="--ramp 2e-3" code/hpc/habrok_run.sh
sbatch -c 32 --mem=32G -t 06:00:00 --export=ALL,CASE=orig,DX=0.5,DP=2000,T=20e-3,TAG=_bigbox,GEOMARGS="--margin 30",SIMARGS="--sponge 30" code/hpc/habrok_run.sh
python code/src/07_compare_runs.py runs/dx0.5_dp2000 runs/dx0.5_dp2000_ramp2ms runs/dx0.5_dp2000_bigbox --t0 5e-3 --t1 20e-3
```

A ramp four times slower gives the same oscillation (tone within the 67 Hz resolution of the window, level at P5 +0.3 dB). In a box 30 mm larger on every side with a 30 mm layer, the level at P5 and the flow are unchanged and the amplitude in the sphere is 9 % lower.

Instead of running the simulations, you can download the raw output of every run from Zenodo (DOI to be added) and unpack it into `runs/`.

**Figures, numbers and tables** once the runs exist:

```bash
sbatch code/hpc/habrok_figures.sh          # or: python code/paper/paper_figures.py --out results/paper_figures
```

## How to cite

Please cite the paper, and the software if you use the code. GitHub shows a "Cite this repository" button based on [`CITATION.cff`](CITATION.cff).

```bibtex
@unpublished{munozarias2026whistle,
  author = {Mu{\~n}oz-Arias, Mauricio and Barradas-Berglind, Jes{\'u}s},
  title  = {Why the {A}ztec Death Whistle Screams: A Three-Dimensional Lattice {B}oltzmann Simulation of a Skull-Whistle Replica},
  note   = {Manuscript},
  year   = {2026}
}

@software{aztec_whistle_lbm,
  author  = {Mu{\~n}oz-Arias, Mauricio and Barradas-Berglind, Jes{\'u}s},
  title   = {aztec-whistle-lbm: 3D lattice Boltzmann simulation of an Aztec skull whistle},
  version = {1.1},
  year    = {2026},
  publisher = {Zenodo},
  doi     = {10.5281/zenodo.24629264},
  url     = {https://github.com/mmunozarias/aztec-whistle-lbm}
}
```

## License

The code is released under the [MIT license](LICENSE). The figures, tables, results, recordings, video material and the raw simulation output are released under the Creative Commons Attribution 4.0 International license (CC BY 4.0), see [`LICENSE-DATA.md`](LICENSE-DATA.md). The input model is by HawkWorley under CC BY.

## Acknowledgments

We thank HawkWorley for releasing the replica model under an open license, and the Center for Information Technology of the University of Groningen for access to the Hábrók cluster. The archaeological and perceptual background draws on S. Guilliem Arroyo (INAH, 1999) and Frühholz et al., *Communications Psychology* 2, 108 (2024).

Both authors are with the Faculty of Science and Engineering, University of Groningen.
