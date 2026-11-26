# Recordings of the 3D-printed model

Audio used in the section "Comparison with a 3D-printed model" of the paper.

## Set-up

- Model: the Thingiverse STL used in the simulation (thing 5148690), printed in PLA on a
  Bambu Lab X1 Carbon, 0.12 mm layers, at 100 % and at 125 % scale.
- Blowing: by mouth. **The blowing pressure was not measured.**
- Microphone: Thronmax Pulse (USB condenser, cardioid), on axis, 30 cm in front of the mouth
  of the skull. Windows audio enhancements switched off. Not calibrated: levels are relative.
- Recording: Audacity 4.0.2, WASAPI, 48 kHz, mono, exported as 24-bit PCM WAV.
- Each file starts with at least 6 s of silence (noise floor), followed by the blows.

`recordings.csv` lists what each file contains. "Position 2" is the same set-up moved to
another place in the room, to check that the spectral peaks belong to the whistle and not
to the room.

## Analysis

From the repository root:

    pip install numpy scipy matplotlib soundfile
    python experiments/analyse_recordings.py

This writes `results/numbers.json` (per-blow frequencies, means, noise floor, peak level)
and `results/fig_printed.pdf` (Fig. 11 of the paper).

## Main result

The 100 % model shows two bands at 1238 ± 15 Hz and 2460 ± 87 Hz (14 blows, two positions),
ratio 1.99 ± 0.06, matching the octave pair of the simulation (1.16 and 2.30 kHz at 2 kPa).
The 125 % model did not show the expected 20 % drop in pitch; see the paper for discussion.
This is a qualitative comparison; a controlled study with measured blowing pressure is
future work.
