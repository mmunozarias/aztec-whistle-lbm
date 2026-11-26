"""Step 8: roughness and tonal quality of the listener signal (probe P5).
Human screams carry fast amplitude modulation at 30-150 Hz ("roughness", Arnal et al. 2015,
Curr. Biol. 25:2051), a range normal speech avoids. For each run this script band-passes the
listener signal to 300-8000 Hz, takes its Hilbert envelope and reports:
  f0_Hz          tone (peak of the listener spectrum)
  tonal_frac     fraction of band power within +-4 % of f0 and its harmonics (1 = pure tone)
  env_depth      rms envelope fluctuation / mean envelope, all modulation rates 5-1000 Hz
  rough_depth    same, but only modulation rates 30-150 Hz (the roughness band)
  rough_frac     rough_depth^2 / env_depth^2 (share of the fluctuation that is "rough")
  fmod_peak_Hz   strongest modulation rate in 5-1000 Hz
Modulation rates below about 3/window are poorly resolved: 30 Hz needs ~100 ms of signal.
With --plot it draws, per run, the listener spectrogram and the envelope modulation spectrum.
Example:
  python code/src/08_roughness.py runs/dx0.5_dp*_long/res_orig.npz --t0 5e-3 --csv runs/roughness.csv --plot runs/roughness.png
"""
import argparse, csv, os
import numpy as np
from scipy.signal import hilbert, spectrogram
ap = argparse.ArgumentParser()
ap.add_argument('runs', nargs='+', help='res_*.npz files')
ap.add_argument('--t0', type=float, default=5e-3, help='start of the analysis window [s]')
ap.add_argument('--csv', default=None); ap.add_argument('--plot', default=None)
a = ap.parse_args()

def band(x, dt, lo, hi):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), dt); X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, len(x))

rows, curves = [], []
for fn in a.runs:
    r = np.load(fn); h = r['hist'].astype(float); dt = float(r['dt'])
    dp = float(r['dp']) if 'dp' in r.files else float('nan')
    case = os.path.basename(fn).replace('res_', '').replace('.npz', '')
    dx = float(r['y'][1] - r['y'][0])
    x = h[int(a.t0 / dt):, 4]; x = x - x.mean(); T = len(x) * dt
    xb = band(x, dt, 300, 8000)
    # tone and tonal fraction (zero-padded spectrum for a precise peak)
    n = 8 * len(xb); P = np.abs(np.fft.rfft(xb * np.hanning(len(xb)), n=n))**2; f = np.fft.rfftfreq(n, dt)
    s = (f > 300) & (f < 8000); f0 = f[s][np.argmax(P[s])]
    tone = np.zeros_like(f, bool)
    for k in range(1, int(8000 // f0) + 1):
        tone |= np.abs(f - k * f0) <= 0.04 * k * f0
    tonal_frac = P[s & tone].sum() / P[s].sum()
    # envelope and its modulation spectrum
    env = np.abs(hilbert(xb)); m = env.mean(); e = env - m
    env_depth = band(e, dt, 5, 1000).std() / m
    rough_depth = band(e, dt, 30, 150).std() / m
    E = np.abs(np.fft.rfft(e * np.hanning(len(e)), n=8 * len(e)))**2; fe = np.fft.rfftfreq(8 * len(e), dt)
    se = (fe >= 5) & (fe <= 1000); fmod = fe[se][np.argmax(E[se])]
    rows.append(dict(file=fn, case=case, dx_mm=round(dx, 3), dp_Pa=dp, window_ms=round(T * 1e3, 1),
                     mod_res_Hz=round(1 / T, 1), f0_Hz=round(f0), tonal_frac=round(tonal_frac, 3),
                     env_depth=round(env_depth, 3), rough_depth=round(rough_depth, 3),
                     rough_frac=round((rough_depth / env_depth)**2, 3) if env_depth > 0 else float('nan'),
                     fmod_peak_Hz=round(fmod)))
    curves.append((rows[-1], xb, dt, fe[se], E[se] / max(E[se].max(), 1e-30)))

hdr = ['case', 'dx_mm', 'dp_Pa', 'window_ms', 'mod_res_Hz', 'f0_Hz', 'tonal_frac', 'env_depth', 'rough_depth', 'rough_frac', 'fmod_peak_Hz']
print(('{:>11} ' * len(hdr)).format(*hdr))
for r in rows:
    print(('{:>11} ' * len(hdr)).format(*[r[k] for k in hdr]))
    if r['window_ms'] < 90:
        print(f"{'':>11} note: {r['window_ms']} ms window; rates near 30 Hz are poorly resolved (need ~90-100 ms)")
if a.csv and rows:
    with open(a.csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
if a.plot and rows:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    k = len(curves); fig, ax = plt.subplots(k, 2, figsize=(9, 2.2 * k), squeeze=False)
    for i, (r, xb, dt, fe, E) in enumerate(curves):
        nper = int(4e-3 / dt); fs_, ts_, S = spectrogram(xb, fs=1 / dt, nperseg=nper, noverlap=nper * 3 // 4)
        keep = fs_ <= 8000
        ax[i, 0].pcolormesh((ts_ + a.t0) * 1e3, fs_[keep] / 1e3, 10 * np.log10(S[keep] + 1e-12), shading='auto', cmap='magma')
        ax[i, 0].set_ylabel('kHz'); ax[i, 0].set_title(f"{r['case']}, {r['dx_mm']} mm, {r['dp_Pa']:.0f} Pa: listener spectrogram", fontsize=8)
        ax[i, 1].axvspan(30, 150, color='tab:red', alpha=0.12, lw=0, label='roughness band 30-150 Hz')
        ax[i, 1].plot(fe, E, color='k', lw=1); ax[i, 1].set_xscale('log'); ax[i, 1].set_xlim(5, 1000)
        ax[i, 1].set_title(f"envelope modulation, rough depth {r['rough_depth']}", fontsize=8)
        if i == 0: ax[i, 1].legend(fontsize=7)
    ax[-1, 0].set_xlabel('time [ms]'); ax[-1, 1].set_xlabel('modulation rate [Hz]')
    fig.tight_layout(); fig.savefig(a.plot, dpi=150); print('plot written to', a.plot)
