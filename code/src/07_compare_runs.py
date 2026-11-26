"""Step 7: compare runs (grid convergence and blowing-pressure sweep).

For every run folder (or res_*.npz file) given on the command line, prints and saves:
  dx, blowing pressure, tone f0 at the sphere (P1) and at the listener (P5) with the
  frequency resolution, mean sphere overpressure, sphere amplitude, listener level,
  duct flow rate, Bernoulli jet speed and Strouhal number St = f0 d / U_j.
With --plot it draws f0 against U_j, with a constant-Strouhal line through the runs and
the single- and two-chamber Helmholtz brackets of the paper, to show which regime the
whistle is in (resonator-controlled: f0 nearly constant; jet-controlled: f0 grows with U_j).

Examples:
  python code/src/07_compare_runs.py runs/dx0.5_dp* --t0 5e-3 --csv runs/sweep_summary.csv --plot runs/sweep_regime.png
  python code/src/07_compare_runs.py runs/dx0.5_dp2000 runs/dx0.25_dp2000 --t0 5e-3 --t1 20e-3   # same window on both grids
"""
import argparse, csv, glob, os
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

D_OPEN = 3.94e-3; RHO = 1.2
ap = argparse.ArgumentParser()
ap.add_argument('runs', nargs='+', help='run folders or res_*.npz files')
ap.add_argument('--t0', type=float, default=5e-3, help='start of the analysis window [s]')
ap.add_argument('--t1', type=float, default=None, help='end of the analysis window [s] (default: end of run); use the same t0/t1 for every grid in a convergence study')
ap.add_argument('--dp', type=float, default=None, help='blowing pressure [Pa] for files that do not store it')
ap.add_argument('--csv', default=None); ap.add_argument('--plot', default=None)
a = ap.parse_args()

files = []
for r in a.runs:
    files += sorted(glob.glob(os.path.join(r, 'res_*.npz'))) if os.path.isdir(r) else [r]
rows = []
for f in files:
    r = np.load(f); h = r['hist'].astype(float); dt = float(r['dt']); y = r['y']; dx = float(y[1] - y[0])
    dp = float(r['dp']) if 'dp' in r.files else (a.dp or 2000.0)
    i0 = int(a.t0 / dt)
    if a.t1 is not None:
        if len(h) * dt < a.t1 - dt: print(f'{f}: shorter than t1 = {a.t1*1e3:.1f} ms, skipped'); continue
        h = h[:int(round(a.t1 / dt))]
    if len(h) - i0 < int(2e-3 / dt):
        print(f'{f}: too short for t0 = {a.t0*1e3:.1f} ms, skipped'); continue
    T = (len(h) - i0) * dt
    def peak(sig):
        x = sig - sig.mean(); n = 16 * len(x)
        F = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=n)); fr = np.fft.rfftfreq(n, dt); s = (fr > 300) & (fr < 8000)
        return fr[s][np.argmax(F[s])]
    f1 = peak(h[i0:, 0]); f5 = peak(h[i0:, 4]); f0 = f5   # tone = listener peak; sphere P1 also carries slow (<500 Hz) modulation
    sos = butter(3, [0.6 * f0, 1.5 * f0], btype='band', fs=1 / dt, output='sos')   # SOS form: (b, a) form is unstable at fs ~1.2 MHz
    a1 = hilbert(sosfiltfilt(sos, h[i0:, 0])); a3 = hilbert(sosfiltfilt(sos, h[i0:, 2]))   # sphere P1 vs gap P3, at the tone
    amp1 = np.abs(a1).mean()
    ph13 = np.degrees(np.angle(np.mean(a3 * np.conj(a1))))                            # paper: -172 deg (case O)
    x = h[i0:, 4] - h[i0:, 4].mean(); X = np.fft.rfft(x); fr = np.fft.rfftfreq(len(x), dt); X[(fr < 300) | (fr > 8000)] = 0
    spl = 20 * np.log10(np.fft.irfft(X, len(x)).std() / 2e-5)
    uj = np.sqrt(2 * dp / RHO)
    rows.append(dict(file=f, dx_mm=round(dx, 3), dp_Pa=dp, window_ms=round(T * 1e3, 1), resolution_Hz=round(1 / T),
                     f0_sphere_Hz=round(f1), f0_listener_Hz=round(f5), p1_mean_Pa=round(h[i0:, 0].mean()),
                     p1_amp_Pa=round(amp1), listener_dB=round(spl, 1), Q_Lps=round(h[i0:, -1].mean(), 3),
                     phase_p3_p1_deg=round(ph13), Uj_mps=round(uj, 1), St=round(f0 * D_OPEN / uj, 3), f0_tone_Hz=round(f0)))
keys = list(rows[0].keys()) if rows else []
print(('{:>8} ' * 12).format('dx[mm]', 'dp[Pa]', 'win[ms]', 'res[Hz]', 'f0 P1', 'f0 P5', 'p1 mean', 'p1 amp', 'ph P3-P1', 'P5 [dB]', 'Q[L/s]', 'St'))
for r in rows:
    print(('{:>8} ' * 12).format(r['dx_mm'], int(r['dp_Pa']), r['window_ms'], r['resolution_Hz'], r['f0_sphere_Hz'], r['f0_listener_Hz'],
                                  r['p1_mean_Pa'], r['p1_amp_Pa'], r['phase_p3_p1_deg'], r['listener_dB'], r['Q_Lps'], r['St']))
print('Note: Q is reliable only for runs made with the current 02_simulate.py (duct cross-section monitor).')
if a.csv and rows:
    with open(a.csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
if a.plot and rows:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    U = np.array([r['Uj_mps'] for r in rows]); F0 = np.array([r['f0_tone_Hz'] for r in rows]) / 1e3
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.axhspan(1.7, 3.3, color='tab:blue', alpha=0.10, lw=0, label='two-chamber Helmholtz bracket')
    ax.axhspan(1.2, 2.3, color='tab:green', alpha=0.10, lw=0, label='single-sphere bracket')
    uu = np.linspace(0, U.max() * 1.15, 50); st = np.median(F0 * 1e3 * D_OPEN / U)
    ax.plot(uu, st * uu / D_OPEN / 1e3, 'k--', lw=1, label=f'constant St = {st:.2f}')
    ax.errorbar(U, F0, yerr=np.array([r['resolution_Hz'] for r in rows]) / 2e3, fmt='o', color='tab:orange', label='simulation')
    ax.set_xlabel('jet speed $U_j=\\sqrt{2\\Delta p/\\rho}$ [m/s]'); ax.set_ylabel('$f_0$ [kHz]'); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(a.plot, dpi=200); print('plot written to', a.plot)
