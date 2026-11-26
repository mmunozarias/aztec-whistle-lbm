"""Analyse the recordings of the 3D-printed whistle (Section "Comparison with a 3D-printed model").

Reads experiments/recordings/*.wav, detects each blow, estimates its spectrum (Welch, 2.9 Hz
resolution, 0.3 s trimmed at both ends), and writes experiments/results/numbers.json and
experiments/results/fig_printed.pdf/.png.

Usage (from the repository root):  python experiments/analyse_recordings.py
"""
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REC, OUT = HERE / "recordings", HERE / "results"
TRIM, NPERSEG = 0.3, 16384          # s trimmed at each end of a blow; Welch segment length
SIM = (1.16, 2.30)                  # kHz, LBM bands at 2 kPa, 0.25 mm grid
BLUE, RED = "#1f3fbf", "#c41e1e"


def blows(x, fs, win=0.05, drop_db=20, min_len=0.3):
    """Return (start, end) in s of segments within drop_db of the loudest 50 ms frame."""
    w = int(win * fs); n = len(x) // w
    db = np.array([20 * np.log10(np.sqrt(np.mean(x[i*w:(i+1)*w]**2)) + 1e-12) for i in range(n)])
    on = db > db.max() - drop_db
    seg, s = [], None
    for i, v in enumerate(on):
        if v and s is None: s = i
        if not v and s is not None: seg.append((s*w/fs, i*w/fs)); s = None
    if s is not None: seg.append((s*w/fs, n*w/fs))
    return [(a, b) for a, b in seg if b - a > min_len]


def analyse(path):
    x, fs = sf.read(path)
    if x.ndim > 1: x = x[:, 0]
    rows, spectra = [], []
    first = blows(x, fs)
    noise = x[: int(max(0.5, first[0][0] - 0.5) * fs)]
    for a, b in first:
        y = x[int((a + TRIM) * fs): int((b - TRIM) * fs)]
        f, P = welch(y, fs, nperseg=NPERSEG); spectra.append(P)
        m1 = (f > 1000) & (f < 1600); f1 = f[m1][np.argmax(P[m1])]
        m2 = (f > 1.85 * f1) & (f < 2.15 * f1); f2 = f[m2][np.argmax(P[m2])]
        rows.append(dict(t0=round(a, 2), dur=round(b - a, 2),
                         rms_dbfs=round(20*np.log10(np.sqrt(np.mean(y**2))), 1),
                         f_low=round(float(f1), 1), f_high=round(float(f2), 1)))
    info = dict(fs=fs, duration_s=round(len(x)/fs, 2),
                peak_dbfs=round(20*np.log10(np.max(np.abs(x))), 2),
                noise_floor_dbfs=round(20*np.log10(np.sqrt(np.mean(noise**2))), 1),
                blows=rows)
    return info, f, np.mean(spectra, 0)


def stats(rows):
    lo = np.array([r["f_low"] for r in rows]); hi = np.array([r["f_high"] for r in rows]); q = hi / lo
    return dict(n=len(rows), f_low_mean=round(lo.mean()), f_low_sd=round(lo.std(ddof=1)),
                f_high_mean=round(hi.mean()), f_high_sd=round(hi.std(ddof=1)),
                ratio_mean=round(q.mean(), 3), ratio_sd=round(q.std(ddof=1), 3))


def main():
    OUT.mkdir(exist_ok=True)
    R = {p.stem: analyse(p) for p in sorted(REC.glob("*.wav"))}
    pooled = R["w100_pos1"][0]["blows"] + R["w100_pos2"][0]["blows"]
    numbers = {k: dict(**{kk: vv for kk, vv in v[0].items() if kk != "blows"}, **stats(v[0]["blows"]), blows=v[0]["blows"])
               for k, v in R.items()}
    numbers["w100_pooled"] = stats(pooled)
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=2))

    f = R["w100_pos1"][1]
    def nd(P):
        m = (f > 600) & (f < 4000); return 10*np.log10(P / P[m].max())
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "mathtext.fontset": "stix", "axes.linewidth": 0.6})
    fig, ax = plt.subplots(2, 1, figsize=(3.45, 3.9), sharex=True)
    ax[0].plot(f/1e3, nd(R["w100_pos1"][2]), color=BLUE, lw=0.9, label=f"100 %, position 1 ($n={numbers['w100_pos1']['n']}$)")
    ax[0].plot(f/1e3, nd(R["w100_pos2"][2]), color=RED, lw=0.9, label=f"100 %, position 2 ($n={numbers['w100_pos2']['n']}$)")
    for v in SIM:
        ax[0].axvline(v, color="k", ls="--", lw=0.7); ax[0].text(v + 0.03, -43, "LBM, 2 kPa", fontsize=7)
    A = (R["w100_pos1"][2] + R["w100_pos2"][2]) / 2; B = (R["w125_pos1"][2] + R["w125_pos2"][2]) / 2
    ax[1].plot(f/1e3, nd(A), color=BLUE, lw=0.9, label="100 % model")
    ax[1].plot(f/1e3, nd(B), color=RED, lw=0.9, label="125 % model")
    ax[1].plot(0.8*f/1e3, nd(A), color="k", ls=":", lw=0.9, label="100 % shifted by 0.8")
    for a, lab in zip(ax, "ab"):
        a.set_ylim(-46, 4); a.set_ylabel("PSD [dB re max]"); a.legend(fontsize=7, frameon=False, loc="upper right")
        a.text(0.02, 0.92, f"({lab})", transform=a.transAxes); a.tick_params(direction="in", width=0.6)
    ax[1].set_xlim(0.6, 4); ax[1].set_xlabel("$f$ [kHz]")
    plt.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_printed.pdf"); fig.savefig(OUT / "fig_printed.png", dpi=200)
    print(json.dumps(numbers["w100_pooled"]), json.dumps(stats(R["w100_harder"][0]["blows"])))


if __name__ == "__main__":
    main()
