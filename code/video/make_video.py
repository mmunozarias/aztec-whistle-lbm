"""Explainer video of the Aztec skull-whistle simulations, made from the simulation output.

Run from the repository root (on Hábrók through code/hpc/habrok_video.sh):
  python code/video/make_video.py --runs runs --out results/video --stl code/input/_BothDeathwhistlefile98.stl
Quick test at low resolution:  add  --scale 0.5 --fps 10
Writes to --out:
  whistle_video.mp4   the video with sound (1920x1080, 30 fps)
  thumbnail.jpg       1280x720 cover for YouTube
  chapters.txt        chapter times for the YouTube description
  preview.jpg         one still per scene, to check the video without downloading it

Runs used (under --runs): dx0.25_dp2000_long (main), dx0.25_dp2000, dx1/0.5/0.25 at 2 kPa (convergence),
dx0.5_dp{500,1000,1500,2000} (pressure sweep). Numbers come from results/paper_figures/numbers.json when present.
Style: University of Groningen look, after the RING seminar slides: white background, RUG red (220, 0, 45) with dark red
and blue, Computer Modern Sans (Latin Modern Sans when found), red bars at top and bottom, the logo top right,
gray tiles with red headers and thin block diagrams with red arrows. No orange or yellow except in the spectrogram.
"""
import argparse, json, os, subprocess, textwrap, time, traceback
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, PowerNorm, Normalize
from scipy.signal import welch, butter, sosfiltfilt, hilbert, spectrogram
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument('--runs', default='runs'); ap.add_argument('--out', default='results/video')
ap.add_argument('--stl', default='code/input/_BothDeathwhistlefile98.stl')
ap.add_argument('--numbers', default='results/paper_figures/numbers.json')
ap.add_argument('--main', default='dx0.25_dp2000_long')
ap.add_argument('--scale', type=float, default=1.0); ap.add_argument('--fps', type=int, default=30)
ap.add_argument('--ffmpeg', default='ffmpeg')
ap.add_argument('--recordings', default='experiments', help='folder with recordings/ and results/numbers.json of the printed model')
ap.add_argument('--no-music', action='store_true', help='leave out the background music')
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)

# ---------------------------------------------------------------- style
W, H = int(round(1920 * A.scale / 2) * 2), int(round(1080 * A.scale / 2) * 2); FPS = A.fps; SR = 48000
BG, FG, MUTED, TILE = '#ffffff', '#1a1a1a', '#5f6670', '#f2f2f2'           # white page, near-black text, RUG grey tiles
RED, DRED = '#dc002d', '#8c001e'                                             # RUG red and dark red (seminar theme)
BLUE, LBLUE, DBLUE = '#005496', '#8fb4d6', '#002f57'
SOLIDC = '#9a9ea6'
SPEED = LinearSegmentedColormap.from_list('sp', [BG, LBLUE, BLUE, DBLUE, '#001222'])
PRESS = LinearSegmentedColormap.from_list('pr', [DBLUE, BLUE, LBLUE, BG, '#f29aab', RED, DRED])
LEVEL = LinearSegmentedColormap.from_list('lv', [BG, '#f6b3c0', RED, DRED])

# fonts: Latin Modern Sans (as in the slides) when its OpenType files are found, else matplotlib's Computer Modern Sans
import glob
from matplotlib import font_manager as fm, patheffects as pe
def _find_lm():
    roots = [os.environ.get('LMSANS_DIR', ''), os.path.expanduser('~/.local/share/fonts'), '/usr/share/fonts', '/usr/share/texmf',
             '/usr/share/texlive', '/usr/local/texlive', os.environ.get('EBROOTTEXLIVE', '')]
    found = {}
    for r in filter(None, roots):
        for f in glob.glob(os.path.join(r, '**', 'lmsans10-*.otf'), recursive=True):
            found.setdefault(os.path.basename(f), f)
    return found.get('lmsans10-regular.otf'), found.get('lmsans10-bold.otf')
_lr, _lb = _find_lm()
if _lr and _lb:
    fm.fontManager.addfont(_lr); fm.fontManager.addfont(_lb); FONT = fm.FontProperties(fname=_lr).get_name(); REAL_BOLD = True
else:
    FONT = 'cmss10'; REAL_BOLD = False                                       # ships with matplotlib, regular weight only
print('font:', FONT, '(real bold)' if REAL_BOLD else '(bold drawn with a thin outline)', flush=True)
plt.rcParams.update({'font.family': [FONT, 'DejaVu Sans'], 'text.color': FG, 'axes.unicode_minus': False,
                     'mathtext.fontset': 'cm', 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.edgecolor': '#9a9ea6', 'axes.facecolor': BG, 'figure.facecolor': BG, 'savefig.facecolor': BG,
                     'font.size': 18})
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
def _img(name):
    try:
        from PIL import Image; return np.asarray(Image.open(os.path.join(ASSETS, name)).convert('RGBA'))
    except Exception as e: print('  logo missing:', e); return None
LOGO, LOGO_FULL = _img('rug_logo.png'), _img('rug_logo_full.png')
BGRGB = np.array([int(BG[i:i + 2], 16) for i in (1, 3, 5)], float)

def P(*p): return os.path.join(A.runs, *p)
def exists(run, case='orig'): return os.path.exists(P(run, f'res_{case}.npz'))
def load(run, case='orig'): return np.load(P(run, f'res_{case}.npz'))
NUM = json.load(open(A.numbers)) if os.path.exists(A.numbers) else {}
def num(path, default):
    d = NUM
    try:
        for k in path: d = d[k]
        return d
    except Exception: return default

def tone(x, dt, lo=300, hi=8000):
    x = x - x.mean(); n = 8 * len(x); F = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=n)); f = np.fft.rfftfreq(n, dt)
    s = (f > lo) & (f < hi); return float(f[s][np.argmax(F[s])])
def band(x, dt, lo, hi):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), dt); X[(f < lo) | (f > hi)] = 0; return np.fft.irfft(X, len(x))
def fmt_k(f): return f'{f/1e3:.1f}'.rstrip('0').rstrip('.') + ' kHz' if f else 'none'

# ---------------------------------------------------------------- drawing helpers
def new_fig():
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100 * A.scale); fig.patch.set_facecolor(BG); return fig
def grab(fig):
    fig.canvas.draw(); img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    if img.shape[:2] != (H, W):
        from PIL import Image
        img = np.asarray(Image.fromarray(img).resize((W, H)))
    return img.copy()
def LH(size, nlines=1): return size * 1.39 * 1.35 * nlines   # pixel height of nlines of text
def nl(s, wrap): return len(textwrap.wrap(s, wrap)) if wrap else s.count('\n') + 1
def T(fig, x, y, s, size=30, color=FG, weight='normal', wrap=None, ha='left', va='top', alpha=1.0, **kw):
    """Text at pixel position (x, y) of a 1920x1080 frame, measured from the top left."""
    if wrap: s = '\n'.join(textwrap.fill(p, wrap) for p in s.split('\n'))
    if weight == 'bold' and not REAL_BOLD:
        kw.setdefault('path_effects', [pe.withStroke(linewidth=max(0.6, size * 0.045) * A.scale, foreground=color)]); weight = 'normal'
    return fig.text(x / 1920, 1 - y / 1080, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, alpha=alpha,
                    linespacing=1.35, **kw)
def chars(width, size, weight='normal'):
    """Characters per line that fit in a width of pixels (DejaVu Sans, about 0.56 em per character)."""
    return max(8, int(width / (size * 1.39 * (0.58 if weight == 'bold' else 0.52))))
def TB(fig, x, y, s, size=30, color=FG, weight='normal', width=800, gap=0.6):
    """Wrapped text block in a given pixel width; returns the y position just below it (plus a gap of gap lines)."""
    c = chars(width, size, weight); T(fig, x, y, s, size, color, weight, wrap=c); return y + LH(size, nl(s, c)) + LH(size, gap)
def AX(fig, x, y, w, h, **kw):
    """Axes at pixel box (x, y, w, h) from the top left."""
    return fig.add_axes([x / 1920, 1 - (y + h) / 1080, w / 1920, h / 1080], **kw)
FOOT_L = 'Why the Aztec death whistle screams'
FOOT_R = 'M. Muñoz-Arias and J. Barradas-Berglind, University of Groningen'
def rect(fig, x, y, w, h, color, **kw):
    """Filled rectangle at a pixel box of the 1920x1080 frame."""
    fig.add_artist(matplotlib.patches.Rectangle((x / 1920, 1 - (y + h) / 1080), w / 1920, h / 1080, transform=fig.transFigure,
                                                facecolor=color, edgecolor='none', zorder=kw.pop('zorder', -1), **kw))
def logo(fig, x, y, h, full=False):
    im = LOGO_FULL if full else LOGO
    if im is None: return
    w = h * im.shape[1] / im.shape[0]; ax = AX(fig, x, y, w, h); ax.imshow(im, interpolation='antialiased'); ax.axis('off')
def header(fig, n, name):
    """Slide frame of the seminar theme: thin red bar on top, title in RUG red, logo top right, red footer bar."""
    rect(fig, 0, 0, 1920, 12, RED)
    T(fig, 80, 42, name, 44, RED)
    logo(fig, 1840 - 62 * 760 / 206, 34, 62)
    rect(fig, 0, 1042, 1920, 38, RED)
    T(fig, 80, 1061, FOOT_L, 17, '#ffffff', va='center'); T(fig, 1840, 1061, FOOT_R, 17, '#ffffff', va='center', ha='right')
def tile(fig, x, y, w, title, body, size=26, red=True):
    """Seminar block: coloured header strip, grey body. Returns the y just below it."""
    hh = LH(size, 1) + 18; c = chars(w - 40, size); nb = nl(body, c); bh = LH(size, nb) + 30
    rect(fig, x, y, w, hh, RED if red else DRED); T(fig, x + 20, y + 9, title, size, '#ffffff', 'bold')
    rect(fig, x, y + hh, w, bh, TILE); T(fig, x + 20, y + hh + 14, body, size, FG, wrap=c)
    return y + hh + bh
def bullet(fig, x, y, s, size=30, color=FG, weight='normal', width=900, gap=0.6, mark=None):
    """Item with a red disc (or a numbered red disc), as in the seminar's itemize."""
    r = size * 0.42
    fig.add_artist(matplotlib.patches.Ellipse(((x + r) / 1920, 1 - (y + LH(size) * 0.42) / 1080), 2 * r / 1920, 2 * r / 1080,
                                              transform=fig.transFigure, facecolor=RED, edgecolor='none'))
    if mark is not None: T(fig, x + r, y + LH(size) * 0.42, str(mark), size * 0.62, '#ffffff', 'bold', ha='center', va='center')
    return TB(fig, x + 2.6 * r, y, s, size, color, weight, width=width - 2.6 * r, gap=gap)
def plane_ax(fig, box, ylim, zlim):
    ax = AX(fig, *box); ax.set_xlim(*ylim); ax.set_ylim(*zlim); ax.set_aspect('equal'); ax.axis('off'); return ax
def solid_img(S):
    g = np.zeros(S.shape + (4,)); g[S] = matplotlib.colors.to_rgba(SOLIDC); return g.transpose(1, 0, 2)
def ease(t): t = np.clip(t, 0, 1); return t * t * (3 - 2 * t)
def fade(img, a): return (BGRGB + (img.astype(float) - BGRGB) * a).astype(np.uint8)
def mix(a, b, t): return (a.astype(float) * (1 - t) + b.astype(float) * t).astype(np.uint8)
def box_fit(ylim, zlim, x0, y0, wmax, hmax):
    """Largest pixel box with the aspect of the plane region that fits in wmax x hmax."""
    asp = (ylim[1] - ylim[0]) / (zlim[1] - zlim[0]); w = min(wmax, hmax * asp); h = w / asp
    return (x0 + (wmax - w) / 2, y0 + (hmax - h) / 2, w, h)

# ---------------------------------------------------------------- video and sound
class Movie:
    def __init__(self, path):
        self.path = path; self.n = 0; self.audio = np.zeros(SR * 400); self.chapters = []; self.stills = []
        self.p = subprocess.Popen([A.ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                                   '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                                   '-pix_fmt', 'yuv420p', path], stdin=subprocess.PIPE)
    @property
    def t(self): return self.n / FPS
    def add(self, img): self.p.stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes()); self.n += 1
    def chapter(self, name): self.chapters.append((self.t, name))
    def still(self, img, name): self.stills.append((img, name))
    def sound(self, x, t0, gain=1.0, ramp=0.03):
        i0 = int(t0 * SR); x = np.asarray(x, float) * gain; r = int(ramp * SR)
        if r and len(x) > 2 * r: w = np.ones(len(x)); w[:r] = np.linspace(0, 1, r); w[-r:] = np.linspace(1, 0, r); x = x * w
        need = i0 + len(x)
        if need > len(self.audio): self.audio = np.concatenate([self.audio, np.zeros(need - len(self.audio))])
        self.audio[i0:need] += x
    def close(self):
        self.p.stdin.close(); self.p.wait(); return self.n / FPS

def scene(mv, dur, states, times=None, xfade=0.5, fin=0.5, fout=0.5, still_at=-1, name=''):
    """Show a list of pre-rendered frames that build up one after the other, with cross-fades."""
    n = len(states); times = times if times is not None else [dur * 0.85 * i / max(n, 1) for i in range(n)]
    N = int(round(dur * FPS))
    for k in range(N):
        t = k / FPS; i = max(j for j in range(n) if times[j] <= t); img = states[i]
        if i > 0 and t - times[i] < xfade: img = mix(states[i - 1], states[i], ease((t - times[i]) / xfade))
        a = min(ease(t / fin) if fin else 1, ease((dur - t) / fout) if fout else 1)
        mv.add(fade(img, a) if a < 1 else img)
    mv.still(states[still_at], name)

def fader(dur, t, fin=0.5, fout=0.5): return min(ease(t / fin) if fin else 1, ease((dur - t) / fout) if fout else 1)

def listener_audio(r, t0, t1=None, lo=150, hi=12000):
    """Listener pressure (P5) in Pa, band-passed and resampled to 48 kHz."""
    h = r['hist']; dt = float(r['dt']); i0 = int(t0 / dt); i1 = len(h) if t1 is None else min(len(h), int(t1 / dt))
    x = h[i0:i1, 4].astype(float); x = band(x - x.mean(), dt, lo, hi)
    ts = np.arange(len(x)) * dt; to = np.arange(0, ts[-1], 1 / SR); return np.interp(to, ts, x)
def loop(x, dur, xf=0.01):
    """Repeat a clip to the given duration with short cross-fades, so the slow wobble of the sound is kept."""
    n = int(dur * SR); f = int(xf * SR); seg = len(x)
    if seg <= 2 * f: return np.resize(x, n)
    w = np.ones(seg); w[:f] = np.sqrt(np.linspace(0, 1, f)); w[-f:] = np.sqrt(np.linspace(1, 0, f)); out = np.zeros(n + seg); pos = 0
    while pos < n: out[pos:pos + seg] += x * w; pos += seg - f
    return out[:n]

# ---------------------------------------------------------------- data
t_start = time.time()
print('loading', A.main, flush=True)
R = load(A.main); h = R['hist'].astype(float); dt = float(R['dt']); se = int(R['save_every'])
uf = float(R['ufac']); pf = float(R['pfac']); y, z = R['y'], R['z']; SOL = R['solid_x']; dxm = float(y[1] - y[0])
EXT = [y[0] - dxm / 2, y[-1] + dxm / 2, z[0] - dxm / 2, z[-1] + dxm / 2]
SX = R['sx']                                  # (frames, 4, ny, nz); loaded once, it is large
TF = np.arange(len(SX)) * se * dt             # time of each frame
T0 = 20e-3; K0 = int(np.ceil(T0 / (se * dt)))  # settled part of the run
print(f'  {SX.shape} frames, {SX.nbytes/1e9:.1f} GB, frame step {se*dt*1e6:.1f} us', flush=True)
F0 = tone(h[int(T0 / dt):, 4], dt, 1700, 3300)
ncell = int(len(R['x']) * len(y) * len(z))
Uj = float(np.sqrt(2 * float(R['dp']) / 1.2))

def pmean_and_rms(k0, k1, chunk=100):
    s = np.zeros(SX.shape[2:]); s2 = np.zeros(SX.shape[2:]); n = 0
    for k in range(k0, k1, chunk):
        p = SX[k:min(k + chunk, k1), 3].astype(np.float64) * pf; s += p.sum(0); s2 += (p ** 2).sum(0); n += len(p)
    m = s / n; return m, np.sqrt(np.maximum(s2 / n - m ** 2, 0))
PM, PRMS = pmean_and_rms(K0, len(SX))

def phase_average(nb=16, chunk=100):
    """Phase-averaged field (nb bins) over the settled part, phase from the band-passed sphere pressure."""
    sos = butter(3, [0.8 * F0, 1.2 * F0], btype='band', fs=1 / dt, output='sos')
    phi = np.angle(hilbert(sosfiltfilt(sos, h[:, 0] - h[int(T0 / dt):, 0].mean())))
    ks = np.arange(K0, len(SX)); steps = np.minimum(ks * se, len(h) - 1)
    b = np.clip(((phi[steps] + np.pi) / (2 * np.pi) * nb).astype(int), 0, nb - 1)
    acc = np.zeros((nb, 4) + SX.shape[2:]); cnt = np.zeros(nb)
    for j in range(0, len(ks), chunk):
        a = SX[ks[j]:ks[j] + len(ks[j:j + chunk])].astype(np.float64); bb = b[j:j + chunk]
        for q in range(nb):
            s = bb == q
            if s.any(): acc[q] += a[s].sum(0); cnt[q] += s.sum()
    avg = acc / np.maximum(cnt, 1)[:, None, None, None]
    ph = -np.pi + (np.arange(nb) + 0.5) * 2 * np.pi / nb
    return ph, avg
PH, AVG = phase_average()
# Fourier fit (up to the 3rd harmonic) of the phase-averaged pressure, for a smooth loop
PPH = AVG[:, 3] * pf - PM
Bm = np.column_stack([np.ones(len(PH))] + [f(k * PH) for k in (1, 2, 3) for f in (np.cos, np.sin)])
COEF = np.linalg.lstsq(Bm, PPH.reshape(len(PH), -1), rcond=None)[0]
def phase_field(phi):
    b = np.array([1] + [f(k * phi) for k in (1, 2, 3) for f in (np.cos, np.sin)]); return (b @ COEF).reshape(PPH.shape[1:])
print(f'  analysis done, f0 = {F0:.0f} Hz, {time.time()-t_start:.0f} s', flush=True)

# sounds (Pa); one scale for the whole video so that levels can be compared
SND_MAIN = listener_audio(R, T0)
GAIN = 0.12 / max(SND_MAIN.std(), 1e-9)
SWEEP = {}
for p in (500, 1000, 1500, 2000):
    run = f'dx0.5_dp{p}'
    if exists(run): SWEEP[p] = listener_audio(load(run), 5e-3)
SWEEP_GAIN = 0.12 / SWEEP[2000].std() if 2000 in SWEEP else GAIN

# recordings of the 3D-printed model (experiments/): blows cut out with the times found by analyse_recordings.py
def read_wav(path):
    try:
        import soundfile as sf; x, fs = sf.read(path, always_2d=True); x = x.mean(1)
    except ImportError:
        from scipy.io import wavfile
        fs, x = wavfile.read(path); full = {'int16': 2 ** 15, 'int32': 2 ** 31}.get(str(x.dtype), 1.0)   # 24-bit is read as int32
        x = x.astype(float) / full
        if x.ndim > 1: x = x.mean(1)
    if fs != SR: x = np.interp(np.arange(0, len(x) / fs, 1 / SR), np.arange(len(x)) / fs, x)
    return x
REC_NUM = {}
_rn = os.path.join(A.recordings, 'results', 'numbers.json')
if os.path.exists(_rn): REC_NUM = json.load(open(_rn))
def rec_blows(name, keep=0.8):
    """Blows of one recording, each trimmed to its steady middle part (keep = fraction of the blow), high-passed at 150 Hz."""
    path = os.path.join(A.recordings, 'recordings', name + '.wav')
    if name not in REC_NUM or not os.path.exists(path): return []
    x = read_wav(path); sos = butter(4, 150, 'high', fs=SR, output='sos'); x = sosfiltfilt(sos, x); out = []
    for b in REC_NUM[name]['blows']:
        m = (1 - keep) / 2 * b['dur']; i0 = int((b['t0'] + m) * SR); i1 = int((b['t0'] + b['dur'] - m) * SR); out.append(x[i0:i1])
    return out
REC100 = rec_blows('w100_pos1') + rec_blows('w100_pos2')
REC_GAIN = 0.12 / np.median([b.std() for b in REC100]) if REC100 else 1.0

# ---------------------------------------------------------------- background music
def make_music(total, seed=7):
    """A quiet ambient track synthesized here from sine tones (no samples, no third-party audio, so no rights issues):
    slow pads on Am, F, C, G with a soft plucked arpeggio and a simple reverb. Kept below 1.3 kHz, under the whistle."""
    rng = np.random.default_rng(seed); n = int(total * SR); t = np.arange(n) / SR; out = np.zeros(n)
    hz = lambda m: 440.0 * 2 ** ((m - 69) / 12)
    chords = [[45, 57, 60, 64], [41, 57, 60, 65], [48, 55, 60, 64], [43, 55, 59, 62]]   # Am, F, C, G
    seg = 8.0; att = 2.5
    for k in range(int(np.ceil(total / seg)) + 1):
        c = chords[k % 4]; t0 = k * seg - att; i0 = max(0, int(t0 * SR)); i1 = min(n, int((t0 + seg + 2 * att) * SR))
        if i1 <= i0: continue
        tt = t[i0:i1] - t0; L = seg + 2 * att
        env = np.minimum(1, np.minimum(tt / att, (L - tt) / att)) ** 2
        for j, m in enumerate(c):
            f = hz(m) if j else hz(m) / 2                                   # the root an octave lower as bass
            for d in (-0.15, 0.15):                                         # two slightly detuned voices
                out[i0:i1] += env * (0.5 if j else 0.8) * np.sin(2 * np.pi * (f + d) * tt + rng.uniform(0, 2 * np.pi))
        for q in range(int(seg / 0.5)):                                     # arpeggio, one note every half second
            if rng.random() < 0.35: continue
            ts = k * seg + q * 0.5
            if ts >= total: break
            f = hz(c[1 + (q % 3)] + 12 * (q // 6 % 2)); j0 = int(ts * SR); j1 = min(n, j0 + int(1.6 * SR)); tt = t[j0:j1] - ts
            out[j0:j1] += 0.6 * np.exp(-tt / 0.45) * (np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(4 * np.pi * f * tt)) * np.minimum(1, tt / 0.005)
    from scipy.signal import fftconvolve
    ir_t = np.arange(int(1.8 * SR)) / SR; ir = rng.standard_normal(len(ir_t)) * np.exp(-ir_t / 0.45); ir[0] = 6.0
    out = fftconvolve(out, ir / np.abs(ir).sum() * 8, mode='full')[:n]
    out = sosfiltfilt(butter(2, 1300, 'low', fs=SR, output='sos'), out)
    fi = np.minimum(1, np.minimum(t / 3.0, (total - t) / 4.0)); out *= np.clip(fi, 0, 1)
    return out / (out.std() + 1e-12)
def duck(whistle, music, level=0.025, floor=0.12):
    """Music at a low level that drops to `floor` of that level whenever a whistle sound plays."""
    e = ndi.uniform_filter1d(np.abs(whistle), int(0.25 * SR)); d = np.clip(e / 0.03, 0, 1)
    d = ndi.uniform_filter1d(ndi.maximum_filter1d(d, int(0.6 * SR)), int(0.6 * SR))
    return music * level * (1 - (1 - floor) * d)

# ---------------------------------------------------------------- the skull, rendered once
def skull_render(w, h_):
    try:
        import trimesh
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        m = trimesh.load(A.stl, force='mesh')
        try: m = m.simplify_quadric_decimation(face_count=120000)
        except Exception: pass
        fig = plt.figure(figsize=(w / 100, h_ / 100), dpi=100 * A.scale); fig.patch.set_facecolor(BG)
        ax = fig.add_axes([0, 0, 1, 1], projection='3d'); ax.set_facecolor(BG)
        tri = m.triangles; nrm = m.face_normals; L = np.array([0.35, -0.55, 0.75]); L /= np.linalg.norm(L)
        sh = 0.18 + 0.78 * np.clip(nrm @ L, 0, 1); base = np.array([0.86, 0.82, 0.74])
        fc = np.column_stack([sh[:, None] * base, np.ones(len(sh))])
        ax.add_collection3d(Poly3DCollection(tri, facecolors=fc, edgecolor='none'))
        c = m.bounds.mean(0); rad = m.extents.max() / 2 * 0.55
        ax.set_xlim(c[0] - rad, c[0] + rad); ax.set_ylim(c[1] - rad, c[1] + rad); ax.set_zlim(c[2] - rad, c[2] + rad)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(18, -58); ax.set_axis_off()
        img = grab_any(fig); plt.close(fig); return img
    except Exception as e:
        print('  skull render skipped:', e); return None
def grab_any(fig):
    fig.canvas.draw(); return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
SKULL = skull_render(900, 900)
def put_skull(fig, box, alpha=1.0):
    if SKULL is None: return
    ax = AX(fig, *box); ax.imshow(SKULL, alpha=alpha); ax.axis('off')

# ---------------------------------------------------------------- scenes
F0k = f'{F0/1e3:.1f}'
f0_pair = num(['pair_orig', 'f0_listener_Hz'], F0)

def s01_cover(mv):
    mv.chapter('Why the death whistle screams')
    def draw(n):
        fig = new_fig(); put_skull(fig, (1030, 110, 860, 860))
        logo(fig, 90, 70, 90, full=True); rect(fig, 0, 1062, 1920, 18, RED)
        if n >= 1:
            T(fig, 90, 300, 'Why the Aztec Death\nWhistle Screams', 72, RED, 'bold', va='top')
        if n >= 2:
            T(fig, 92, 560, 'Inside the airflow of a skull whistle,\nsimulated on a supercomputer', 34, DRED)
            T(fig, 92, 780, 'Mauricio Muñoz-Arias and Jesús Barradas-Berglind', 28, FG, 'bold')
            T(fig, 92, 828, 'Faculty of Science and Engineering, University of Groningen', 22, MUTED)
        img = grab(fig); plt.close(fig); return img
    st = [draw(n) for n in range(3)]; scene(mv, 6.5, st, [0, 0.6, 2.0], name='Cover')
    return st[-1]

def s02_discovery(mv):
    mv.chapter('The discovery')
    lines = [('Tlatelolco, Mexico City, 1987 to 1989', 26, RED, 'bold'),
             ('At the temple of Ehecatl, god of the wind, archaeologists found a sacrificed young man '
              'holding a small clay skull in each hand.', 40, FG, 'normal'),
             ('They were whistles. Blown hard, they shriek like a human scream.', 40, FG, 'normal'),
             ('How does air make that sound?', 46, RED, 'bold')]
    def draw(n):
        fig = new_fig(); header(fig, 1, 'The discovery'); put_skull(fig, (1180, 190, 700, 700), 0.35); yy = 220
        for i, (s, sz, c, wgt) in enumerate(lines[:n]): yy = TB(fig, 90, yy, s, sz, c, wgt, width=1000, gap=0.7)
        img = grab(fig); plt.close(fig); return img
    scene(mv, 14, [draw(n) for n in range(5)], [0, 0.5, 2.0, 6.5, 10.0], name='Discovery')

def s03_sound(mv):
    mv.chapter('The sound')
    dur = 14; x = SND_MAIN; t0 = mv.t
    mv.sound(loop(x, dur - 1.5), t0 + 1.0, GAIN)
    fw, Pw = welch(x, fs=SR, nperseg=min(len(x), int(0.02 * SR))); L = 10 * np.log10(Pw * (fw[1] - fw[0]) / (2e-5) ** 2 + 1e-12)
    s = (fw > 900) & (fw < 1500); f_low = float(fw[s][np.argmax(L[s])]) if s.any() else F0 / 2
    fig = new_fig(); header(fig, 2, 'The sound')
    T(fig, 90, 150, 'Listen', 90, FG, 'bold')
    yy = TB(fig, 90, 330, 'This is the sound computed by the simulation, just in front of the mouth.', 32, MUTED, width=700)
    TB(fig, 90, yy, f'A tone near {F0k} kHz, wrapped in noise, with a second band one octave lower.', 32, FG, 'bold', width=700)
    T(fig, 90, 990, 'Turn the sound on.', 24, MUTED)
    a1 = AX(fig, 900, 150, 930, 330); a2 = AX(fig, 900, 600, 930, 360)
    win = 0.004; nw = int(win * SR); tt = np.arange(nw) / SR * 1e3; amp = np.percentile(np.abs(x), 99.5)
    ln, = a1.plot(tt, np.zeros(nw), color=BLUE, lw=1.6); a1.set_ylim(-amp, amp); a1.set_xlim(0, win * 1e3)
    a1.set_xlabel('time [ms]'); a1.set_ylabel('pressure'); a1.set_yticks([]); [a1.spines[k].set_visible(False) for k in ('top', 'right')]
    a2.plot(fw / 1e3, L, color=FG, lw=1.4); a2.set_xlim(0, 8); a2.set_ylim(L[(fw > 300) & (fw < 8000)].max() - 45, L.max() + 6)
    a2.set_xlabel('frequency [kHz]'); a2.set_ylabel('level [dB]'); [a2.spines[k].set_visible(False) for k in ('top', 'right')]
    yt = a2.get_ylim()[1]
    a2.axvline(F0 / 1e3, color=RED, ls='--', lw=1.2); a2.text(F0 / 1e3 + 0.12, yt - 2, f'tone {F0k} kHz', color=RED, fontsize=20, va='top')
    a2.axvline(f_low / 1e3, color=BLUE, ls='--', lw=1.2); a2.text(f_low / 1e3 - 0.12, yt - 2, f'octave below\n{f_low/1e3:.2f} kHz', color=BLUE, fontsize=20, va='top', ha='right')
    N = int(dur * FPS)
    for k in range(N):
        t = k / FPS; i = int(max(t - 1.0, 0) * SR * 0.02) % max(len(x) - nw, 1)   # window drifts slowly through the clip
        ln.set_ydata(x[i:i + nw]); img = grab(fig); a = fader(dur, t); mv.add(fade(img, a) if a < 1 else img)
        if k == N // 2: mv.still(img, 'Sound')
    plt.close(fig)

LAB = [((46, 21), 'Your breath enters here'), ((22, 19.5), 'A narrowing duct speeds it up'),
       ((12.5, 17.2), 'The jet hits the lip of a small opening'), ((0, 22), 'A hollow sphere: an air spring'),
       ((-1, 34.8), 'The gap around it: a second air spring'), ((-22, 26), 'Sound leaves through the mouth')]
def s04_instrument(mv):
    mv.chapter('The instrument')
    ylim, zlim = (-52, 56), (-6, 64); box = box_fit(ylim, zlim, 60, 130, 1150, 880)
    def draw(n):
        fig = new_fig(); header(fig, 3, 'The instrument'); ax = plane_ax(fig, box, ylim, zlim)
        ax.imshow(solid_img(SOL), origin='lower', extent=EXT, interpolation='nearest')
        yy = TB(fig, 1260, 140, 'A cut through the middle of the skull', 32, FG, 'bold', width=600, gap=1.0)
        for i, ((py, pz), s) in enumerate(LAB[:n]):
            c = RED if i == 2 else BLUE
            ax.plot(py, pz, 'o', ms=26, mfc=BG, mec=c, mew=2.5); ax.text(py, pz, str(i + 1), ha='center', va='center', fontsize=17, color=FG, fontweight='bold')
            T(fig, 1260, yy, f'{i+1}', 28, c, 'bold'); yy = TB(fig, 1310, yy, s, 28, FG, width=560, gap=0.5)
        img = grab(fig); plt.close(fig); return img
    st = [draw(n) for n in range(len(LAB) + 1)]
    scene(mv, 16, st, [0] + [1.2 + 2.0 * i for i in range(len(LAB))], name='Instrument')

def s04b_diagram(mv):
    """How it works, as a block diagram in the seminar style: outlined boxes, red arrows, built up one step at a time."""
    mv.chapter('How it works')
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    B = {'breath': (90, 470, 300, 'Breath', '1 to 2 kPa'), 'jet': (480, 470, 300, 'Jet', f'about {Uj:.0f} m/s'),
         'lip': (870, 470, 300, 'Lip of the opening', 'jet flips in and out'),
         'sphere': (1290, 290, 330, 'Inner sphere', 'air spring'), 'gap': (1290, 650, 330, 'Gap around it', 'second air spring'),
         'sound': (1290, 870, 330, 'Sound', f'tone near {F0k} kHz')}
    BH = 120
    def box(fig, k, on):
        x, y, w, t1, t2 = B[k]; ec = RED if on else '#d0d3d8'
        fig.add_artist(FancyBboxPatch((x / 1920, 1 - (y + BH) / 1080), w / 1920, BH / 1080, boxstyle='round,pad=0,rounding_size=0.006',
                                      transform=fig.transFigure, facecolor='#ffffff' if k != 'lip' else TILE, edgecolor=ec, lw=2.4))
        T(fig, x + w / 2, y + 22, t1, 28, FG if on else '#b8bcc3', 'bold', ha='center')
        T(fig, x + w / 2, y + 70, t2, 22, MUTED if on else '#c8ccd2', ha='center')
    def arrow(fig, p, q, color=RED, rad=0.0, label=None, lpos=None):
        fig.add_artist(FancyArrowPatch((p[0] / 1920, 1 - p[1] / 1080), (q[0] / 1920, 1 - q[1] / 1080), transform=fig.transFigure,
                                       arrowstyle='-|>', mutation_scale=26 * A.scale, lw=2.4, color=color, connectionstyle=f'arc3,rad={rad}'))
        if label: T(fig, lpos[0], lpos[1], label, 21, color, ha='center')
    steps = ['breath', 'jet', 'lip', 'springs', 'feedback', 'sound']
    text = ['You blow: a small overpressure in the mouth.', 'The narrowing duct turns it into a fast jet.',
            'The jet grazes the lip of a small opening in the hidden sphere.',
            'The sphere and the gap around it are two coupled air springs.',
            'Their pressure pushes the jet in and out: a loop that sustains itself.',
            'Part of the oscillation leaves through the mouth as sound.']
    def draw(n):
        fig = new_fig(); header(fig, 0, 'How it works: a loop of jet and air springs')
        cy = 470 + BH / 2
        box(fig, 'breath', n >= 1); box(fig, 'jet', n >= 2); box(fig, 'lip', n >= 3)
        box(fig, 'sphere', n >= 4); box(fig, 'gap', n >= 4); box(fig, 'sound', n >= 6)
        if n >= 2: arrow(fig, (390, cy), (480, cy))
        if n >= 3: arrow(fig, (780, cy), (870, cy))
        if n >= 4:
            arrow(fig, (1170, cy - 30), (1290, 290 + BH / 2), BLUE); arrow(fig, (1170, cy + 30), (1290, 650 + BH / 2), BLUE)
            arrow(fig, (1455, 290 + BH), (1455, 650), DRED); arrow(fig, (1455, 650), (1455, 290 + BH), DRED)
            T(fig, 1475, 545, 'in anti-phase', 21, DRED)
        if n >= 5: arrow(fig, (1290, 290 + 25), (1020, 470), RED, rad=0.35, label='pressure feeds back', lpos=(1130, 300))
        if n >= 6: arrow(fig, (1455, 650 + BH), (1455, 870), RED)
        if n: bullet(fig, 90, 150, text[n - 1], 30, FG, width=1750)
        img = grab(fig); plt.close(fig); return img
    st = [draw(n) for n in range(len(steps) + 1)]
    scene(mv, 19, st, [0] + [0.8 + 2.6 * i for i in range(len(steps))], name='Diagram')

def s05_method(mv):
    mv.chapter('How we simulated it')
    rows = num(['convergence', 'orig'], [])
    grid = [(1.0, None), (0.5, None), (0.25, None)]
    for r_ in rows:
        for i, (g, _) in enumerate(grid):
            if abs(r_.get('dx_mm', 0) - g) < 1e-6: grid[i] = (g, r_.get('f0_listener_Hz'))
    if grid[2][1] is None: grid[2] = (0.25, f0_pair)
    grid[0] = (1.0, None)                                 # the 1 mm grid does not whistle
    lines = ['We computed the flow of air inside and around a 3D-printable replica.',
             'The lattice Boltzmann method uses the real speed of sound and viscosity of air, so airflow and sound come out together.',
             f'Three grids: 1, 0.5 and 0.25 mm. The finest has {ncell/1e6:.1f} million cells.',
             'Run on Hábrók, the supercomputer of the University of Groningen.']
    def draw(n, nb):
        fig = new_fig(); header(fig, 4, 'How we simulated it'); yy = 170
        for s in lines[:n]: yy = TB(fig, 90, yy, s, 30, FG, width=880, gap=0.7)
        ax = AX(fig, 1080, 220, 760, 560)
        for i, (g, f) in enumerate(grid[:nb]):
            yv = 2 - i
            if f: ax.barh(yv, f / 1e3, height=0.55, color=[MUTED, BLUE, RED][i]); ax.text(f / 1e3 + 0.05, yv, f'{f/1e3:.2f} kHz', va='center', fontsize=24, color=FG)
            else: ax.text(0.05, yv, 'no whistle: grid too coarse', va='center', fontsize=22, color=MUTED)
        ax.set_yticks([2, 1, 0]); ax.set_yticklabels(['1 mm', '0.5 mm', '0.25 mm'], fontsize=22); ax.set_xlim(0, 3.2); ax.set_ylim(-0.6, 2.6)
        ax.set_xlabel('pitch of the whistle [kHz]', fontsize=20); [ax.spines[k].set_visible(False) for k in ('top', 'right')]
        T(fig, 1080, 150, 'Pitch on each grid', 28, FG, 'bold')
        if nb >= 3 and grid[1][1] and grid[2][1]:
            TB(fig, 1080, 870, f'The two finer grids agree within {abs(grid[2][1]-grid[1][1])/grid[2][1]*100:.0f} %.', 28, FG, width=760)
        img = grab(fig); plt.close(fig); return img
    st = [draw(n, nb) for n, nb in [(0, 0), (1, 0), (2, 0), (3, 1), (3, 2), (3, 3), (4, 3)]]
    scene(mv, 17, st, [0, 0.5, 3.0, 6.5, 8.0, 9.5, 12.5], name='Method')

def s06_flow(mv):
    slow = 1 / FPS / (se * dt)
    mv.chapter(f'The simulation, {round(slow, -2):,.0f} times slower')
    dur = 20; N = int(dur * FPS); k0 = K0 + 50; ks = k0 + np.arange(N) % (len(SX) - k0)
    ylim, zlim = (-48, 52), (0, 50)
    mv.sound(loop(SND_MAIN, dur - 1), mv.t + 0.5, GAIN * 0.35)
    fig = new_fig(); header(fig, 5, f'The simulation, {round(slow, -2):,.0f} times slower')
    b1 = box_fit(ylim, zlim, 40, 120, 910, 640); b2 = box_fit(ylim, zlim, 970, 120, 910, 640)
    a1 = plane_ax(fig, b1, ylim, zlim); a2 = plane_ax(fig, b2, ylim, zlim)
    sp0 = np.zeros(SOL.shape); vmax = float(min(1.3 * Uj, 90)); pl = float(np.percentile(PRMS[~SOL], 99)) * 1.2
    im1 = a1.imshow(sp0.T, origin='lower', extent=EXT, cmap=SPEED, norm=PowerNorm(0.6, 0, vmax), interpolation='bilinear')
    im2 = a2.imshow(sp0.T, origin='lower', extent=EXT, cmap=PRESS, norm=TwoSlopeNorm(0, -pl, pl), interpolation='bilinear')
    for a in (a1, a2): a.imshow(solid_img(SOL), origin='lower', extent=EXT, interpolation='nearest')
    T(fig, b1[0], 780, 'Air speed', 28, FG, 'bold'); T(fig, b1[0], 825, f'white: still air; dark blue: up to {vmax:.0f} m/s', 22, MUTED)
    T(fig, b2[0], 780, 'Pressure', 28, FG, 'bold'); T(fig, b2[0], 825, 'red: above average; blue: below average', 22, MUTED)
    tt = T(fig, 1840, 118, '', 24, MUTED, ha='right')
    a3 = AX(fig, 300, 900, 1320, 130); win = 2e-3; nwin = int(win / dt)
    sos = butter(3, [0.6 * F0, 1.6 * F0], btype='band', fs=1 / dt, output='sos'); b1s = sosfiltfilt(sos, h[:, 0]); b3s = sosfiltfilt(sos, h[:, 2])
    l1, = a3.plot([], [], color=BLUE, lw=2, label='sphere'); l3, = a3.plot([], [], color=RED, lw=2, label='gap')
    am = np.percentile(np.abs(b3s[int(T0 / dt):]), 99.5); a3.set_ylim(-am, am); a3.set_xlim(-win * 1e3, 0); a3.axis('off')
    a3.legend(loc='center left', bbox_to_anchor=(-0.17, 0.5), frameon=False, fontsize=20, labelcolor=FG)
    for j, k in enumerate(ks):
        a = SX[k].astype(np.float32); sp = np.sqrt((a[:3] ** 2).sum(0)) * uf; pp = a[3] * pf - PM
        im1.set_data(np.where(SOL, np.nan, sp).T); im2.set_data(np.where(SOL, np.nan, pp).T)
        i = min(int(k * se), len(h) - 1); i0 = max(i - nwin, 0); tx = (np.arange(i0, i) - i) * dt * 1e3
        l1.set_data(tx, b1s[i0:i]); l3.set_data(tx, b3s[i0:i]); tt.set_text(f'simulated time {TF[k]*1e3:6.2f} ms')
        img = grab(fig); t = j / FPS; f_ = fader(dur, t); mv.add(fade(img, f_) if f_ < 1 else img)
        if j == N // 2: mv.still(img, 'Flow')
    plt.close(fig)

def s07_cycle(mv):
    mv.chapter('One cycle of the oscillation')
    period = 2.5; slow = period * F0; dur = 16; N = int(dur * FPS)
    ylim, zlim = (-22, 33), (4, 40); box = box_fit(ylim, zlim, 50, 120, 1100, 800)
    fig = new_fig(); header(fig, 6, 'One cycle of the oscillation'); ax = plane_ax(fig, box, ylim, zlim)
    allp = np.array([phase_field(p) for p in np.linspace(-np.pi, np.pi, 16)]); pl = float(np.percentile(np.abs(allp[:, ~SOL]), 98))
    im = ax.imshow(np.zeros(SOL.T.shape), origin='lower', extent=EXT, cmap=PRESS, norm=TwoSlopeNorm(0, -pl, pl), interpolation='bilinear')
    ax.imshow(solid_img(SOL), origin='lower', extent=EXT, interpolation='nearest')
    iy = lambda v: int(np.abs(y - v).argmin()); iz = lambda v: int(np.abs(z - v).argmin())
    g1 = (iy(0), iz(21.7)); g3 = (iy(0), iz(35.3)); A1 = np.abs(allp[:, g1[0], g1[1]]).max(); A3 = np.abs(allp[:, g3[0], g3[1]]).max()
    yy = TB(fig, 1210, 140, f'One cycle, slowed down about {round(slow, -2):,.0f} times', 26, MUTED, width=640, gap=1.0)
    yy = TB(fig, 1210, yy, f'The sphere and the gap around it breathe in turns, about {round(F0, -2):,.0f} times a second.', 32, FG, 'bold', width=640, gap=0.8)
    TB(fig, 1210, yy, 'Two coupled air springs, kept going by the jet.', 30, FG, width=640)
    ga = AX(fig, 1260, 720, 520, 270); ga.set_xlim(-0.6, 1.6); ga.set_ylim(-1.15, 1.15); ga.axis('off')
    ga.axhline(0, color=MUTED, lw=1)
    bars = ga.bar([0, 1], [0, 0], width=0.55, color=[BLUE, RED])
    ga.text(0, -1.15, 'sphere', ha='center', va='top', fontsize=22, color=FG); ga.text(1, -1.15, 'gap', ha='center', va='top', fontsize=22, color=FG)
    for k in range(N):
        t = k / FPS; ph = ((t / period) % 1) * 2 * np.pi - np.pi; f = phase_field(ph)
        im.set_data(np.where(SOL, np.nan, f).T)
        nrm = max(A1, A3, 1e-9); bars[0].set_height(np.nan_to_num(f[g1] / nrm)); bars[1].set_height(np.nan_to_num(f[g3] / nrm))
        img = grab(fig); a = fader(dur, t); mv.add(fade(img, a) if a < 1 else img)
        if k == int(0.6 * period * FPS): mv.still(img, 'Cycle')
    plt.close(fig)

def s08_source(mv):
    mv.chapter('Where the sound is born')
    dur = 15; N = int(dur * FPS); Lp = 20 * np.log10(np.maximum(PRMS, 1e-3) / 2e-5); Lm = float(np.nanmax(np.where(SOL, np.nan, Lp)))
    ylim, zlim = (-30, 30), (0, 44); box = box_fit(ylim, zlim, 50, 160, 860, 650)
    fig = new_fig(); header(fig, 7, 'Where the sound is born'); ax = plane_ax(fig, box, ylim, zlim)
    ax.imshow(np.where(SOL, np.nan, Lp).T, origin='lower', extent=EXT, cmap=LEVEL, vmin=Lm - 30, vmax=Lm, interpolation='bilinear')
    ax.imshow(solid_img(SOL), origin='lower', extent=EXT, interpolation='nearest')
    Lf = np.where(SOL, -np.inf, Lp); jm = np.unravel_index(np.argmax(Lf), Lf.shape)
    ax.plot(y[jm[0]], z[jm[1]], 'o', ms=34, mfc='none', mec=DBLUE, mew=2)
    TB(fig, box[0], 850, f'Strength of the pressure swings. Strongest at the lip: about {Lm:.0f} dB.', 26, FG, width=box[2])
    zy, zz = (4, 22), (9, 27); bz = box_fit(zy, zz, 1000, 160, 860, 650); az = plane_ax(fig, bz, zy, zz)
    vort = np.zeros(SOL.shape); im = az.imshow(vort.T, origin='lower', extent=EXT, cmap=PRESS, interpolation='bilinear')
    az.imshow(solid_img(SOL), origin='lower', extent=EXT, interpolation='nearest')
    for s in az.spines.values(): s.set_visible(True)
    TB(fig, bz[0], 850, 'Close-up of the lip: swirls of air (vortices) shed in step with the tone, spinning one way (red) or the other (blue).', 26, FG, width=bz[2])
    dmm = dxm * 1e-3; k0 = K0 + 50; sel = (slice(int(np.abs(y - zy[0]).argmin()), int(np.abs(y - zy[1]).argmin())), slice(int(np.abs(z - zz[0]).argmin()), int(np.abs(z - zz[1]).argmin())))
    def vorticity(k):
        a = SX[k, :2].astype(np.float32) * uf; w = np.gradient(a[1], dmm, axis=0) - np.gradient(a[0], dmm, axis=1); w[SOL] = np.nan; return w
    vl = float(np.nanpercentile(np.abs(np.array([vorticity(k0 + i)[sel] for i in range(0, 40, 4)])), 97)); im.set_norm(TwoSlopeNorm(0, -vl, vl))
    for k in range(N):
        im.set_data(vorticity(k0 + k % (len(SX) - k0)).T); img = grab(fig); a = fader(dur, k / FPS); mv.add(fade(img, a) if a < 1 else img)
        if k == N // 2: mv.still(img, 'Source')
    plt.close(fig)

def s09_sweep(mv):
    mv.chapter('Blow harder, higher pitch')
    sw = num(['sweep_0p5mm'], {}); pts = []
    for p in (500, 1000, 1500, 2000):
        f = sw.get(str(p), sw.get(p, {})).get('f0_listener_Hz') if sw else None
        if f is None and p in SWEEP: f = tone(SWEEP[p], 1 / SR, 1000, 4000)
        pts.append((p, f))
    seg = 3.2; t_in = 2.0; dur = t_in + seg * len(pts) + 1.5
    for i, (p, _) in enumerate(pts):
        if p in SWEEP: mv.sound(loop(SWEEP[p], seg - 0.4), mv.t + t_in + seg * i + 0.2, SWEEP_GAIN)
    def draw(n):
        fig = new_fig(); header(fig, 8, 'Blow harder, higher pitch')
        yy = TB(fig, 90, 170, 'Blow harder and the pitch goes up.', 40, FG, 'bold', width=780, gap=0.6)
        TB(fig, 90, yy, 'The tone follows the speed of the jet, but stays locked to the air springs inside.', 30, MUTED, width=780)
        T(fig, 90, 990, 'Simulations on the 0.5 mm grid, 40 ms each. Listen to each step.', 22, MUTED)
        ax = AX(fig, 1000, 170, 820, 650)
        ax.set_xlim(0, 2.5); ax.set_ylim(0, 3); ax.set_xlabel('blowing pressure [kPa]', fontsize=22); ax.set_ylabel('pitch [kHz]', fontsize=22)
        [ax.spines[k].set_visible(False) for k in ('top', 'right')]; ax.grid(alpha=0.15)
        good = [(p / 1e3, f / 1e3) for p, f in pts[:n] if p >= 1000 and f]
        if len(good) > 1: ax.plot(*zip(*good), color=BLUE, lw=2)
        for i, (p, f) in enumerate(pts[:n]):
            cur = i == n - 1
            if p < 1000 or not f:
                ax.plot(p / 1e3, 0.25, 'o', ms=16, mfc=BG, mec=MUTED, mew=2); ax.text(p / 1e3, 0.5, 'just a hiss', ha='center', fontsize=20, color=MUTED)
            else:
                ax.plot(p / 1e3, f / 1e3, 'o', ms=20 if cur else 15, color=RED if cur else BLUE)
                ax.text(p / 1e3, f / 1e3 + 0.18, f'{f/1e3:.1f} kHz', ha='center', fontsize=22, color=FG)
        if n: T(fig, 90, 600, f'{pts[n-1][0]/1e3:g} kPa', 70, RED if n > 1 else MUTED, 'bold')
        img = grab(fig); plt.close(fig); return img
    st = [draw(n) for n in range(len(pts) + 1)]
    scene(mv, dur, st, [0] + [t_in + seg * i for i in range(len(pts))], name='Sweep')

def s09b_printed(mv):
    if not REC100: print('   no recordings found under', A.recordings, '; scene skipped'); return
    mv.chapter('The real whistle')
    blows = sorted(REC100, key=len, reverse=True)[:2]; b1, b2 = [b[:int(2.4 * SR)] for b in blows]
    t_in = 1.5; t_b2 = t_in + 3.0; t_sim = t_b2 + 3.4; t_end = t_sim + 4.0; dur = t_end + 5.0
    mv.sound(b1, mv.t + t_in, REC_GAIN, ramp=0.08); mv.sound(b2, mv.t + t_b2, REC_GAIN, ramp=0.08)
    mv.sound(loop(SND_MAIN, 3.4), mv.t + t_sim, GAIN, ramp=0.08)
    def psd(x):
        f, p = welch(x, fs=SR, nperseg=8192); s = (f >= 500) & (f <= 4000); d = 10 * np.log10(p[s] + 1e-30); return f[s], d - d.max()
    fr, Pr = psd(np.concatenate(REC100)); fs_, Ps = psd(SND_MAIN)
    lo = hi = None                                    # mean band frequencies over both positions, as in the paper
    allb = [b for k in ('w100_pos1', 'w100_pos2') for b in REC_NUM.get(k, {}).get('blows', [])]
    if allb: lo = np.mean([b['f_low'] for b in allb]); hi = np.mean([b['f_high'] for b in allb])
    def draw(n):
        fig = new_fig(); header(fig, 9, 'The real whistle')
        yy = TB(fig, 90, 170, 'We 3D-printed the same model and blew it.', 40, FG, 'bold', width=780, gap=0.6)
        if n >= 1: yy = TB(fig, 90, yy, 'Listen: the printed whistle.', 30, RED, width=780, gap=0.4)
        if n >= 2: yy = TB(fig, 90, yy, 'And the simulation.', 30, BLUE, width=780, gap=0.8)
        if n >= 3:
            s = 'Both show two bands an octave apart'
            if lo and hi: s += f': {lo/1e3:.2f} and {hi/1e3:.2f} kHz in the print, 1.16 and 2.30 kHz in the simulation.'
            TB(fig, 90, yy, s, 30, FG, 'bold', width=780)
        T(fig, 90, 990, 'Blown by mouth; the blowing pressure was not measured, so the match is qualitative.', 22, MUTED)
        ax = AX(fig, 1000, 170, 820, 650)
        ax.set_xlim(0.5, 4); ax.set_ylim(-45, 5); ax.set_xlabel('frequency [kHz]', fontsize=22); ax.set_ylabel('level [dB]', fontsize=22)
        [ax.spines[k].set_visible(False) for k in ('top', 'right')]; ax.grid(alpha=0.15)
        if n >= 1: ax.plot(fr / 1e3, Pr, color=RED, lw=2.2, label='printed model (recorded)')
        if n >= 2: ax.plot(fs_ / 1e3, Ps, color=BLUE, lw=2.2, label='simulation, 2 kPa')
        if n >= 1: ax.legend(loc='upper right', frameon=False, fontsize=20, labelcolor=FG)
        img = grab(fig); plt.close(fig); return img
    scene(mv, dur, [draw(n) for n in range(4)], [0, t_in, t_sim, t_end], name='Printed')

def s10_scary(mv):
    mv.chapter('Why it sounds like a scream')
    rr = num(['roughness', A.main], {}); tonal = rr.get('tonal_frac'); env = rr.get('env_depth')
    if tonal is None or env is None:                      # compute here if the numbers file is missing
        x = h[int(T0 / dt):, 4]; xb = band(x - x.mean(), dt, 300, 8000); e = np.abs(hilbert(xb)); env = band(e - e.mean(), dt, 5, 1000).std() / e.mean()
        n = 8 * len(xb); Pp = np.abs(np.fft.rfft(xb * np.hanning(len(xb)), n=n)) ** 2; f = np.fft.rfftfreq(n, dt); s = (f > 300) & (f < 8000)
        tm = np.zeros_like(f, bool)
        for k in range(1, int(8000 // F0) + 1): tm |= np.abs(f - k * F0) <= 0.04 * k * F0
        tonal = float(Pp[s & tm].sum() / Pp[s].sum())
    dur = 19; mv.sound(loop(SND_MAIN, dur - 1.5), mv.t + 1.0, GAIN)
    x = SND_MAIN; nper = int(0.004 * SR); fs_, ts_, Sg = spectrogram(x, fs=SR, nperseg=nper, noverlap=nper * 7 // 8); keep = fs_ <= 6000
    Ld = 10 * np.log10(Sg[keep] + 1e-20)
    pts = [f'Pitched noise: only about {round(tonal*100, -1):.0f} % of the sound is in the tone and its overtones.',
           'Two bands an octave apart, like a voice that cracks.',
           f'The loudness wobbles by about {round(env*100, -1):.0f} %: the sound is rough.',
           'Rough, noisy and unstable: the same acoustic features as human screams, which our brains treat as an alarm.']
    def draw(n):
        fig = new_fig(); header(fig, 10, 'Why it sounds like a scream')
        ax = AX(fig, 1080, 160, 780, 620); ax.pcolormesh(ts_ * 1e3, fs_[keep] / 1e3, Ld, shading='auto', cmap='magma', vmin=Ld.max() - 45, vmax=Ld.max(), rasterized=True)
        ax.set_xlabel('time [ms]', fontsize=20); ax.set_ylabel('frequency [kHz]', fontsize=20)
        T(fig, 1080, 860, 'The simulated sound over time (bright = loud)', 22, MUTED)
        yy = 170
        for i, s in enumerate(pts[:n]):
            yy = TB(fig, 90, yy, s, 30, FG, 'bold' if i == 3 else 'normal', width=880, gap=0.8)
        if n >= 4: T(fig, 90, 990, 'Whether this is what frightens listeners still has to be tested.', 22, MUTED)
        img = grab(fig); plt.close(fig); return img
    scene(mv, dur, [draw(n) for n in range(5)], [0, 1.5, 5.0, 8.5, 12.5], name='Scream')

def s11_findings(mv):
    mv.chapter('Key findings')
    Lm = num(['Lp', 'max_dB'], None)
    pts = [f'A breath of {float(R["dp"])/1e3:g} kPa makes a jet of about {Uj:.0f} m/s.',
           f'The jet drives the sphere and the gap in turns: a self-excited tone near {f0_pair/1e3:.1f} kHz.',
           'The pitch holds on a finer grid and rises when you blow harder.',
           'The sound is rough, pitched noise with a second band an octave lower: scream-like.',
           'A 3D-printed whistle, blown by mouth, shows the same two bands an octave apart.']
    if Lm: pts.append(f'The pressure swings are strongest at the lip, about {Lm:.0f} dB.')
    def draw(n):
        fig = new_fig(); header(fig, 11, 'Key findings'); yy = 180
        for i, s in enumerate(pts[:n]):
            yy = bullet(fig, 90, yy, s, 34, FG, width=1700, gap=0.7, mark=i + 1)
        img = grab(fig); plt.close(fig); return img
    scene(mv, 15, [draw(n) for n in range(len(pts) + 1)], [0] + [0.6 + 2.2 * i for i in range(len(pts))], name='Findings')

def s12_experts(mv):
    mv.chapter('For experts')
    def draw(n):
        fig = new_fig(); header(fig, 12, 'For experts')
        yy = tile(fig, 90, 170, 1740, 'Method', f'D3Q19 lattice Boltzmann, regularized BGK collision with a Smagorinsky model (large-eddy simulation). '
                        f'Grids of 1, 0.5 and 0.25 mm, up to {ncell/1e6:.1f} million cells. Blowing pressure 0.5 to 4 kPa. '
                        'Up to 105 ms of simulated time.', 28) + 50
        if n:
            tile(fig, 90, yy, 1740, 'Limits', 'A modern replica, not an original whistle. The pitch is grid-converged; the loudness, the flow rate '
                            'and the roughness are not yet. The printed whistle was blown by mouth at unknown pressure, '
                            'so that comparison is qualitative.', 28, red=False)
        img = grab(fig); plt.close(fig); return img
    scene(mv, 13, [draw(0), draw(1)], [0, 5.0], name='Experts')

def s13_credits(mv):
    mv.chapter('Credits')
    def draw(n):
        fig = new_fig(); header(fig, 13, 'Credits')
        T(fig, 90, 160, 'Code, data and paper', 28, MUTED); T(fig, 90, 205, 'github.com/mmunozarias/aztec-whistle-lbm', 46, BLUE, 'bold')
        if n:
            rows = [('Authors', 'Mauricio Muñoz-Arias and Jesús Barradas-Berglind, Faculty of Science and Engineering, University of Groningen'),
                    ('3D model', '"Working Death Whistle from Ghostbusters: Afterlife" by HawkWorley, Thingiverse 5148690, CC BY'),
                    ('Music', 'Synthesized by the video script itself; no third-party audio'),
                    ('Computing', 'Hábrók cluster, Center for Information Technology, University of Groningen'),
                    ('Sources', 'S. Guilliem Arroyo, INAH (1999); S. Frühholz et al., Communications Psychology (2024); '
                                'L. H. Arnal et al., Current Biology (2015)')]
            yy = 360
            for k_, v in rows:
                T(fig, 90, yy, k_, 24, RED, 'bold'); yy = TB(fig, 330, yy, v, 24, FG, width=1500, gap=0.9)
        img = grab(fig); plt.close(fig); return img
    scene(mv, 12, [draw(0), draw(1)], [0, 1.5], fout=1.0, name='Credits')

# ---------------------------------------------------------------- build
mv = Movie(os.path.join(A.out, '_video_only.mp4'))
SCENES = [s01_cover, s02_discovery, s03_sound, s04_instrument, s04b_diagram, s05_method, s06_flow, s07_cycle, s08_source,
          s09_sweep, s09b_printed, s10_scary, s11_findings, s12_experts, s13_credits]
cover = None
for fn in SCENES:
    t1 = time.time(); print('==', fn.__name__, f'starts at {mv.t:.1f} s', flush=True)
    try:
        out = fn(mv)
        if fn is s01_cover: cover = out
    except Exception:
        print('!!! FAILED', fn.__name__); traceback.print_exc()
    print(f'   {time.time()-t1:.0f} s to render', flush=True)
total = mv.close()

# sound track: gentle limiter, then mux
au = mv.audio[:int(total * SR)]
if not A.no_music: au = au + duck(au, make_music(total))
au = np.tanh(au * 1.2) / 1.2 * 0.9
from scipy.io import wavfile
wavfile.write(os.path.join(A.out, '_audio.wav'), SR, (au * 32767).astype(np.int16))
final = os.path.join(A.out, 'whistle_video.mp4')
subprocess.run([A.ffmpeg, '-y', '-loglevel', 'error', '-i', os.path.join(A.out, '_video_only.mp4'), '-i', os.path.join(A.out, '_audio.wav'),
                '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', final], check=True)
for f in ('_video_only.mp4', '_audio.wav'): os.remove(os.path.join(A.out, f))

# chapters, thumbnail, preview sheet
with open(os.path.join(A.out, 'chapters.txt'), 'w') as fh:
    for t, name in mv.chapters: fh.write(f'{int(t//60)}:{int(t%60):02d} {name}\n')
from PIL import Image
if cover is not None: Image.fromarray(cover).resize((1280, 720), Image.LANCZOS).save(os.path.join(A.out, 'thumbnail.jpg'), quality=92)
if mv.stills:
    tw, th = 480, 270; cols = 4; rows_ = int(np.ceil(len(mv.stills) / cols)); sheet = Image.new('RGB', (cols * tw, rows_ * th), BG)
    for i, (img, name) in enumerate(mv.stills): sheet.paste(Image.fromarray(img).resize((tw, th), Image.LANCZOS), ((i % cols) * tw, (i // cols) * th))
    sheet.save(os.path.join(A.out, 'preview.jpg'), quality=88)
print(f'done: {final}, {total:.1f} s long, {mv.n} frames, {(time.time()-t_start)/60:.1f} min')
print(open(os.path.join(A.out, 'chapters.txt')).read())
