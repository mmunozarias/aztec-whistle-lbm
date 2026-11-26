"""Figures and key numbers for the paper, in a red / blue / black style.

Run from the repository root, with runs/ holding the simulation folders:
  python code/paper/paper_figures.py --out paper_figures --stl code/input/_BothDeathwhistlefile98.stl
Optional: --only signals mean ...   (make only some figures)

Folders used (all under --runs):
  dx0.25_dp2000_long  finest grid, 105 ms, case O         phase averages, vortices, roughness
  dx0.25_dp2000       finest grid, 20 ms, cases O and S   signals, mean flow
  dx1_dp2000, dx0.5_dp2000, dx0.25_dp2000                 grid convergence (window 5-20 ms)
  dx0.5_dp{500..4000}, dx0.25_dp{2000,4000}               blowing-pressure sweep
  dx*_dp*_long                                            roughness
Every figure is made in its own try-block; numbers quoted in the paper go to <out>/numbers.json.
"""
import argparse, glob, json, os, re, traceback
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, PowerNorm, ListedColormap, LinearSegmentedColormap, to_rgba
from scipy.signal import welch, butter, sosfiltfilt, hilbert, spectrogram
from scipy import ndimage as ndi

# ---------------------------------------------------------------- style
BLUE, RED, BLACK, GREY = '#1f4fb4', '#c8102e', '#111111', '#8c8c8c'
SOLID = (0.62, 0.62, 0.62, 1.0); AIR = '#c6daf3'; LIGHTBLUE = '#dbe6f6'; LIGHTRED = '#f6dcdf'
COL, WIDE = 3.45, 7.1
REDS = LinearSegmentedColormap.from_list('wr', ['#ffffff', '#f2b8bf', RED, '#5c0010'])
BLUES = LinearSegmentedColormap.from_list('wb', ['#ffffff', '#b9cbef', BLUE, '#0b1f4d'])
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'axes.edgecolor': '#444', 'axes.labelcolor': '#111', 'xtick.color': '#333', 'ytick.color': '#333',
    'axes.prop_cycle': matplotlib.cycler(color=[BLUE, RED, BLACK])})
CASE_COL = {'orig': BLUE, 'sealed': RED}
CASE_LAB = {'orig': 'case O (original)', 'sealed': 'case S (slots sealed)'}
GRID_STYLE = {1.0: dict(color=BLACK, ls=':', lw=1.0), 0.5: dict(color=BLUE, ls='-', lw=0.9), 0.25: dict(color=RED, ls='-', lw=0.9)}
PROBES = {'P1': (0.1, 21.7), 'P2': (17, 18.5), 'P3': (0, 35.3), 'P4': (-20, 24), 'P5': (-54, 20)}
LIP = (12.5, 18.0); SPHERE_C = np.array([-0.37, 0.08, 21.63]); D_OPEN = 3.94e-3; RHO = 1.2

ap = argparse.ArgumentParser()
ap.add_argument('--runs', default='runs'); ap.add_argument('--out', default='paper_figures')
ap.add_argument('--stl', default='code/input/_BothDeathwhistlefile98.stl')
ap.add_argument('--main', default='dx0.25_dp2000_long'); ap.add_argument('--pair', default='dx0.25_dp2000')
ap.add_argument('--only', nargs='*', default=None)
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
NUM = json.load(open(f'{A.out}/numbers.json')) if os.path.exists(f'{A.out}/numbers.json') else {}

def P(*p): return os.path.join(A.runs, *p)
def exists(run, case='orig'): return os.path.exists(P(run, f'res_{case}.npz'))
def long_enough(run, T=0.09):
    if not exists(run): return False
    r = load(run); return len(r['hist']) * float(r['dt']) >= T
def load(run, case='orig'): return np.load(P(run, f'res_{case}.npz'))
def dxof(r): return float(r['y'][1] - r['y'][0])
def save(fig, name):
    fig.savefig(f'{A.out}/{name}.pdf'); fig.savefig(f'{A.out}/{name}.png', dpi=150); plt.close(fig); print('  wrote', name)
def label(ax, s, x=-0.14, y=1.02): ax.text(x, y, s, transform=ax.transAxes, fontweight='bold', va='bottom')
def solid_rgba(S):
    g = np.zeros(S.shape + (4,)); g[S] = SOLID; return g.transpose(1, 0, 2)

def tone(x, dt, lo=300, hi=8000):
    x = x - x.mean(); n = 8 * len(x); F = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=n)); f = np.fft.rfftfreq(n, dt)
    s = (f > lo) & (f < hi); return float(f[s][np.argmax(F[s])])

def band(x, dt, lo, hi):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), dt); X[(f < lo) | (f > hi)] = 0; return np.fft.irfft(X, len(x))

def spl(x, dt):
    return float(20 * np.log10(band(x - x.mean(), dt, 300, 8000).std() / 2e-5))

def summary(r, t0, t1=None):
    """Probe-based numbers for one run and window."""
    h = r['hist'].astype(float); dt = float(r['dt']); i0 = int(t0 / dt); i1 = len(h) if t1 is None else min(len(h), int(round(t1 / dt)))
    w = h[i0:i1]; f5 = tone(w[:, 4], dt); dp = float(r['dp'])
    sos = butter(3, [0.6 * f5, 1.5 * f5], btype='band', fs=1 / dt, output='sos')
    a1 = hilbert(sosfiltfilt(sos, w[:, 0])); a3 = hilbert(sosfiltfilt(sos, w[:, 2]))
    return dict(dx_mm=round(dxof(r), 3), dp_Pa=dp, window_ms=round((i1 - i0) * dt * 1e3, 1), res_Hz=round(1 / ((i1 - i0) * dt), 1),
                f0_listener_Hz=round(f5), f0_sphere_Hz=round(tone(w[:, 0], dt)), p1_mean_Pa=round(w[:, 0].mean()),
                p1_amp_Pa=round(float(np.abs(a1).mean())), p3_amp_Pa=round(float(np.abs(a3).mean())),
                phase_p3_p1_deg=round(float(np.degrees(np.angle(np.mean(a3 * np.conj(a1)))))),
                listener_dB=round(spl(w[:, 4], dt), 1), Q_Lps=round(float(w[:, -1].mean()), 3),
                Uj_mps=round(float(np.sqrt(2 * dp / RHO)), 1), St=round(f5 * D_OPEN / np.sqrt(2 * dp / RHO), 3))

def plane_stats(r, key, t0, t1=None, chunk=150):
    """Mean velocity (3,n1,n2) and mean speed in m/s, rms pressure in Pa, mean pressure in Pa, chunked over frames."""
    S = r[key]; dt = float(r['dt']); se = int(r['save_every']); uf = float(r['ufac']); pf = float(r['pfac'])
    k0 = int(np.ceil(t0 / (dt * se))); k1 = len(S) if t1 is None else min(len(S), int(t1 / (dt * se)) + 1)
    su = np.zeros((3,) + S.shape[2:]); ss = np.zeros(S.shape[2:]); sp = np.zeros(S.shape[2:]); sp2 = np.zeros(S.shape[2:]); n = 0
    for k in range(k0, k1, chunk):
        a = S[k:min(k + chunk, k1)].astype(np.float64)
        su += a[:, :3].sum(0); ss += np.sqrt((a[:, :3] ** 2).sum(1)).sum(0); p = a[:, 3] * pf; sp += p.sum(0); sp2 += (p ** 2).sum(0); n += len(a)
    pm = sp / n
    return su / n * uf, ss / n * uf, np.sqrt(np.maximum(sp2 / n - pm ** 2, 0)), pm

def run_fig(name, fn):
    if A.only and name not in A.only: return
    print('==', name)
    try: fn()
    except Exception: print('!!! FAILED', name); traceback.print_exc()

# ---------------------------------------------------------------- 1. geometry
def fig_geometry():
    gpath = P('dx0.5_dp2000', 'geom.npz') if os.path.exists(P('dx0.5_dp2000', 'geom.npz')) else P(A.pair, 'geom.npz')
    g = np.load(gpath); solid = g['orig']; x, y, z = g['x'], g['y'], g['z']; dx = float(g['dx'])
    air, rr = None, None
    for r_mm in (4, 6, 8, 10, 12):           # internal air: close the solid with a ball, fill, remove the solid
        r = r_mm / dx
        dil = ndi.distance_transform_edt(~solid) <= r; clo = ndi.distance_transform_edt(dil) > r
        a = ndi.binary_fill_holes(clo | solid) & ~solid; vol = a.sum() * (dx / 10) ** 3
        lab, _ = ndi.label(a); c0 = lab[int(np.abs(x).argmin()), int(np.abs(y).argmin()), int(np.abs(z - 21.6).argmin())]
        if c0 > 0: a = lab == c0
        vol = a.sum() * (dx / 10) ** 3
        if 8 < vol < 30: air, rr = a, r_mm; break
    if air is None: air = np.zeros_like(solid); print('  internal air not found; panel (b) shows the solid only')
    else: print(f'  internal air: closing radius {rr} mm, volume {air.sum()*(dx/10)**3:.1f} cm3'); NUM['air_volume_cm3'] = round(float(air.sum() * (dx / 10) ** 3), 1)
    fig = plt.figure(figsize=(WIDE, 2.55)); gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.75, 0.95], wspace=0.12)
    ax = fig.add_subplot(gs[0], projection='3d'); ok = False
    try:
        import trimesh
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        m = trimesh.load(A.stl, force='mesh')
        try: m = m.simplify_quadric_decimation(face_count=70000)
        except Exception: pass
        tri = m.triangles; nrm = m.face_normals; L = np.array([0.35, -0.55, 0.75]); L /= np.linalg.norm(L)
        sh = 0.30 + 0.62 * np.clip(nrm @ L, 0, 1); fc = np.stack([sh, sh, sh, np.ones_like(sh)], 1)
        ax.add_collection3d(Poly3DCollection(tri, facecolors=fc, edgecolor='none', rasterized=True))
        c = m.bounds.mean(0); rad = m.extents.max() / 2 * 0.52
        ax.set_xlim(c[0] - rad, c[0] + rad); ax.set_ylim(c[1] - rad, c[1] + rad); ax.set_zlim(c[2] - rad, c[2] + rad); ok = True
    except Exception as e: print('  3D rendering skipped:', e)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(18, -58); ax.set_axis_off(); ax.text2D(0.02, 0.95, '(a)', transform=ax.transAxes, fontweight='bold')
    img = np.where(solid, 1, np.where(air, 2, 0)); cm = ListedColormap(['#ffffff', SOLID[:3], AIR])
    ix = lambda v, arr: int(np.abs(arr - v).argmin())
    ax = fig.add_subplot(gs[1]); im = img[ix(0, x)]
    ax.imshow(im.T, origin='lower', cmap=cm, vmin=0, vmax=2, extent=[y[0], y[-1], z[0], z[-1]], interpolation='nearest')
    B = dict(fontsize=6.5, ha='center', va='center', bbox=dict(boxstyle='circle,pad=0.18', fc='white', ec=BLACK, lw=0.5))
    for k, (py, pz) in enumerate([(46, 21), (22, 19.5), (12.5, 17.2), (0, 22), (-1, 34.8), (-22, 26), (-4, 6.5)], 1): ax.text(py, pz, str(k), **B)
    ax.set_xlim(-52, 56); ax.set_ylim(-10, 66); ax.set_xlabel('$y$ [mm]'); ax.set_ylabel('$z$ [mm]'); ax.set_aspect('equal')
    ax.text(0.01, 0.97, '(b)', transform=ax.transAxes, fontweight='bold', va='top')
    ax = fig.add_subplot(gs[2]); im = img[:, ix(0, y), :]
    ax.imshow(im.T, origin='lower', cmap=cm, vmin=0, vmax=2, extent=[x[0], x[-1], z[0], z[-1]], interpolation='nearest')
    ax.annotate('wall 1.2 mm', (11.3, 24), (6, 58), fontsize=6.5, arrowprops=dict(arrowstyle='-', lw=0.5, color='#333'))
    ax.set_xlim(-27, 27); ax.set_ylim(-10, 66); ax.set_xlabel('$x$ [mm]'); ax.set_aspect('equal'); ax.set_yticklabels([])
    ax.text(0.03, 0.97, '(c)', transform=ax.transAxes, fontweight='bold', va='top')
    save(fig, 'fig_geometry')

# ---------------------------------------------------------------- 2. set-up
def fig_setup():
    g = np.load(P(A.pair, 'geom.npz')); o = g['orig']; se = g['sealed']; inl = g['inlet']; x, y, z = g['x'], g['y'], g['z']; dx = float(g['dx'])
    i0 = int(np.abs(x).argmin()); j0 = int(np.abs(y).argmin()); sp = 6.0
    fig, axs = plt.subplots(1, 2, figsize=(WIDE, 2.45), gridspec_kw={'width_ratios': [2.15, 1], 'wspace': 0.12})
    ax = axs[0]; S = o[i0]; ext = [y[0] - dx / 2, y[-1] + dx / 2, z[0] - dx / 2, z[-1] + dx / 2]
    img = np.ones(S.shape + (4,)); img[S] = SOLID; img[se[i0] & ~S] = to_rgba(BLACK); img[inl[i0]] = to_rgba(RED)
    ax.imshow(img.transpose(1, 0, 2), origin='lower', extent=ext, interpolation='nearest')
    from matplotlib.patches import Rectangle
    for (a, b, w, hh) in [(y[0], z[0], sp, z[-1] - z[0]), (y[-1] - sp, z[0], sp, z[-1] - z[0]), (y[0], z[0], y[-1] - y[0], sp), (y[0], z[-1] - sp, y[-1] - y[0], sp)]:
        ax.add_patch(Rectangle((a, b), w, hh, fc=BLUE, alpha=0.12, ec='none'))
    ax.text(28, z[0] + 1.2, 'sponge layer (6 mm)', fontsize=6.5, color=BLUE)
    for nm, (py, pz) in PROBES.items():
        ax.plot(py, pz, marker='o', ms=3.2, mfc='white', mec=BLACK, mew=0.7); ax.annotate(nm, (py, pz), (py + 2.5, pz + 3.2), fontsize=6.5)
    ax.annotate('pressure inlet', (44, 18), (36, -3), fontsize=6.5, color=RED, arrowprops=dict(arrowstyle='-', lw=0.5, color=RED))
    ax.annotate('lip seal', (48, 23.5), (51, 34), fontsize=6.5, arrowprops=dict(arrowstyle='-', lw=0.5))
    ax.annotate('slot seal (case S)', (-3, 6), (-36, -6), fontsize=6.5, arrowprops=dict(arrowstyle='-', lw=0.5))
    ax.set_xlim(y[0], y[-1]); ax.set_ylim(z[0], z[-1]); ax.set_aspect('equal'); ax.set_xlabel('$y$ [mm]'); ax.set_ylabel('$z$ [mm]')
    ax.text(0.01, 0.98, '(a)', transform=ax.transAxes, fontweight='bold', va='top')
    ax = axs[1]; S = o[:, j0]; img = np.ones(S.shape + (4,)); img[S] = SOLID; img[se[:, j0] & ~S] = to_rgba(BLACK)
    ax.imshow(img.transpose(1, 0, 2), origin='lower', extent=[x[0] - dx / 2, x[-1] + dx / 2, z[0] - dx / 2, z[-1] + dx / 2], interpolation='nearest')
    ax.set_xlim(-30, 30); ax.set_ylim(z[0], z[-1]); ax.set_aspect('equal'); ax.set_xlabel('$x$ [mm]')
    ax.text(0.03, 0.98, '(b)', transform=ax.transAxes, fontweight='bold', va='top')
    NUM['grid_cells_pair'] = list(o.shape); NUM['grid_dx_pair_mm'] = dx
    save(fig, 'fig_setup')

# ---------------------------------------------------------------- 3. signals (fine grid, O and S, 20 ms)
def fig_signals():
    R = {c: load(A.pair, c) for c in ['orig', 'sealed'] if exists(A.pair, c)}
    dt = float(R['orig']['dt']); fig, axs = plt.subplots(3, 1, figsize=(COL, 4.6), gridspec_kw={'hspace': 0.55})
    for c, r in R.items():
        h = r['hist']; t = np.arange(len(h)) * dt * 1e3; axs[0].plot(t, h[:, 0], color=CASE_COL[c], lw=0.6, label=CASE_LAB[c])
        NUM[f'pair_{c}'] = summary(r, 5e-3, 20e-3)
    axs[0].axvspan(0, 5, color=GREY, alpha=0.15, lw=0); axs[0].text(2.5, 0.92, 'start-up', ha='center', fontsize=6.5, color='#444', transform=axs[0].get_xaxis_transform())
    axs[0].set_xlim(0, 20); axs[0].set_xlabel('$t$ [ms]'); axs[0].set_ylabel('$p_1$ [Pa]'); axs[0].legend(frameon=False, loc='lower right', ncol=2, fontsize=6.5); label(axs[0], '(a)', -0.17)
    h = R['orig']['hist']; t = np.arange(len(h)) * dt * 1e3; m = (t > 15) & (t < 16.3); mm = t > 5
    f0 = tone(h[mm, 4].astype(float), dt); sosb = butter(3, [0.7 * f0, 1.3 * f0], btype='band', fs=1 / dt, output='sos')
    b1 = sosfiltfilt(sosb, h[:, 0].astype(float)); b3 = sosfiltfilt(sosb, h[:, 2].astype(float))
    axs[1].plot(t[m], b1[m], color=BLUE, lw=0.9, label='$p_1$ sphere (P1)')
    axs[1].plot(t[m], b3[m], color=BLACK, lw=0.9, ls='--', label='$p_3$ gap (P3)')
    axs[1].text(0.99, 0.04, f'band-passed around $f_0$ = {f0/1e3:.2f} kHz', transform=axs[1].transAxes, ha='right', fontsize=6, color='#444')
    axs[1].set_xlabel('$t$ [ms]'); axs[1].set_ylabel("$p'$ [Pa]"); axs[1].set_xlim(15, 16.3)
    axs[1].legend(frameon=False, loc='upper center', ncol=2, fontsize=6.5, bbox_to_anchor=(0.5, 1.25)); label(axs[1], '(b)', -0.17)
    for c, r in R.items():
        h = r['hist']; i0 = int(5e-3 / dt); i1 = int(20e-3 / dt); x = h[i0:i1, 4] - h[i0:i1, 4].mean()
        f, Pw = welch(x, fs=1 / dt, nperseg=int(7.5e-3 / dt)); L = 10 * np.log10(Pw * (f[1] - f[0]) / (2e-5) ** 2 + 1e-12)
        axs[2].plot(f / 1e3, L, color=CASE_COL[c], lw=0.9, label=CASE_LAB[c])
    axs[2].set_xlim(0, 8); axs[2].set_ylim(40, 115); axs[2].set_xlabel('$f$ [kHz]'); axs[2].set_ylabel('SPL per band [dB]'); axs[2].legend(frameon=False, loc='upper right', fontsize=6.5); label(axs[2], '(c)', -0.17)
    for a in axs: a.grid(alpha=0.2, lw=0.4)
    save(fig, 'fig_signals')

# ---------------------------------------------------------------- 4. mean fields (fine grid, O and S)
def fig_mean():
    R = {c: load(A.pair, c) for c in ['orig', 'sealed'] if exists(A.pair, c)}
    r0 = R['orig']; x, y, z = r0['x'], r0['y'], r0['z']; ext = [y[0], y[-1], z[0], z[-1]]; ext2 = [x[0], x[-1], z[0], z[-1]]
    st = {c: plane_stats(R[c], 'sx', 5e-3, 20e-3) for c in R}
    vmax = 0.85 * float(np.nanmax(np.where(R['orig']['solid_x'], np.nan, st['orig'][1])))
    fig = plt.figure(figsize=(WIDE, 4.6)); gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.035], hspace=0.42, wspace=0.08)
    yl, zl = (-50, 55), (-3, 58)
    for k, c in enumerate(['orig', 'sealed']):
        if c not in st: continue
        mu, sp, _, _ = st[c]; S = R[c]['solid_x']; ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(np.where(S, np.nan, sp).T, origin='lower', extent=ext, cmap=BLUES, vmin=0, vmax=vmax, interpolation='bilinear')
        ax.imshow(solid_rgba(S), origin='lower', extent=ext)
        ax.streamplot(y, z, np.where(S, 0, mu[0]).T, np.where(S, 0, mu[1]).T, density=1.5, color=BLACK, linewidth=0.4, arrowsize=0.5)
        ax.set_xlim(*yl); ax.set_ylim(*zl); ax.set_aspect('equal'); ax.set_title(f"({'ab'[k]}) case {'OS'[k]}: mean speed", loc='left', pad=2); ax.set_xlabel('$y$ [mm]')
        if k == 0: ax.set_ylabel('$z$ [mm]')
        else: ax.set_yticklabels([])
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 2])); cb.set_label('$\\overline{|\\mathbf{u}|}$ [m/s]')
    mu, sp, _, _ = plane_stats(r0, 'sy', 5e-3, 20e-3); S = r0['solid_y']; ax = fig.add_subplot(gs[1, 0])
    ax.imshow(np.where(S, np.nan, sp).T, origin='lower', extent=ext2, cmap=BLUES, vmin=0, vmax=vmax, interpolation='bilinear'); ax.imshow(solid_rgba(S), origin='lower', extent=ext2)
    ax.streamplot(x, z, np.where(S, 0, mu[0]).T, np.where(S, 0, mu[1]).T, density=1.4, color=BLACK, linewidth=0.4, arrowsize=0.5)
    ax.set_xlim(-26, 26); ax.set_ylim(*zl); ax.set_aspect('equal'); ax.set_title('(c) case O: mean flow, cross-cut $y=0$', loc='left', pad=2); ax.set_xlabel('$x$ [mm]'); ax.set_ylabel('$z$ [mm]')
    _, _, pr, _ = st['orig']; S = r0['solid_x']; Lp = 20 * np.log10(np.maximum(pr, 1e-3) / 2e-5); Lm = float(np.nanmax(np.where(S, np.nan, Lp)))
    ax = fig.add_subplot(gs[1, 1]); im2 = ax.imshow(np.where(S, np.nan, Lp).T, origin='lower', extent=ext, cmap=REDS, vmin=Lm - 35, vmax=Lm, interpolation='bilinear')
    ax.imshow(solid_rgba(S), origin='lower', extent=ext); ax.set_xlim(*yl); ax.set_ylim(*zl); ax.set_aspect('equal'); ax.set_yticklabels([])
    ax.set_title("(d) case O: fluctuation level $L_{p'}$", loc='left', pad=2); ax.set_xlabel('$y$ [mm]')
    ax.annotate('lip of opening', LIP, (22, 40), fontsize=6.5, arrowprops=dict(arrowstyle='-', lw=0.5))
    cb = fig.colorbar(im2, cax=fig.add_subplot(gs[1, 2])); cb.set_label("$L_{p'}$ [dB re 20 $\\mu$Pa]")
    iy = lambda v: int(np.abs(y - v).argmin()); iz = lambda v: int(np.abs(z - v).argmin())
    Lf = np.where(S, -np.inf, Lp); jm = np.unravel_index(np.argmax(Lf), Lf.shape)
    NUM['Lp'] = dict(max_dB=round(Lm, 1), max_at_yz_mm=[round(float(y[jm[0]]), 1), round(float(z[jm[1]]), 1)],
                     sphere_dB=round(float(Lp[iy(0), iz(22)]), 1), gap_top_dB=round(float(Lp[iy(0), iz(35.3)]), 1))
    NUM['peak_mean_speed_mps'] = round(float(np.nanmax(np.where(S, np.nan, st['orig'][1]))), 1)
    save(fig, 'fig_mean')

# ---------------------------------------------------------------- 5. phase averages (fine grid, long run)
def fig_phase():
    r = load(A.main); h = r['hist'].astype(float); dt = float(r['dt']); se = int(r['save_every']); uf = float(r['ufac']); pf = float(r['pfac'])
    y, z = r['y'], r['z']; sol = r['solid_x']; t0 = 20e-3; i0 = int(t0 / dt)
    f0 = tone(h[i0:, 4], dt, 1700, 3300)                       # the whistling tone (two-chamber band)
    sos = butter(3, [0.8 * f0, 1.2 * f0], btype='band', fs=1 / dt, output='sos')
    phi = np.angle(hilbert(sosfiltfilt(sos, h[:, 0] - h[i0:, 0].mean())))
    S = r['sx']; k0 = int(np.ceil(t0 / (dt * se))); ks = np.arange(k0, len(S)); steps = np.minimum(ks * se, len(h) - 1)
    edges = np.linspace(-np.pi, np.pi, 9); b = np.clip(np.digitize(phi[steps], edges) - 1, 0, 7)
    acc = np.zeros((8, 4) + S.shape[2:]); cnt = np.zeros(8); pmean = np.zeros(S.shape[2:]); n = 0
    for j in range(0, len(ks), 150):
        kk = ks[j:j + 150]; a = S[kk[0]:kk[-1] + 1].astype(np.float64); bb = b[j:j + 150]
        for q in range(8):
            sel = bb == q
            if sel.any(): acc[q] += a[sel].sum(0); cnt[q] += sel.sum()
        pmean += a[:, 3].sum(0) * pf; n += len(a)
    avg = acc / np.maximum(cnt, 1)[:, None, None, None]; pmean /= n
    np.save(f'{A.out}/phase_avg_{A.main}.npy', avg.astype(np.float32))
    NUM['phase_average'] = dict(run=A.main, f0_Hz=round(f0), frames_per_bin=[int(c) for c in cnt])
    cen = np.degrees((edges[:-1] + edges[1:]) / 2); ylim, zlim = (-22, 33), (5, 39); ext = [y[0], y[-1], z[0], z[-1]]
    pp_all = [avg[i, 3] * pf - pmean for i in (1, 3, 5, 7)]; pl = float(np.nanpercentile(np.abs(np.where(sol, np.nan, np.array(pp_all))), 99))
    fig = plt.figure(figsize=(WIDE, 2.75)); gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.05], hspace=0.12, wspace=0.06)
    for col, i in enumerate([1, 3, 5, 7]):
        a = avg[i]; sp = np.sqrt((a[:3] ** 2).sum(0)) * uf; pp = a[3] * pf - pmean
        for row in range(2):
            ax = fig.add_subplot(gs[row, col])
            if row == 0:
                im0 = ax.imshow(np.where(sol, np.nan, sp).T, origin='lower', extent=ext, cmap=BLUES, norm=PowerNorm(0.6, 0, float(np.nanpercentile(sp, 99.5))), interpolation='bilinear')
                ax.streamplot(y, z, np.where(sol, 0, a[0] * uf).T, np.where(sol, 0, a[1] * uf).T, density=1.3, color=BLACK, linewidth=0.4, arrowsize=0.5)
                ax.set_title(f'$\\varphi={cen[i]:.0f}^\\circ$', pad=2)
            else:
                im1 = ax.imshow(np.where(sol, np.nan, pp).T, origin='lower', extent=ext, cmap='RdBu_r', norm=TwoSlopeNorm(0, -pl, pl), interpolation='bilinear')
            ax.imshow(solid_rgba(sol), origin='lower', extent=ext); ax.set_xlim(*ylim); ax.set_ylim(*zlim); ax.set_aspect('equal')
            if col > 0: ax.set_yticklabels([])
            else: ax.set_ylabel('$z$ [mm]')
            if row == 0: ax.set_xticklabels([])
            else: ax.set_xlabel('$y$ [mm]')
    cb = fig.colorbar(im0, cax=fig.add_subplot(gs[0, 4])); cb.set_label('$|\\mathbf{u}|$ [m/s]')
    cb = fig.colorbar(im1, cax=fig.add_subplot(gs[1, 4])); cb.set_label('$p-\\bar p$ [Pa]')
    save(fig, 'fig_phase')

# ---------------------------------------------------------------- 6. vortices (fine grid, end of long run)
def fig_vortices():
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage import measure
    g = np.load(P(A.main, 'geom.npz')); solid = g['orig']; x, y, z = g['x'], g['y'], g['z']; dx = float(g['dx']); uf = 343 / np.sqrt(1 / 3)
    u = np.load(P(A.main, 'snap_orig.npy')).astype(np.float32)[:3] * uf
    for a in u: a[solid] = 0
    Q = np.zeros_like(u[0])
    G = [np.gradient(a, dx * 1e-3) for a in u]
    for i in range(3):
        for j in range(3):
            Sij = 0.5 * (G[i][j] + G[j][i]); Oij = 0.5 * (G[i][j] - G[j][i]); Q += 0.5 * (Oij * Oij - Sij * Sij)
    del G; Q[solid] = 0; Q = ndi.gaussian_filter(Q, 0.6 * 0.5 / dx); sp = np.sqrt((u ** 2).sum(0))
    thr = np.percentile(Q[~solid & (sp > 3)], 94); NUM['vortex_Q_threshold'] = float('%.3g' % thr)
    v, f, _, _ = measure.marching_cubes(Q, thr, spacing=(dx,) * 3); v += np.array([x[0], y[0], z[0]])
    cen = v[f].mean(1); keep = (cen[:, 1] > -16) & (cen[:, 1] < 30) & (np.abs(cen[:, 0]) < 15) & (cen[:, 2] > 6) & (cen[:, 2] < 37)
    idx = np.clip(np.round((cen - [x[0], y[0], z[0]]) / dx).astype(int), 0, np.array(sp.shape) - 1); col = sp[idx[:, 0], idx[:, 1], idx[:, 2]]
    Rr = np.sqrt((x[:, None, None] - SPHERE_C[0]) ** 2 + (y[None, :, None] - SPHERE_C[1]) ** 2 + (z[None, None, :] - SPHERE_C[2]) ** 2)
    shell = solid & (Rr < 12.4) & (Rr > 10.6)
    vs, fs, _, _ = measure.marching_cubes(ndi.gaussian_filter(shell.astype(np.float32), 0.8 * 0.5 / dx), 0.4, spacing=(dx,) * 3); vs += np.array([x[0], y[0], z[0]]); fs = fs[vs[fs].mean(1)[:, 0] < 0]
    vmax = float(np.percentile(col[keep], 98)) if keep.any() else 50
    fig = plt.figure(figsize=(COL, 2.3)); ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(Poly3DCollection(vs[fs], facecolor=(0.55, 0.55, 0.55, 0.16), edgecolor='none', rasterized=True))
    ax.add_collection3d(Poly3DCollection(v[f][keep], facecolors=BLUES(0.3 + 0.7 * np.clip(col[keep] / vmax, 0, 1)), edgecolor='none', alpha=0.92, rasterized=True))
    ax.set_xlim(-15, 15); ax.set_ylim(-16, 30); ax.set_zlim(6, 37); ax.set_box_aspect((30, 46, 31)); ax.view_init(12, 30)
    ax.set_xlabel('$x$ [mm]', labelpad=-6); ax.set_ylabel('$y$ [mm]', labelpad=-4); ax.set_zlabel('$z$ [mm]', labelpad=-6); ax.tick_params(pad=-3, labelsize=6)
    ax.text2D(0.70, 0.62, 'jet from duct', transform=ax.transAxes, fontsize=6.5)
    from matplotlib.colors import Normalize
    smap = plt.cm.ScalarMappable(cmap=LinearSegmentedColormap.from_list('b2', [BLUES(0.3), BLUES(1.0)]), norm=Normalize(0, vmax))
    cb = fig.colorbar(smap, ax=ax, shrink=0.6, pad=0.06); cb.set_label('$|\\mathbf{u}|$ [m/s]')
    save(fig, 'fig_vortices')

# ---------------------------------------------------------------- 7. grid convergence (5-20 ms)
def fig_convergence():
    fig, axs = plt.subplots(1, 2, figsize=(WIDE, 2.3), sharey=True, gridspec_kw={'wspace': 0.08}); NUM['convergence'] = {}
    for k, c in enumerate(['orig', 'sealed']):
        rows = []
        for dx in (1.0, 0.5, 0.25):
            run = f"dx{dx:g}_dp2000"
            if not exists(run, c): continue
            r = load(run, c); h = r['hist'].astype(float); dt = float(r['dt']); i0 = int(5e-3 / dt); i1 = int(20e-3 / dt)
            x = h[i0:i1, 4] - h[i0:i1, 4].mean(); f, Pw = welch(x, fs=1 / dt, nperseg=int(7.5e-3 / dt)); L = 10 * np.log10(Pw * (f[1] - f[0]) / (2e-5) ** 2 + 1e-12)
            axs[k].plot(f / 1e3, L, label=f'$\\Delta x$ = {dx:g} mm', **GRID_STYLE[dx]); rows.append(summary(r, 5e-3, 20e-3))
        NUM['convergence'][c] = rows
        axs[k].set_xlim(0, 8); axs[k].set_ylim(10, 115); axs[k].set_xlabel('$f$ [kHz]'); axs[k].set_title(f"({'ab'[k]}) {CASE_LAB[c]}", loc='left', pad=2); axs[k].grid(alpha=0.2, lw=0.4)
    axs[0].set_ylabel('listener SPL per band [dB]'); axs[0].legend(frameon=False, fontsize=6.5, loc='upper right')
    save(fig, 'fig_convergence')

# ---------------------------------------------------------------- 8. blowing-pressure sweep
def fig_sweep():
    def get(run, t1=None):
        if not exists(run): return None
        return summary(load(run), 5e-3, t1)
    s05 = {p: get(f'dx0.5_dp{p}') for p in (500, 1000, 1500, 2000, 3000, 4000)}; s05 = {p: v for p, v in s05.items() if v}
    s025 = {p: get(f'dx0.25_dp{p}', 20e-3) for p in (2000, 4000)}; s025 = {p: v for p, v in s025.items() if v}
    NUM['sweep_0p5mm'] = s05; NUM['sweep_0p25mm'] = s025
    fig, axs = plt.subplots(1, 2, figsize=(WIDE, 2.4), gridspec_kw={'wspace': 0.3})
    ax = axs[0]; ax.axhspan(1.7, 3.3, color=LIGHTBLUE, lw=0, label='two-chamber estimate'); ax.axhspan(1.2, 2.3, color='#e6e6e6', lw=0, alpha=0.8, label='single-sphere estimate')
    ok = [p for p in s05 if 1000 <= p <= 2000]; weak = [p for p in s05 if p < 1000]; bad = [p for p in s05 if p > 2000]
    U = lambda d, ps: np.array([d[p]['Uj_mps'] for p in ps]); F = lambda d, ps: np.array([d[p]['f0_listener_Hz'] for p in ps]) / 1e3
    if ok: ax.plot(U(s05, ok), F(s05, ok), 'o', color=BLUE, ms=4, label='0.5 mm');
    if weak: ax.plot(U(s05, weak), F(s05, weak), 'o', mfc='white', mec=BLUE, ms=4, label='0.5 mm, no sustained tone')
    if bad: ax.plot(U(s05, bad), F(s05, bad), 'x', color=GREY, ms=4, label='0.5 mm, under-resolved')
    if s025: ax.plot(U(s025, list(s025)), F(s025, list(s025)), 's', color=RED, ms=4, label='0.25 mm')
    if 1000 in s05:
        uu = np.linspace(20, 90, 50); ax.plot(uu, s05[1000]['f0_listener_Hz'] / 1e3 * uu / s05[1000]['Uj_mps'], '--', color=BLACK, lw=0.8, label='$f\\propto U_j$ (jet scaling)')
    ax.set_xlim(20, 90); ax.set_ylim(0, 4); ax.set_xlabel('jet speed $U_j=\\sqrt{2\\Delta p/\\rho}$ [m/s]'); ax.set_ylabel('$f_0$ at listener [kHz]'); ax.legend(frameon=False, fontsize=5.8, loc='upper left'); label(ax, '(a)')
    ax = axs[1]; ps = np.array(sorted(s05)); q = np.array([s05[p]['Q_Lps'] for p in ps])
    ax.plot(ps / 1e3, q, 'o', color=BLUE, ms=4, label='0.5 mm')
    if s025: ax.plot(np.array(list(s025)) / 1e3, [v['Q_Lps'] for v in s025.values()], 's', color=RED, ms=4, label='0.25 mm')
    fit = (ps >= 1000) & (ps <= 2000)
    if fit.any():
        kq = float(np.sum(q[fit] * np.sqrt(ps[fit])) / np.sum(ps[fit])); pp = np.linspace(0, 4200, 60)
        ax.plot(pp / 1e3, kq * np.sqrt(pp), '--', color=BLACK, lw=0.8, label='$Q\\propto\\sqrt{\\Delta p}$'); NUM['orifice_k_Lps_per_sqrtPa'] = round(kq, 5)
    ax.set_xlabel('blowing pressure $\\Delta p$ [kPa]'); ax.set_ylabel('flow $Q$ [L/s]'); ax.set_xlim(0, 4.3); ax.set_ylim(0, None); ax.legend(frameon=False, fontsize=6.5, loc='upper left'); label(ax, '(b)')
    for a in axs: a.grid(alpha=0.2, lw=0.4)
    save(fig, 'fig_sweep')

# ---------------------------------------------------------------- 9. roughness and the two-band spectrum (long runs)
def rough(r, t0=20e-3):
    h = r['hist'].astype(float); dt = float(r['dt']); x = h[int(t0 / dt):, 4]; x = x - x.mean(); xb = band(x, dt, 300, 8000)
    env = np.abs(hilbert(xb)); m = env.mean(); e = env - m
    E = np.abs(np.fft.rfft(e * np.hanning(len(e)), n=8 * len(e))) ** 2; fe = np.fft.rfftfreq(8 * len(e), dt); se_ = (fe >= 5) & (fe <= 1000)
    n = 8 * len(xb); Pp = np.abs(np.fft.rfft(xb * np.hanning(len(xb)), n=n)) ** 2; f = np.fft.rfftfreq(n, dt); s = (f > 300) & (f < 8000); f0 = f[s][np.argmax(Pp[s])]
    tonem = np.zeros_like(f, bool)
    for k in range(1, int(8000 // f0) + 1): tonem |= np.abs(f - k * f0) <= 0.04 * k * f0
    ed = band(e, dt, 5, 1000).std() / m; rd = band(e, dt, 30, 150).std() / m
    return dict(f0_Hz=round(float(f0)), tonal_frac=round(float(Pp[s & tonem].sum() / Pp[s].sum()), 3), env_depth=round(float(ed), 3),
                rough_depth=round(float(rd), 3), rough_frac=round(float((rd / ed) ** 2), 3), fmod_peak_Hz=round(float(fe[se_][np.argmax(E[se_])])),
                window_ms=round(len(x) * dt * 1e3, 1)), xb, dt, fe[se_], E[se_] / E[se_].max()

def fig_roughness():
    runs = [os.path.basename(p) for p in sorted(glob.glob(P('dx*_dp*_long')), key=lambda s: (-float(re.search(r'dx([\d.]+)', s).group(1)), int(re.search(r'dp(\d+)', s).group(1))))]
    runs = [r for r in runs if long_enough(r)]; NUM['roughness'] = {}
    fig, ax = plt.subplots(len(runs), 2, figsize=(WIDE, 1.55 * len(runs) + 0.3), squeeze=False, gridspec_kw={'width_ratios': [1.25, 1], 'hspace': 0.65, 'wspace': 0.22})
    for i, run in enumerate(runs):
        r = load(run); d, xb, dt, fe, E = rough(r); NUM['roughness'][run] = d
        nper = int(4e-3 / dt); fs_, ts_, S = spectrogram(xb, fs=1 / dt, nperseg=nper, noverlap=nper * 3 // 4); keep = fs_ <= 6000
        Ld = 10 * np.log10(S[keep] + 1e-12); ax[i, 0].pcolormesh((ts_ + 20e-3) * 1e3, fs_[keep] / 1e3, Ld, shading='auto', cmap='magma', vmin=Ld.max() - 45, vmax=Ld.max(), rasterized=True)
        ax[i, 0].set_ylabel('$f$ [kHz]'); ax[i, 0].set_title(f"$\\Delta x$ = {dxof(r):g} mm, $\\Delta p$ = {float(r['dp'])/1e3:g} kPa", loc='left', pad=2)
        ax[i, 1].axvspan(30, 150, color=LIGHTRED, lw=0); ax[i, 1].plot(fe, E, color=BLACK, lw=0.8); ax[i, 1].set_xscale('log'); ax[i, 1].set_xlim(5, 1000); ax[i, 1].set_ylim(0, 1.05)
        ax[i, 1].set_ylabel('envelope power'); ax[i, 1].text(0.02, 0.9, f"roughness depth {d['rough_depth']:.2f}", transform=ax[i, 1].transAxes, ha='left', fontsize=6.5, color=RED)
    ax[-1, 0].set_xlabel('$t$ [ms]'); ax[-1, 1].set_xlabel('modulation rate [Hz]'); ax[0, 1].text(30, 1.07, '30-150 Hz', fontsize=6.5, color=RED)
    save(fig, 'fig_roughness')
    # two-band spectrum at 2 kPa
    fig, axx = plt.subplots(figsize=(COL, 1.9)); NUM['twoband'] = {}
    for run, sty in [('dx0.5_dp2000_long', dict(color=BLUE, lw=0.8)), ('dx0.25_dp2000_long', dict(color=RED, lw=0.8)), ('dx0.25_dp1500_long', dict(color=RED, lw=0.8, ls='--'))]:
        if not long_enough(run): continue
        r = load(run); h = r['hist'].astype(float); dt = float(r['dt']); x = h[int(20e-3 / dt):, 4]; x = x - x.mean()
        f, Pw = welch(x, fs=1 / dt, nperseg=int(20e-3 / dt)); L = 10 * np.log10(Pw * (f[1] - f[0]) / (2e-5) ** 2 + 1e-12)
        axx.plot(f / 1e3, L, label=f"{dxof(r):g} mm, {float(r['dp'])/1e3:g} kPa", **sty)
        s = (f > 500) & (f < 5000); ipk = np.argsort(L[s])[-1]; NUM['twoband'][run] = round(float(f[s][ipk]))
    axx.set_ylim(50, 110)
    for fm_, lab_ in [(1.156, '$f_0/2$'), (2.31, '$f_0$')]: axx.axvline(fm_, color=BLACK, ls='--', lw=0.6); axx.text(fm_ + 0.04, 106, lab_, fontsize=6.5)
    axx.set_xlim(0.5, 5); axx.set_xlabel('$f$ [kHz]'); axx.set_ylabel('listener SPL per band [dB]'); axx.legend(frameon=False, fontsize=6.5, loc='lower left'); axx.grid(alpha=0.2, lw=0.4)
    save(fig, 'fig_twoband')

for name, fn in [('geometry', fig_geometry), ('setup', fig_setup), ('signals', fig_signals), ('mean', fig_mean), ('phase', fig_phase),
                 ('vortices', fig_vortices), ('convergence', fig_convergence), ('sweep', fig_sweep), ('roughness', fig_roughness)]:
    run_fig(name, fn)
with open(f'{A.out}/numbers.json', 'w') as fh: json.dump(NUM, fh, indent=1, default=float)
print('numbers written to', f'{A.out}/numbers.json'); print(json.dumps(NUM, indent=1, default=float)[:6000])
