"""Step 2: 3D lattice Boltzmann LES of the whistle (D3Q19).

Model
  * regularised BGK collision + Smagorinsky sub-grid model (Cs = 0.17)
    (--collision bgk and --cs switch these off, for the stability test of plain BGK)
  * molecular viscosity of air (1.5e-5 m2/s), speed of sound 343 m/s, rho 1.2 kg/m3
  * lattice speed of sound mapped to the real one, so Mach number and acoustics are physical
  * pressure inlet inside the sealed mouthpiece, ramped to --dp over 0.5 ms
  * half-way bounce-back walls; sponge layer at the box edges absorbs outgoing sound
Output <workdir>/res_<case>.npz: probe histories every step, mid-plane (x=0) and
cross-plane (y=0) slices every ~21 microseconds, and snap_<case>.npy (final 3D field).
Parallel via numba: set NUMBA_NUM_THREADS to the cores you want.
"""
import argparse, time, numpy as np, numba as nb
ap = argparse.ArgumentParser()
ap.add_argument('--case', default='orig', choices=['orig', 'sealed'])
ap.add_argument('--time', type=float, default=10e-3, help='simulated time [s]')
ap.add_argument('--dp', type=float, default=2000.0, help='blowing pressure [Pa]')
ap.add_argument('--workdir', default='out')
ap.add_argument('--collision', default='reg', choices=['reg', 'bgk'],
                help='reg: regularized BGK (all runs of the paper); bgk: plain BGK, for the stability test')
ap.add_argument('--cs', type=float, default=0.17, help='Smagorinsky constant; 0 switches the eddy viscosity off')
a = ap.parse_args()

g = np.load(f'{a.workdir}/geom.npz'); solid = np.ascontiguousarray(g[a.case]); inlet = g['inlet']
x, y, z = g['x'], g['y'], g['z']; dx = float(g['dx'])
c_phys, rho0, cs, nu_air = 343.0, 1.2, 1 / np.sqrt(3), 1.5e-5
dt = dx * 1e-3 * cs / c_phys; ufac = c_phys / cs; pfac = rho0 * c_phys**2
tau0 = 0.5 + 3 * nu_air * dt / (dx * 1e-3)**2; Csm = a.cs; BGK = a.collision == 'bgk'
drho = a.dp / pfac; nsteps = int(a.time / dt); ramp = int(0.5e-3 / dt); save_every = max(1, round(21e-6 / dt))
print(f'{a.case}: grid {solid.shape}, dt={dt:.3e} s, {nsteps} steps, tau0={tau0:.6f}, collision={a.collision}, Cs={Csm}', flush=True)

c = np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[1,1,0],[-1,-1,0],[1,-1,0],[-1,1,0],
              [1,0,1],[-1,0,-1],[1,0,-1],[-1,0,1],[0,1,1],[0,-1,-1],[0,1,-1],[0,-1,1]])
w = np.array([1/3] + [1/18]*6 + [1/36]*12); opp = np.array([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17])
cx, cy, cz = c[:, 0].copy(), c[:, 1].copy(), c[:, 2].copy()
# Sponge layer, defined in physical units so that only the grid changes between resolutions.
# Reference: the paper's 0.5 mm runs (12 cells = 6 mm, peak strength 1, extra tau 0.5 * strength).
# Thickness is kept at 6 mm. The per-step relaxation strength scales with dx (same damping rate per
# unit physical time, since dt is proportional to dx) and the added relaxation time scales so that the
# added physical viscosity is also the same. At dx = 0.5 mm this reproduces the original layer exactly.
SPONGE_MM, DX_REF = 6.0, 0.5
sp = max(1, int(round(SPONGE_MM / dx))); I, J, K = np.indices(solid.shape)
dist = np.minimum.reduce([I, solid.shape[0]-1-I, J, solid.shape[1]-1-J, K, solid.shape[2]-1-K]).astype(np.float32)
sponge = np.minimum(1.0, (np.clip((sp - dist) / sp, 0, 1)**2) * (dx / DX_REF)).astype(np.float32); del I, J, K, dist
sp_visc = np.float32(0.5 * (DX_REF / dx)**2)     # extra tau = sp_visc * sponge (= 0.5 * profile at dx = 0.5 mm)
print(f'sponge: {sp} cells = {sp*dx:.1f} mm, peak strength {sponge.max():.3f}, extra-tau factor {sp_visc:.3f}', flush=True)

@nb.njit(parallel=True, fastmath=True, cache=True)
def step(f, fn, solid, inlet, sponge, sp_visc, rho_in, rho, ux, uy, uz, tau0, Csm, bgk):
    NX, NY, NZ = solid.shape
    for i in nb.prange(NX):
        fl = np.empty(19, np.float32); feq = np.empty(19, np.float32)
        for j in range(NY):
            for k in range(NZ):
                if solid[i, j, k]: continue
                for q in range(19):                                   # pull streaming + bounce-back
                    si = i - cx[q]; sj = j - cy[q]; sk = k - cz[q]
                    if si < 0 or sj < 0 or sk < 0 or si >= NX or sj >= NY or sk >= NZ: fl[q] = w[q]
                    elif solid[si, sj, sk]: fl[q] = f[opp[q], i, j, k]
                    else: fl[q] = f[q, si, sj, sk]
                r = 0.0; mx = 0.0; my = 0.0; mz = 0.0
                for q in range(19):
                    r += fl[q]; mx += fl[q]*cx[q]; my += fl[q]*cy[q]; mz += fl[q]*cz[q]
                u = mx / r; v = my / r; ww = mz / r
                isin = inlet[i, j, k]
                if isin: r = rho_in
                sg = sponge[i, j, k]
                if sg > 0:
                    r = r*(1-sg) + sg; u *= (1-0.5*sg); v *= (1-0.5*sg); ww *= (1-0.5*sg)
                uu = u*u + v*v + ww*ww
                Pxx = 0.0; Pyy = 0.0; Pzz = 0.0; Pxy = 0.0; Pxz = 0.0; Pyz = 0.0
                for q in range(19):
                    cu = cx[q]*u + cy[q]*v + cz[q]*ww
                    feq[q] = w[q]*r*(1 + 3*cu + 4.5*cu*cu - 1.5*uu)
                    fq = fl[q] - feq[q]
                    Pxx += cx[q]*cx[q]*fq; Pyy += cy[q]*cy[q]*fq; Pzz += cz[q]*cz[q]*fq
                    Pxy += cx[q]*cy[q]*fq; Pxz += cx[q]*cz[q]*fq; Pyz += cy[q]*cz[q]*fq
                Q = np.sqrt(Pxx*Pxx + Pyy*Pyy + Pzz*Pzz + 2*(Pxy*Pxy + Pxz*Pxz + Pyz*Pyz))
                te = 0.5*(tau0 + np.sqrt(tau0*tau0 + 18*np.sqrt(2.0)*Csm*Csm*Q/r)) + sg*sp_visc   # Smagorinsky + sponge viscosity
                om = 1 - 1/te
                if isin: om = 0.0
                if bgk:
                    for q in range(19):                               # plain BGK collision (stability test only)
                        fn[q, i, j, k] = feq[q] + om*(fl[q] - feq[q])
                else:
                    for q in range(19):                               # regularised collision
                        aa = cx[q]*cx[q]-1/3; bb = cy[q]*cy[q]-1/3; cc = cz[q]*cz[q]-1/3
                        f1 = w[q]*4.5*(aa*Pxx + bb*Pyy + cc*Pzz + 2*(cx[q]*cy[q]*Pxy + cx[q]*cz[q]*Pxz + cy[q]*cz[q]*Pyz))
                        fn[q, i, j, k] = feq[q] + om*f1
                rho[i, j, k] = r; ux[i, j, k] = u; uy[i, j, k] = v; uz[i, j, k] = ww

f = np.empty((19,) + solid.shape, np.float32); f[:] = w[:, None, None, None]; fn = f.copy()
rho = np.ones(solid.shape, np.float32); ux = np.zeros_like(rho); uy = np.zeros_like(rho); uz = np.zeros_like(rho)
probes = {'chamber': (-0.4, 0.1, 21.7), 'throat': (0, 17, 18.5), 'gap_top': (-0.4, 0, 35.3),
          'mouth_exit': (0, -20, 24), 'listener': (0, -54, 20)}
P = {k: tuple(int(round((p[i] - [x, y, z][i][0]) / dx)) for i in range(3)) for k, p in probes.items()}
# flow-rate monitor: one cross-section of the converging duct, perpendicular to its axis
from scipy import ndimage as _ndi
_ax = np.array([0.04, 0.94, 0.33]); _ax /= np.linalg.norm(_ax); _p0 = np.array([0.1, 29.1, 22.5]); _s0 = -6.0
_X, _Y, _Z = np.meshgrid(x, y, z, indexing='ij')
_sc = (_X - _p0[0]) * _ax[0] + (_Y - _p0[1]) * _ax[1] + (_Z - _p0[2]) * _ax[2]
_perp = np.sqrt(((_X - _p0[0]) - _sc * _ax[0])**2 + ((_Y - _p0[1]) - _sc * _ax[1])**2 + ((_Z - _p0[2]) - _sc * _ax[2])**2)
_lab, _ = _ndi.label((~solid) & (np.abs(_sc - _s0) < dx / 2) & (_perp < 9))
_c = _p0 + _ax * _s0; _k = _lab[tuple(int(round((_c[i] - [x, y, z][i][0]) / dx)) for i in range(3))]
ii = np.where(_lab == _k); del _X, _Y, _Z, _sc, _perp, _lab
nin = (-_ax).astype(np.float32)           # flow direction: from mouthpiece toward the sphere
hist = np.zeros((nsteps, len(P) + 1), np.float32); sx = []; sy = []
i0 = np.abs(x).argmin(); j0 = np.abs(y).argmin(); t0 = time.time()
def save(nmax):
    np.savez_compressed(f'{a.workdir}/res_{a.case}.npz', sx=np.array(sx), sy=np.array(sy), hist=hist[:nmax], dt=dt,
        save_every=save_every, pfac=pfac, ufac=ufac, solid_x=solid[i0], solid_y=solid[:, j0], inlet_x=inlet[i0],
        x=x, y=y, z=z, names=np.array(list(P.keys())), dp=a.dp)
for s_ in range(nsteps):
    rin = np.float32(1 + drho * min(1.0, s_ / ramp))
    step(f, fn, solid, inlet, sponge, sp_visc, rin, rho, ux, uy, uz, np.float32(tau0), np.float32(Csm), BGK); f, fn = fn, f
    for q, v in enumerate(P.values()): hist[s_, q] = (rho[v] - 1) * pfac          # Pa above ambient
    hist[s_, -1] = (ux[ii]*nin[0] + uy[ii]*nin[1] + uz[ii]*nin[2]).sum() * ufac * (dx*1e-3)**2 * 1e3  # duct flow [L/s]
    if s_ % save_every == 0:
        sx.append(np.stack([uy[i0], uz[i0], ux[i0], rho[i0] - 1]).astype(np.float16))
        sy.append(np.stack([ux[:, j0], uz[:, j0], uy[:, j0], rho[:, j0] - 1]).astype(np.float16))
    if s_ % 1000 == 0:
        print(f'step {s_}/{nsteps} t={s_*dt*1e3:.2f} ms  Q={hist[s_,-1]:.3f} L/s  p_chamber={hist[s_,0]:.0f} Pa  {time.time()-t0:.0f}s', flush=True)
        if not np.isfinite(rho).all(): raise SystemExit('diverged')
    if s_ % 5000 == 0: save(s_ + 1)
save(nsteps)
np.save(f'{a.workdir}/snap_{a.case}.npy', np.stack([ux, uy, uz, rho - 1]).astype(np.float16))
print(f'done in {time.time()-t0:.0f}s', flush=True)
