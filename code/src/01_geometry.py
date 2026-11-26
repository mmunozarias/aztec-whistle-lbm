"""Step 1: voxelise the whistle STL and build the two fluid geometries.

Produces <workdir>/geom.npz with
  orig    : solid mask of the whistle as published (+ lip cap that seals the mouthpiece)
  sealed  : same, with the three base slots filled in
  inlet   : the pressure-inlet cells just inside the mouthpiece
Coordinates are those of the STL in the Thingiverse download (thing 5148690,
file _BothDeathwhistlefile98.stl, millimetres, printed orientation).
The cap line, seal region and sphere centre below are specific to that file.
"""
import argparse, time, numpy as np, trimesh, igl
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument('--stl', required=True)
ap.add_argument('--dx', type=float, default=0.5, help='grid spacing [mm]')
ap.add_argument('--workdir', default='out')
a = ap.parse_args()
import os; os.makedirs(a.workdir, exist_ok=True)

m = trimesh.load(a.stl, force='mesh')
dx = a.dx
x = np.arange(-30, 30 + 1e-9, dx); y = np.arange(-62, 62 + 1e-9, dx); z = np.arange(-12, 64 + 1e-9, dx)
G = np.stack(np.meshgrid(x, y, z, indexing='ij'), -1).reshape(-1, 3)
t = time.time()
# generalised winding number: robust to the small cracks in this mesh
w = igl.fast_winding_number(np.asarray(m.vertices, float), np.asarray(m.faces, np.int64), G)
solid = (w > 0.5).reshape(len(x), len(y), len(z)); del G, w
print(f'voxelised {solid.shape} ({solid.size/1e6:.1f} M cells) in {time.time()-t:.0f}s')

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
# lip cap across the mouthpiece opening (a line in the y-z plane, extruded in x)
A = np.array([52.0, 27.2]); B = np.array([37.0, 12.5])
tv = (B - A); L = np.linalg.norm(tv); tv /= L
n = np.array([-tv[1], tv[0]]); n = -n if n[0] > 0 else n           # points into the bore
s = (Y - A[0]) * tv[0] + (Z - A[1]) * tv[1]; dn = (Y - A[0]) * n[0] + (Z - A[1]) * n[1]
cap = (s > -2) & (s < L + 2) & (dn > -1.5) & (dn < 0.7) & (np.abs(X) < 16)
base = solid | cap
slab = (s > 0) & (s < L) & (dn >= 0.7) & (dn < 0.7 + max(1.5, 3 * dx)) & (np.abs(X) < 16) & ~base
lab, _ = ndi.label(slab)
Mp = (A + B) / 2 + n * (0.7 + max(0.75, 1.5 * dx))
seed = (np.abs(x).argmin(), np.abs(y - Mp[0]).argmin(), np.abs(z - Mp[1]).argmin())
assert lab[seed] > 0, 'inlet seed not in bore: adjust A/B for your geometry'
inlet = lab == lab[seed]

# sealed variant: fill the region between the annular gap and the underside
c0 = np.array([-0.37, 0.08, 21.63])          # inner sphere centre (fitted)
R = np.sqrt((X - c0[0])**2 + (Y - c0[1])**2 + (Z - c0[2])**2)
seal = (R > 14.6) & (R < 21) & (Z < 11.5) & (np.abs(X) < 11) & (Y > -14) & (Y < 9) & ~base

np.savez_compressed(f'{a.workdir}/geom.npz', orig=base, sealed=base | seal, inlet=inlet, x=x, y=y, z=z, dx=dx)
print(f'inlet cells {inlet.sum()}, sealed cells {seal.sum()} -> {a.workdir}/geom.npz')
