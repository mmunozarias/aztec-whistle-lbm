"""Step 5: 3D vortex structures (Q-criterion) from the final snapshot."""
import argparse, os
ap = argparse.ArgumentParser()
ap.add_argument('--workdir', default='out')
ap.add_argument('--case', default='orig', choices=['orig','sealed'])
_a = ap.parse_args(); os.chdir(_a.workdir); os.makedirs('figs', exist_ok=True)
import numpy as np, matplotlib, sys, os
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from scipy import ndimage as ndi
case=_a.case
g=np.load('geom.npz'); solid=g[case]; x=g['x'];y=g['y'];z=g['z']; dx=float(g['dx']); ufac=343/np.sqrt(1/3)
s=np.load(f'snap_{case}.npy').astype(np.float32); u=s[:3]*ufac; del s
for a in u: a[solid]=0
G=[np.gradient(a,dx*1e-3) for a in u]  # G[i][j] = du_i/dx_j
Q=np.zeros_like(u[0])
for i in range(3):
    for j in range(3):
        S_=0.5*(G[i][j]+G[j][i]); O=0.5*(G[i][j]-G[j][i]); Q+=0.5*(O*O-S_*S_)
del G
Q[solid]=0; Q=ndi.gaussian_filter(Q,0.6)
sp=np.sqrt((u**2).sum(0))
thr=np.percentile(Q[~solid & (sp>3)],94); print('Q thr',thr)
v,f,_,_=measure.marching_cubes(Q,thr,spacing=(dx,dx,dx)); v+=np.array([x[0],y[0],z[0]])
cen=v[f].mean(1); idx=np.round((cen-[x[0],y[0],z[0]])/dx).astype(int); col=sp[idx[:,0],idx[:,1],idx[:,2]]
keep=(cen[:,1]>-16)&(cen[:,1]<30)&(np.abs(cen[:,0])<15)&(cen[:,2]>6)&(cen[:,2]<37)
# resonator shell for context
c0=np.array([-0.37,0.08,21.63]); R=np.sqrt((x[:,None,None]-c0[0])**2+(y[None,:,None]-c0[1])**2+(z[None,None,:]-c0[2])**2)
shell=solid&(R<12.4)&(R>10.6)
vs,fs,_,_=measure.marching_cubes(ndi.gaussian_filter(shell.astype(np.float32),0.8),0.4,spacing=(dx,dx,dx)); vs+=np.array([x[0],y[0],z[0]])
cs=vs[fs].mean(1); fs=fs[cs[:,0]<0]  # cut away the +x half to look inside
fig=plt.figure(figsize=(17,8),facecolor='white')
for k,(el,az,tt) in enumerate([(22,-60,'from the front-left'),(12,30,'from the mouthpiece side')]):
    ax=fig.add_subplot(1,2,k+1,projection='3d')
    ax.add_collection3d(Poly3DCollection(vs[fs],facecolor=(0.6,0.58,0.55,0.18),edgecolor='none'))
    tri=v[f][keep]; cc=col[keep]
    pc=Poly3DCollection(tri,facecolors=plt.cm.plasma(np.clip(cc/50,0,1)),edgecolor='none',alpha=0.9); ax.add_collection3d(pc)
    ax.set_xlim(-15,15); ax.set_ylim(-16,30); ax.set_zlim(6,37); ax.set_box_aspect((30,46,31)); ax.view_init(el,az)
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]'); ax.set_zlabel('z [mm]'); ax.set_title(tt,fontsize=11)
sm=plt.cm.ScalarMappable(cmap='plasma',norm=plt.Normalize(0,50)); fig.colorbar(sm,ax=fig.axes,shrink=0.5,label='local air speed [m/s]')
fig.suptitle(f'Vortex structures at the end of the run ({"original" if case=="orig" else "slots sealed"}), Q-criterion. Grey: inner resonator shell, cut in half',x=0.02,ha='left',fontsize=13)
fig.savefig(f'figs/fig4_vortices_3d_{case}.png',dpi=110,bbox_inches='tight'); print(len(f),keep.sum())
