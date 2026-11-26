"""Step 3: render the comparison video (speed + pressure on the mid-plane, both cases)."""
import argparse, os
ap = argparse.ArgumentParser()
ap.add_argument('--workdir', default='out')
ap.add_argument('--out', default='death_whistle_3d_flow.mp4')
ap.add_argument('--max-frames', type=int, default=100000)
_a = ap.parse_args(); os.chdir(_a.workdir); os.makedirs('figs', exist_ok=True)
import numpy as np, matplotlib, subprocess, sys, os
matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import map_coordinates
cases=[c for c in ['orig','sealed'] if os.path.exists(f'res_{c}.npz')]
lab={'orig':'Original (base slots open)','sealed':'Base slots sealed'}
R={c:np.load(f'res_{c}.npz') for c in cases}
r0=R[cases[0]]; y=r0['y']; z=r0['z']; dx=float(y[1]-y[0]); dt=float(r0['dt']); se=int(r0['save_every']); pfac=float(r0['pfac']); ufac=float(r0['ufac'])
nfr=min(len(R[c]['sx']) for c in cases); nfr=min(nfr,_a.max_frames)
ylim=(-55,56); zlim=(-5,60); i0,i1=np.searchsorted(y,ylim); j0,j1=np.searchsorted(z,zlim)
sl=(slice(i0,i1),slice(j0,j1)); ext=[y[i0]-dx/2,y[i1-1]+dx/2,z[j0]-dx/2,z[j1-1]+dx/2]
fig=plt.figure(figsize=(19.2,10.8),dpi=100,facecolor='#111')
gs=fig.add_gridspec(3,5,height_ratios=[1,1,0.55],width_ratios=[1,0.02,0.06,1,0.02],hspace=0.3,wspace=0.03,left=0.035,right=0.955,top=0.9,bottom=0.06)
def style(a):
    a.set_facecolor('#111'); a.tick_params(colors='#bbb',labelsize=8)
    for s_ in a.spines.values(): s_.set_color('#555')
kw=dict(color='#fff',fontsize=10,ha='center',path_effects=[pe.withStroke(linewidth=3,foreground='#000')])
P={}
rng=np.random.default_rng(1)
for row,c in enumerate(cases):
    r=R[c]; sol=r['solid_x'][sl]; grey=np.zeros(sol.shape+(4,)); grey[sol]=[0.62,0.6,0.57,1]
    a1=fig.add_subplot(gs[row,0]); a2=fig.add_subplot(gs[row,3]); style(a1); style(a2)
    im1=a1.imshow(np.zeros(sol.shape).T,origin='lower',extent=ext,cmap='inferno',norm=matplotlib.colors.PowerNorm(0.5,vmin=0,vmax=60),interpolation='bilinear'); a1.imshow(grey.transpose(1,0,2),origin='lower',extent=ext)
    im2=a2.imshow(np.zeros(sol.shape).T,origin='lower',extent=ext,cmap='RdBu_r',norm=TwoSlopeNorm(vcenter=0,vmin=-500,vmax=1500),interpolation='bilinear'); a2.imshow(grey.transpose(1,0,2),origin='lower',extent=ext)
    sc=a1.scatter([],[],s=2.2,c='#7fe3ff',alpha=0.8,lw=0)
    a1.set_title(f'{lab[c]}: air speed + tracer particles',color='#eee',fontsize=12,loc='left'); a2.set_title(f'{lab[c]}: pressure',color='#eee',fontsize=12,loc='left')
    if row==0:
        a1.text(47,3,'mouthpiece',**kw); a1.text(-0.4,20,'inner\nresonator',**kw); a1.text(-42,6,'exit via\nmouth',**kw)
        if c=='orig': a1.text(3,-3,'base slot',**kw)
    cb=fig.colorbar(im1,cax=fig.add_subplot(gs[row,1])); cb.set_label('speed [m/s]',color='#ddd'); cb.ax.tick_params(colors='#bbb',labelsize=8)
    cb=fig.colorbar(im2,cax=fig.add_subplot(gs[row,4])); cb.set_label('pressure [Pa]',color='#ddd'); cb.ax.tick_params(colors='#bbb',labelsize=8)
    Yg,Zg=np.meshgrid(y,z,indexing='ij'); inl=np.argwhere((~r['solid_x'])&(Yg>24)&(Yg<30)&(Zg>17.5)&(Zg<25)); P[c]=dict(r=r,sol=r['solid_x'],im1=im1,im2=im2,sc=sc,pts=np.zeros((0,2)),age=np.zeros(0),inl=inl)
a3=fig.add_subplot(gs[2,:]); style(a3)
cols={'orig':'#f39c50','sealed':'#6fd08c'}
for c in cases:
    h=R[c]['hist']; t=np.arange(len(h))*dt*1e3; a3.plot(t,h[:,0],color=cols[c],lw=0.8,label=lab[c]+' (resonator centre)')
a3.set_xlim(0,nfr*se*dt*1e3); a3.set_ylabel('Pa',color='#ccc'); a3.set_xlabel('time [ms]',color='#ccc')
a3.legend(loc='upper right',facecolor='#222',edgecolor='#555',labelcolor='#eee',fontsize=9)
a3.set_title('Pressure inside the inner resonator',color='#eee',fontsize=12,loc='left')
cur=a3.axvline(0,color='#7fe3ff',lw=1.2)
ttl=fig.text(0.035,0.945,'',color='#fff',fontsize=17,ha='left')
fig.text(0.965,0.945,'3D LES simulation, mid-plane cut  |  2 kPa blowing pressure',color='#aaa',fontsize=11,ha='right')
W,H=fig.canvas.get_width_height(); out=_a.out
ff=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgba','-s',f'{W}x{H}','-r','30','-i','-','-c:v','libx264','-pix_fmt','yuv420p','-crf','20',out],stdin=subprocess.PIPE)
fdt=se*dt
def vel(u,p):
    gi=(p[:,0]-y[0])/dx; gj=(p[:,1]-z[0])/dx
    return np.stack([map_coordinates(u[0],[gi,gj],order=1),map_coordinates(u[1],[gi,gj],order=1)],1)
for k in range(nfr):
    for c in cases:
        d=P[c]; s=d['r']['sx'][k].astype(np.float32); u=s[:2]*ufac; sp=np.sqrt((s[0]**2+s[1]**2+s[2]**2))*ufac; p=s[3]*pfac
        nw=d['inl'][rng.integers(0,len(d['inl']),6)]; pn=np.c_[y[nw[:,0]],z[nw[:,1]]]+rng.uniform(-0.25,0.25,(6,2))
        pts=np.r_[d['pts'],pn]; age=np.r_[d['age'],np.zeros(6)]
        for sub in range(2):
            v1=vel(u,pts); pm=pts+v1*fdt/2*1e3*0.5; pts=pts+vel(u,pm)*fdt/2*1e3
        gi=np.round((pts[:,0]-y[0])/dx).astype(int); gj=np.round((pts[:,1]-z[0])/dx).astype(int)
        ok=(gi>=0)&(gi<len(y))&(gj>=0)&(gj<len(z)); ok[ok]&=~d['sol'][gi[ok],gj[ok]]
        ok&=(pts[:,0]>ylim[0])&(pts[:,0]<ylim[1])&(pts[:,1]>zlim[0])&(pts[:,1]<zlim[1]); age+=1; ok&=age<300
        d['pts']=pts[ok]; d['age']=age[ok]
        sol=d['sol'][sl]
        d['im1'].set_data(np.where(sol,np.nan,sp[sl]).T); d['im2'].set_data(np.where(sol,np.nan,p[sl]).T); d['sc'].set_offsets(d['pts'])
    t=k*fdt*1e3; cur.set_xdata([t,t])
    ttl.set_text(f'Aztec death whistle airflow     t = {t:5.2f} ms   (slowed down ~{1/(fdt*30):.0f}×)')
    fig.canvas.draw(); ff.stdin.write(fig.canvas.buffer_rgba().tobytes())
ff.stdin.close(); ff.wait(); print('ok',nfr)
