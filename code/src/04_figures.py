"""Step 4: analysis figures (spectrum, mean flow, sound sources) and summary numbers."""
import argparse, os
ap = argparse.ArgumentParser()
ap.add_argument('--workdir', default='out')

_a = ap.parse_args(); os.chdir(_a.workdir); os.makedirs('figs', exist_ok=True)
import numpy as np, matplotlib, sys, os
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.signal import welch
C={'orig':'#2a78d6','sealed':'#eb6834'}; LAB={'orig':'Original (slots open)','sealed':'Base slots sealed'}
plt.rcParams.update({'font.size':10,'axes.edgecolor':'#888','axes.labelcolor':'#333','xtick.color':'#555','ytick.color':'#555','axes.spines.top':False,'axes.spines.right':False})
C={c:C[c] for c in C if os.path.exists(f'res_{c}.npz')}
R={c:np.load(f'res_{c}.npz') for c in C}
r0=R[list(R)[0]]; dt=float(r0['dt']); pfac=float(r0['pfac']); ufac=float(r0['ufac']); y=r0['y'];z=r0['z'];x=r0['x']; se=int(r0['save_every'])
T0=min(3e-3, 0.5*len(r0['hist'])*dt)   # discard start-up transient
Tend=len(r0['hist'])*dt*1e3
# Fig 1: spectra + waveform
fig,ax=plt.subplots(1,2,figsize=(14,5),gridspec_kw={'width_ratios':[1.3,1]})
for c in C:
    h=R[c]['hist']; i=int(T0/dt)
    f,Pxx=welch(h[i:,4]-h[i:,4].mean(),fs=1/dt,nperseg=min(int(3.5e-3/dt),len(h)-i))
    spl=10*np.log10(Pxx*(f[1]-f[0])/(2e-5)**2+1e-12)
    ax[0].plot(f/1000,spl,color=C[c],lw=2,label=LAB[c])
    xx=h[i:,4]-h[i:,4].mean(); FF=np.abs(np.fft.rfft(xx*np.hanning(len(xx)),n=16*len(xx))); ff=np.fft.rfftfreq(16*len(xx),dt); sel=(ff>500)&(ff<8000); fpk=ff[sel][np.argmax(FF[sel])]
    k=np.argmax(np.where((f>500)&(f<8000),spl,-99)); ax[0].annotate(f'{LAB[c].split(" (")[0]}: {fpk:.0f} Hz',(f[k]/1000,spl[k]),(f[k]/1000+0.9,spl[k]+(2 if c=='orig' else -9)),color='#333',arrowprops=dict(arrowstyle='-',color='#999'))
    t=np.arange(len(h))*dt*1e3; m=(t>Tend-3)&(t<Tend-1); ax[1].plot(t[m],h[m,0],color=C[c],lw=2,label=LAB[c])
ax[0].set_xlim(0,10); ax[0].set_xlabel('frequency [kHz]'); ax[0].set_ylabel('sound pressure level per band [dB re 20 µPa]')
ax[0].set_title('Sound spectrum just in front of the mouth (after start-up)',loc='left',fontsize=12); ax[0].legend(frameon=False); ax[0].grid(alpha=.25)
ax[1].set_xlabel('time [ms]'); ax[1].set_ylabel('pressure above ambient [Pa]'); ax[1].set_title('Pressure inside the inner resonator',loc='left',fontsize=12); ax[1].grid(alpha=.25); ax[1].legend(frameon=False,loc='upper center',bbox_to_anchor=(0.5,-0.13),ncol=2)
fig.tight_layout(); fig.savefig('figs/fig1_sound_spectrum.png',dpi=130)
# mean fields
def stats(c,key):
    a=R[c][key]; i=int(T0/(dt*se)); a=a[i:].astype(np.float32)
    mu=a[:,:3].mean(0)*ufac; sp=np.sqrt((a[:,:3]**2).sum(1)).mean(0)*ufac
    p=a[:,3]*pfac; prms=p.std(0); return mu,sp,prms
yl=(-55,56); zl=(-5,60); xl=(-27,27)
fig,ax=plt.subplots(len(C),2,figsize=(16,5.3*len(C)),gridspec_kw={'width_ratios':[2,1]},squeeze=False)
for row,c in enumerate(C):
    for col,(key,h1,h2,sol) in enumerate([('sx',y,z,'solid_x'),('sy',x,z,'solid_y')]):
        mu,sp,_=stats(c,key); S=R[c][sol]; a=ax[row,col]
        spm=np.where(S,np.nan,sp)
        im=a.imshow(spm.T,origin='lower',extent=[h1[0],h1[-1],z[0],z[-1]],cmap='Blues',vmin=0,vmax=45,interpolation='bilinear')
        g=np.zeros(S.shape+(4,)); g[S]=[0.55,0.54,0.5,1]; a.imshow(g.transpose(1,0,2),origin='lower',extent=[h1[0],h1[-1],z[0],z[-1]])
        U=np.where(S,0,mu[0]).T; V=np.where(S,0,mu[1]).T
        a.streamplot(h1,z,U,V,density=2.2,color='#0d366b',linewidth=0.6,arrowsize=0.7,broken_streamlines=True)
        a.set_xlim(*(yl if col==0 else xl)); a.set_ylim(*zl); a.set_aspect('equal')
        a.set_title(f"{LAB[c]}: {'side cut (x = 0)' if col==0 else 'cross-cut (y = 0)'}",loc='left',fontsize=11)
        a.set_xlabel('y [mm]' if col==0 else 'x [mm]'); a.set_ylabel('z [mm]')
fig.colorbar(im,ax=ax,shrink=0.6,label='time-averaged air speed [m/s]')
fig.suptitle('Time-averaged airflow (after start-up). Lines follow the average flow direction in the cut plane.',x=0.02,ha='left',fontsize=13)
fig.savefig('figs/fig2_mean_flow.png',dpi=110,bbox_inches='tight')
# sound sources: p' rms
fig,ax=plt.subplots(1,len(C),figsize=(8*len(C),5.6),squeeze=False); ax=ax[0]
for a,c in zip(ax,C):
    _,_,pr=stats(c,'sx'); S=R[c]['solid_x']; L=20*np.log10(np.maximum(pr,1e-3)/2e-5)
    im=a.imshow(np.where(S,np.nan,L).T,origin='lower',extent=[y[0],y[-1],z[0],z[-1]],cmap='Oranges',vmin=120,vmax=165,interpolation='bilinear')
    g=np.zeros(S.shape+(4,)); g[S]=[0.55,0.54,0.5,1]; a.imshow(g.transpose(1,0,2),origin='lower',extent=[y[0],y[-1],z[0],z[-1]])
    a.set_xlim(*yl); a.set_ylim(*zl); a.set_title(LAB[c],loc='left'); a.set_xlabel('y [mm]'); a.set_ylabel('z [mm]')
fig.colorbar(im,ax=ax,shrink=0.85,label='pressure fluctuation level [dB re 20 µPa]')
fig.suptitle('Where the sound is made: strength of the pressure fluctuations (mid-plane, after start-up)',x=0.02,ha='left',fontsize=13)
fig.savefig('figs/fig3_sound_sources.png',dpi=110,bbox_inches='tight')
for c in C:
    h=R[c]['hist']; i=int(T0/dt); x_=h[i:,0]-h[i:,0].mean(); FF=np.abs(np.fft.rfft(x_*np.hanning(len(x_)),n=16*len(x_))); ff=np.fft.rfftfreq(16*len(x_),dt); sel=(ff>500)&(ff<8000)
    lis=h[i:,4]-h[i:,4].mean(); FL=np.fft.rfft(lis); fl=np.fft.rfftfreq(len(lis),dt); FL[(fl<300)|(fl>8000)]=0; lb=np.fft.irfft(FL,len(lis))
    print(f"{c}: tone {ff[sel][np.argmax(FF[sel])]:.0f} Hz (resolution ~{1/((len(h)-i)*dt):.0f} Hz), listener {20*np.log10(lb.std()/2e-5):.1f} dB, flow {h[i:,-1].mean():.3f} L/s")
