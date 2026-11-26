"""Step 6: audio clip from the 'listener' probe in front of the mouth.

The settled part of the signal (after start-up) is band-passed to 150-12000 Hz,
resampled to 48 kHz and looped with short crossfades to 2.5 s per case.
Both cases use the same gain, so their loudness can be compared by ear.
"""
import argparse, os, wave, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--workdir', default='out'); a = ap.parse_args(); os.chdir(a.workdir)
fs = 48000; raw = {}
for c in ['orig', 'sealed']:
    if not os.path.exists(f'res_{c}.npz'): continue
    r = np.load(f'res_{c}.npz'); h = r['hist']; dt = float(r['dt']); i = int(min(3e-3, 0.5 * len(h) * dt) / dt)
    x = h[i:, 4].astype(float); x -= x.mean()
    F = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), dt); F[(f < 150) | (f > 12000)] = 0; xb = np.fft.irfft(F, len(x))
    t = np.arange(len(x)) * dt; raw[c] = np.interp(np.arange(0, t[-1], 1 / fs), t, xb)
g = 0.8 / max(np.abs(v).max() for v in raw.values()); segs = []
for c, s in raw.items():
    s = s * g; cf = int(0.0015 * fs); out = s.copy(); win = np.linspace(0, 1, cf)
    while len(out) < int(2.5 * fs): out = np.r_[out[:-cf], out[-cf:] * (1 - win) + s[:cf] * win, s[cf:]]
    out = out[:int(2.5 * fs)]; fade = int(0.05 * fs); out[:fade] *= np.linspace(0, 1, fade); out[-fade:] *= np.linspace(1, 0, fade)
    segs += [out, np.zeros(int(0.6 * fs))]
pcm = (np.concatenate(segs) * 32767).astype(np.int16)
with wave.open('death_whistle_simulated_sound.wav', 'wb') as wv:
    wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(fs); wv.writeframes(pcm.tobytes())
print('wrote death_whistle_simulated_sound.wav, order:', list(raw))
