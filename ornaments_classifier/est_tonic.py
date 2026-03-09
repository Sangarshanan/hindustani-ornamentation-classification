""" import librosa
import numpy as np

audio, sr = librosa.load("zvara.mpeg", mono=True)
# Estimate tonic from the most common f0 value
f0, voiced_flag, _ = librosa.pyin(audio, fmin=80, fmax=800, sr=sr)
f0_voiced = f0[voiced_flag]
# The tonic is often the most frequently occurring pitch
hist, bins = np.histogram(f0_voiced[~np.isnan(f0_voiced)], bins=200)
sa_estimate = bins[np.argmax(hist)]
print(f"Estimated Sa frequency: {sa_estimate:.2f} Hz") """

import librosa
import numpy as np
import matplotlib.pyplot as plt

audio, sr = librosa.load("zvara.mpeg", mono=True)
f0, voiced_flag, _ = librosa.pyin(audio, fmin=80, fmax=800, sr=sr)
f0_voiced = f0[voiced_flag & ~np.isnan(f0)]

plt.figure(figsize=(12, 4))
plt.hist(f0_voiced, bins=200, color='steelblue')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Count")
plt.title("F0 Distribution — Sa should be the dominant peak")
plt.grid(True)
plt.show()

print(f"Min f0: {f0_voiced.min():.1f} Hz")
print(f"Max f0: {f0_voiced.max():.1f} Hz")
print(f"Median f0: {np.median(f0_voiced):.1f} Hz")