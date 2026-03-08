import librosa
import numpy as np

print("Loading audio...")
audio, sr = librosa.load("zvara.mpeg", mono=True)
print(f"Loaded! Duration: {len(audio)/sr:.1f} seconds, Sample rate: {sr}")

print("Extracting f0 with pyin (this may take a while)...")
f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=200, fmax=1500, sr=sr)
print("f0 extraction done!")

timestamps = np.arange(len(f0)) * (512/sr)
np.savetxt("pitch.txt", np.column_stack([timestamps, np.nan_to_num(f0)]), delimiter="\t")
print("Done! f0 saved to f0_output.csv")