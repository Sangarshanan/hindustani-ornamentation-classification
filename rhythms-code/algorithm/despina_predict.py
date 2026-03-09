import sys
sys.path.append(r"C:\Users\despn\Downloads\MIR\automated-symbolic-transcription-hindustani-vocals\algorithm")

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from Raga import Raga
from ornamentation import theWHERE, calc_instab, get_peaks, plot_peaks
from quantization import closest

# FILL IN THESE VALUES

AUDIO_FILE = "zvara.mpeg"
RAGA_IDX   = 9        # yaman
SA_FREQ    = 465.39   # estimated tonic in Hz


# Step 1: Extract f0
""" print("Loading audio...")
audio, sr = librosa.load(AUDIO_FILE, mono=True)
print(f"Duration: {len(audio)/sr:.1f} seconds")

print("Extracting f0 (may take a few minutes)...")
f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=200, fmax=1500, sr=sr)
timestamps = np.arange(len(f0)) * (512/sr)
f0_clean = np.nan_to_num(f0) """ 

# Step 1: Extract f0 with Essentia's PredominantPitchMelodia
import essentia.standard as es

print("Loading audio...")
loader = es.MonoLoader(filename=AUDIO_FILE, sampleRate=44100)
audio = loader()
print(f"Duration: {len(audio)/44100:.1f} seconds")

# Take only first 10 seconds
""" audio = audio[:10 * 44100]
print(f"Trimmed duration: {len(audio)/44100:.1f} seconds") """


print("Extracting f0 with Melodia...")
pmd = es.PredominantPitchMelodia(frameSize=2048, hopSize=128)
f0, confidence = pmd(audio)

#audio information
print(f"Audio length: {len(audio)}")
print(f"Audio min/max: {audio.min():.4f} / {audio.max():.4f}")
print(f"Raw f0 min/max: {f0.min():.4f} / {f0.max():.4f}")
print(f"Raw f0 non-zero: {(f0 > 0).sum()}")
print(f"Confidence min/max: {confidence.min():.4f} / {confidence.max():.4f}")

# Zero out unvoiced frames
f0_clean = f0.copy()
f0_clean[confidence < 0.05] = 0.0 

# Compute timestamps
timestamps = np.arange(len(f0_clean)) * (128/44100)

np.savetxt("pitch.txt", np.column_stack([timestamps, f0_clean]), delimiter="\t")
print("f0 saved to pitch.txt")

# Check what pitch.txt actually contains
import pandas as pd
df_check = pd.read_csv("pitch.txt", delimiter="\t", header=None)
print(df_check.head(10))
print(f"Non-zero rows: {(df_check[1] > 0).sum()}")

# Step 2: Save tonic
with open("ctonic.txt", "w") as f:
    f.write(str(SA_FREQ))

# Step 3: Create Raga object
pitches_path = [""] * RAGA_IDX + ["pitch.txt"]
ctonic_path  = [""] * RAGA_IDX + ["ctonic.txt"]
raga = Raga(idx=RAGA_IDX, pitches_path=pitches_path)
raga.set_raga_object(pitches_path, ctonic_path)
raga.get_raga_object()  # prints raga name, sa, swaras to verify

# MANUALLY LOAD PHRASES
f0_vals = raga.df_f0['f0'].values
phrases = []
in_phrase = False
start = 0

print(raga.df_f0.head(10))
print(f"Non-zero f0 frames: {(raga.df_f0['f0'] > 0).sum()}")
print(f"Total frames: {len(raga.df_f0)}")
print(f"Unique f0 values sample: {raga.df_f0['f0'].unique()[:10]}")

for i, val in enumerate(f0_vals):
    if val > 0 and not in_phrase:
        start = i
        in_phrase = True
    elif val == 0 and in_phrase:
        if i - start > 10:
            phrases.append([start, i])
        in_phrase = False

if in_phrase and len(f0_vals) - start > 10:
    phrases.append([start, len(f0_vals) - 1])

raga.phrases = phrases
### Check if it works
print(f"Found {len(phrases)} phrases")
print(f"First 5 phrases: {phrases[:5]}") 

# Step 4: Run algorithm
print("\nRunning ornamentation detection...")
df_quantized_notes, df_ornamentations = theWHERE(
    raga,
    frame_len = 56,
    hop_len   = 25,
    y_thresh  = 0.04,
    x_thresh  = 2
)

print("\nQuantized notes:")
print(df_quantized_notes.head(20))
print("\nOrnamentations:")
print(df_ornamentations.head(20))

# Step 5: Save outputs
df_quantized_notes.to_csv("quantized_notes.csv", index=False)
df_ornamentations.to_csv("ornamentations.csv", index=False)
print("\nSaved to quantized_notes.csv and ornamentations.csv")

# ============================================================
# PLOTS
# ============================================================

# Plot 1: Raw f0 contour
plt.figure(figsize=(14, 4))
plt.plot(timestamps, f0_clean, color='steelblue', linewidth=0.5)
plt.title("Raw F0 Contour")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.grid(True)
plt.tight_layout()
plt.savefig("f0_contour.png", dpi=150)
plt.show()

# Plot 2: Quantized swaras as piano roll
plt.figure(figsize=(14, 4))
for _, row in df_quantized_notes.iterrows():
    plt.hlines(y=row['swara'], xmin=row['time_s'],
               xmax=row['time_s'] + row['duration'],
               linewidth=3, color='steelblue')
plt.title("Quantized Swaras (Yaman)")
plt.xlabel("Time (s)")
plt.ylabel("Swara")
plt.grid(True)
plt.tight_layout()
plt.savefig("quantized_swaras.png", dpi=150)
plt.show()

# Plot 3: F0 contour with ornament regions highlighted
orn_colors = {1: 'green', 2: 'orange', 3: 'red', 4: 'purple', 5: 'brown'}
orn_labels = {1: 'kan', 2: 'meend1', 3: 'meend2', 4: 'andolan', 5: 'murki'}

plt.figure(figsize=(14, 4))
plt.plot(timestamps, f0_clean, color='lightgray', linewidth=0.5, zorder=1)
for _, row in df_ornamentations.iterrows():
    if len(row['orn_type']) == 0:
        continue
    for i, orn in enumerate(row['orn_type']):
        if i >= len(row['orn_samples']):
            continue
        s, e = row['orn_samples'][i]
        t_start = raga.df_f0['time'].iloc[s]
        t_end   = raga.df_f0['time'].iloc[min(e, len(raga.df_f0)-1)]
        plt.axvspan(t_start, t_end, alpha=0.4,
                    color=orn_colors.get(orn, 'gray'),
                    label=orn_labels.get(orn, 'unknown'))
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right')
plt.title("F0 Contour with Ornament Regions")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.grid(True)
plt.tight_layout()
plt.savefig("ornaments.png", dpi=150)
plt.show()

# Plot 4: Instability scores + peaks for each phrase (from plot_peaks)
for phrase_idx in range(min(5, len(raga.phrases))):  # plot first 5 phrases
    if raga.phrases[phrase_idx][0] == 0 and raga.phrases[phrase_idx][1] == 0:
        continue
    curr_phrase = raga.df_f0.iloc[raga.phrases[phrase_idx][0]:raga.phrases[phrase_idx][1]]
    curr_phrase['closest'], _ = closest(curr_phrase['log_freq'], raga.log_freq, raga.sw)
    x_closest = np.array(curr_phrase['closest'].tolist())
    ma_curve = calc_instab(x_closest, frame_len=56, hop_len=25)
    peaks, properties = get_peaks(ma_curve, prominence=0.04, height=0.0001)
    plot_peaks(phrase_idx, ma_curve, peaks, properties)
    plt.savefig(f"instability_phrase_{phrase_idx}.png", dpi=150)
    plt.show()

""" # Plot 5: Per-phrase detail — original f0, quantized f0, instability
def ornament_plot_fixed(raga1, orig_x, orig_y, quantized_y, orn_x, phrase_idx):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    axes[0].plot(orig_x, orig_y, color='steelblue')
    axes[0].set_title(f"Phrase {phrase_idx} — Original F0")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].grid(True)

    axes[1].plot(orig_x, quantized_y, color='darkorange')
    axes[1].set_title(f"Phrase {phrase_idx} — Quantized F0")
    axes[1].set_ylabel("Log Frequency")
    if len(quantized_y) > 0:
        low_id  = np.argmin(np.abs(raga1.log_freq - min(quantized_y)))
        high_id = np.argmax(np.abs(raga1.log_freq - max(quantized_y)))
        tick_freqs = raga1.log_freq[max(0, low_id-2):min(len(raga1.log_freq), high_id+2)]
        tick_labels = raga1.sw[max(0, low_id-2):min(len(raga1.sw), high_id+2)]
        axes[1].set_yticks(tick_freqs)
        axes[1].set_yticklabels(tick_labels)
    axes[1].grid(True)

    axes[2].plot(orn_x, color='green')
    axes[2].set_title(f"Phrase {phrase_idx} — Instability Scores")
    axes[2].set_ylabel("Instability")
    axes[2].set_xlabel("Frames")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(f"phrase_detail_{phrase_idx}.png", dpi=150)
    plt.show()

for phrase_idx in range(min(5, len(raga.phrases))):
    if raga.phrases[phrase_idx][0] == 0 and raga.phrases[phrase_idx][1] == 0:
        continue
    curr_phrase = raga.df_f0.iloc[raga.phrases[phrase_idx][0]:raga.phrases[phrase_idx][1]]
    curr_phrase['closest'], _ = closest(curr_phrase['log_freq'], raga.log_freq, raga.sw)
    x_closest = np.array(curr_phrase['closest'].tolist())
    ma_curve = calc_instab(x_closest, frame_len=56, hop_len=25)
    ornament_plot_fixed(raga, curr_phrase['time'].values, curr_phrase['f0'].values, x_closest, ma_curve, phrase_idx)   """