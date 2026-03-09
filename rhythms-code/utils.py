import compiam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio, display
import os
import sys

# Add algorithm directory to path to import local modules if needed
sys.path.append(os.path.abspath("algorithm"))

# Mapping for ornamentation types
ORNAMENT_MAPPING = {
    "k": "Kan",
    "g": "Gamak",
    "mu": "Murki",
    "me": "Meend",
    "a": "Andolan",
    "kh": "Khatka",
    "z": "Zamzama",
    "o": "Other"
}

def df_to_anno(df):
    """
    Parses the annotation CSV into a structured DataFrame with start/end times.
    Borrowed and adapted from algorithm/evaluation.py
    """
    all_index = []
    # Ensure column names are set
    df.columns = ["time", "label"]
    
    for index, label in enumerate(df.label):
        # remove none, blanks, etc.
        if label == 'none' or label == '' or label == 'Q':
            all_index.append(index)
            
    df_new = df.drop(all_index)
    df_new = df_new.reset_index(drop=True)

    time_s = []
    time_e = []
    labels = []
    
    # Iterate in steps of 2 assuming start and end pairs
    for i in range(0, len(df_new) - 1, 2):
        s = i
        e = i + 1
        
        # Basic validation that it's a pair
        label_s = str(df_new['label'].iloc[s])
        label_e = str(df_new['label'].iloc[e])
        
        if label_s.endswith('_s') and label_e.endswith('_e'):
            time_s.append(df_new['time'].iloc[s])
            time_e.append(df_new['time'].iloc[e])
            
            # Map the prefix to the full name
            raw_label = label_s[:-2] # remove '_s'
            
            if raw_label.startswith('c_'):
                # Compound label: c_k_me -> Kan + Meend
                parts = raw_label.split('_')[1:] # Skip the 'c'
                mapped_parts = [ORNAMENT_MAPPING.get(p, p) for p in parts]
                labels.append(" + ".join(mapped_parts))
            else:
                # Simple label: me -> Meend
                labels.append(ORNAMENT_MAPPING.get(raw_label, raw_label))
        else:
            # Handle cases where they might not be perfectly paired or none/other labels shifted them
            print(f"Warning: Unexpected pair at index {i}: {label_s}, {label_e}")
            
    anno = pd.DataFrame({
        'time_s': time_s,
        'time_e': time_e,
        'label': labels
    })
    anno['duration'] = anno['time_e'] - anno['time_s']
    return anno

def fetch_ornamentations(raga_name="Aahir Bhairon", num_to_show=5):
    from mapping import MAPPING
    if raga_name not in MAPPING:
        print(f"Raga '{raga_name}' not found in mapping.py")
        return
        
    data = MAPPING[raga_name]
    track_id = data["track_id"]
    annotation_file = f"Ornamentation-In-Hindustani-Vocals-Dataset/{list(data['annotators'].values())[0]}"

    # Set dataset path
    data_home = "/Users/sangarshananveera/Downloads/Datasets"

    # 1. Load dataset and track
    print(f"Loading track {track_id}...")
    saraga_hindustani = compiam.load_dataset("saraga_hindustani", data_home=data_home)
    st = saraga_hindustani.load_tracks()
    track = st[track_id]
    audio_path = track.audio_path
    pitch_path = track.pitch_path
    
    # 2. Load annotations
    print(f"Loading annotations from {annotation_file}...")
    df_raw = pd.read_csv(annotation_file, header=None)
    anno = df_to_anno(df_raw)
    
    # 3. Load pitch data
    print(f"Loading pitch data from {pitch_path}...")
    df_pitch = pd.read_csv(pitch_path, sep="\t", header=None)
    df_pitch.columns = ["time", "f0"]
    # Convert to log frequency (ignoring zero values)
    df_pitch["log_f0"] = df_pitch["f0"].apply(lambda x: np.log2(x) if x > 0 else np.nan)
    
    # 4. Load full audio for slicing
    print(f"Loading audio from {audio_path}...")
    y, sr = librosa.load(audio_path, sr=None)
    
    # 5. Display segments
    print(f"\nShowing first {num_to_show} ornamentations:\n")
    for i in range(min(num_to_show, len(anno))):
        row = anno.iloc[i]
        start_time = row['time_s']
        end_time = row['time_e']
        label = row['label']
        
        # Buffer for context (0.5s before and after)
        plot_start = max(0, start_time - 0.5)
        plot_end = min(len(y)/sr, end_time + 0.5)
        
        print(
            f"{i+1}: {label} | Start: {start_time:.2f}s | End: {end_time:.2f}s | Duration: {row['duration']:.2f}s"
        )
        
        # Filter pitch data for this segment
        segment_pitch = df_pitch[(df_pitch["time"] >= plot_start) & (df_pitch["time"] <= plot_end)]
        
        # Plot
        plt.figure(figsize=(12, 4))
        
        # Pitch Contour Plot
        plt.plot(segment_pitch["time"], segment_pitch["log_f0"], color='blue', marker='o', markersize=2, linestyle='')
        plt.axvspan(start_time, end_time, color='green', alpha=0.2, label='Ornamentation')
        
        plt.ylabel("Log2(F0)")
        plt.xlabel("Time (s)")
        plt.title(f"Pitch Contour - {label}")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        # Audio player (actual ornamentation segment)
        orn_start_sample = int(start_time * sr)
        orn_end_sample = int(end_time * sr)
        y_orn = y[orn_start_sample:orn_end_sample]
        
        display(Audio(y_orn, rate=sr))
        print("-" * 80)

if __name__ == "__main__":
    # If running as a script, we might not see the display() output as an interactive player
    # but the logic remains the same.
    validate_ornamentations()
