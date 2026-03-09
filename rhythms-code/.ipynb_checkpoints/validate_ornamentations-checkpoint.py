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
        label_s = df_new['label'].iloc[s]
        label_e = df_new['label'].iloc[e]
        
        if label_s.endswith('_s') and label_e.endswith('_e'):
            time_s.append(df_new['time'].iloc[s])
            time_e.append(df_new['time'].iloc[e])
            labels.append(label_s.split('_')[0])
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

def validate_ornamentations(track_id="66_Aahir_Bhairon", annotation_file="Ornamentation-In-Hindustani-Vocals-Dataset/AahirBhairon_av.csv", num_to_show=5):
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
        
        print(f"Ornamentation {i+1}: {label} | Start: {start_time:.2f}s | End: {end_time:.2f}s | Duration: {row['duration']:.2f}s")
        
        # Filter pitch data for this segment
        segment_pitch = df_pitch[(df_pitch["time"] >= plot_start) & (df_pitch["time"] <= plot_end)]
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        # Waveform
        start_sample = int(plot_start * sr)
        end_sample = int(plot_end * sr)
        y_segment_full = y[start_sample:end_sample]
        t_audio = np.linspace(plot_start, plot_end, len(y_segment_full))
        
        ax1.plot(t_audio, y_segment_full, color='gray', alpha=0.5)
        ax1.axvspan(start_time, end_time, color='green', alpha=0.2, label='Ornamentation')
        ax1.set_title(f"Waveform - {label}")
        ax1.legend()
        
        # Pitch
        ax2.plot(segment_pitch["time"], segment_pitch["log_f0"], color='blue', marker='o', markersize=2, linestyle='')
        ax2.axvspan(start_time, end_time, color='green', alpha=0.2)
        ax2.set_ylabel("Log2(F0)")
        ax2.set_xlabel("Time (s)")
        ax2.set_title(f"Pitch Contour - {label}")
        
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
