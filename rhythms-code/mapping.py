
import os
import pandas as pd
import librosa

# Mapping between Raga names and their corresponding audio/annotation files
# Note: For multiple annotators, we use a dictionary mapping annotator abbreviations to CSV files.
MAPPING = {
    "Aahir Bhairon": {
        "audio_path": "Geetinandan : Part-3 by Ajoy Chakrabarty/Aahir Bhairon/Aahir Bhairon",
        "annotators": {"av": "AahirBhairon_av.csv"}
    },
    
    "Bairagi": {
        "audio_path": "New Signature by Brajeshwar Mukherjee/Raag Bairagi/Raag Bairagi",
        "annotators": {"rj": "Bairagi_rj.csv"}
    },
    
    "Bhairavi Dadra": {
        "audio_path": "Raag Bihag & Bhairavi Dadra by Omkar Dadarkar/Bhairavi Dadra/Bhairavi Dadra",
        "annotators": {"av": "BhairaviDadr_av.csv"}
    },
    
    "Irani Bhairavi Thumri": {
        "audio_path": "New Signature by Brajeshwar Mukherjee/Irani Bhairavi Thumri/Irani Bhairavi Thumri",
        "annotators": {
            "sb": "Irani Bhairavi Thumri_sb.csv", 
            "av": "IraniBhairaviThumri_av.csv"
        }
    },
    
    "Kalavati": {
        "audio_path": "Geetinandan : Part-3 by Ajoy Chakrabarty/Kalavati/Kalavati",
        "annotators": {"vi": "Kalavati_vi.csv"}
    },
    
    "Malkauns": {
        "audio_path": "Raag Malkauns, Chandrakauns & Majh Khamaj Thumri by Satyasheel Deshpande/Raag Malkauns/Raag Malkauns",
        "annotators": {
            "sb": "Malkauns_sb.csv", 
            "vh": "malkauns_vh.csv"
        }
    },
    
    "Chandrakauns": {
        "audio_path": "Raag Malkauns, Chandrakauns & Majh Khamaj Thumri by Satyasheel Deshpande/Raag Chandrakauns/Raag Chandrakauns",
        "annotators": {
            "SnehaBhat": "Raag Chandrakauns_SnehaBhat.csv", 
            "av": "RaagChandrakauns_av.csv"
        }
    },
    
    "Majh Khamaj Thumri": {
        "audio_path": "Raag Malkauns, Chandrakauns & Majh Khamaj Thumri by Satyasheel Deshpande/Majh Khamaj Thumri/Majh Khamaj Thumri",
        "annotators": {"vh": "manjkhamaj_vh.csv"}
    },
    
    "Nat Bhairon": {
        "audio_path": "Geetinandan : Part-3 by Ajoy Chakrabarty/Nat Bhairon/Nat Bhairon",
        "annotators": {"vh": "natbhairon_vh.csv"}
    },
    
    "Raageshree": {
        "audio_path": "Geetinandan : Part-3 by Ajoy Chakrabarty/Raageshree/Raageshree",
        "annotators": {"vh": "raageshree_vh.csv"}
    },
    
    "Todi": {
        "audio_path": "Raag Todi by Kumar Gandharva/Raag Todi/Raag Todi",
        "annotators": {"dp": "todi_dp.csv"}
    },
}

def get_audio_duration(audio_path):
    """Returns the duration of an audio file in seconds."""
    return librosa.get_duration(filename=audio_path)

def validate_mapping(audio_base_dir, annotation_base_dir, annotator_filter=None):
    """
    Iterates through the mapping and checks if the annotation's maximum timestamp 
    exceeds the duration of the corresponding audio file.
    
    Args:
        audio_base_dir: Directory containing audio files.
        annotation_base_dir: Directory containing CSV files.
        annotator_filter: Optional string to filter by a specific annotator ID.
    """
    for raga_name, data in MAPPING.items():
        audio_path_prefix = data["audio_path"]
        annotators_map = data["annotators"]
        # audio_path_prefix is e.g. "Album/Raga/Track"
        # The actual file is usually "Album/Raga/Track.mp3" or "Track.mp3.mp3"
        full_prefix = os.path.join(audio_base_dir, audio_path_prefix)
        audio_dir = os.path.dirname(full_prefix)
        track_prefix = os.path.basename(full_prefix)
        
        if not os.path.exists(audio_dir):
            print(f"Directory not found: {audio_dir}")
            continue
            
        # Find audio files matching the prefix
        audio_files = [f for f in os.listdir(audio_dir) if f.startswith(track_prefix) and f.endswith(('.wav', '.mp3', '.flac', '.m4a'))]
        if not audio_files:
            print(f"No audio file found matching prefix '{track_prefix}' in {audio_dir}")
            continue
            
        audio_path = os.path.join(audio_dir, audio_files[0])
        print(f"\nValidating mapping for audio: {audio_path_prefix}")
        
        try:
            audio_duration = get_audio_duration(audio_path)
            
            for annotator_id, ann_file in annotators_map.items():
                if annotator_filter and annotator_id != annotator_filter:
                    continue
                    
                ann_path = os.path.join(annotation_base_dir, ann_file)
                if not os.path.exists(ann_path):
                    print(f"  [!] Annotation file not found for annotator {annotator_id}: {ann_path}")
                    continue
                    
                df = pd.read_csv(ann_path, header=None)
                if df.empty:
                    print(f"  [!] Empty annotation file for annotator {annotator_id}: {ann_file}")
                    continue
                    
                # Assuming the first column is time/timestamps based on validate_ornamentations.py
                max_time = df[0].max()
                
                if max_time > audio_duration:
                    print(f"  [ERROR] Annotator {annotator_id} ({ann_file}): Max annotation time ({max_time:.2f}s) EXCEEDS Audio duration ({audio_duration:.2f}s)")
                else:
                    print(f"  [OK] Annotator {annotator_id} ({ann_file}): Max annotation time ({max_time:.2f}s) fits within Audio duration ({audio_duration:.2f}s)")
        except Exception as e:
            print(f"  [!] Error processing {audio_path_prefix}: {e}")

if __name__ == "__main__":
    # Base directory for the uncompressed saraga dataset
    AUDIO_BASE = "/Users/sangarshananveera/Downloads/Datasets/saraga1.5_hindustani"
    
    # Path to the ornamentation dataset
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    ANNO_BASE = os.path.join(CURRENT_DIR, "Ornamentation-In-Hindustani-Vocals-Dataset")
    
    validate_mapping(AUDIO_BASE, ANNO_BASE)
