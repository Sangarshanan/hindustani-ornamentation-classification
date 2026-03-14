
import os
import pandas as pd
import librosa

# Mapping between raga names and their corresponding audio/annotation files.
# Includes MBID, artist, and duration (seconds) for limiting audio↔annotation alignment.

MAPPING = {
    "Aahir Bhairon": {
        "mbid": "51656b20-295c-40f9-8dab-005b9b90fa98",
        "artist": "Ajoy Chakrabarty",
        "audio_path": "Geetinandan : Part-3 by Ajoy Chakrabarty/Aahir Bhairon/Aahir Bhairon",
        "track_id": "66_Aahir_Bhairon",
        "raga_idx": 19,  # ahir_bhairon
        "duration_str": "9:49",
        "duration_s": 589,
        "annotation_end_s": 589,
        "annotators": {"av": "AahirBhairon_av.csv"}
    },

    "Bairagi": {
        "mbid": "b71c2774-2532-4692-8761-5452e2a83118",
        "artist": "Ajoy Chakrabarty",
        "audio_path": "Geetinandan : Part-3 by Ajoy Chakrabarty/Bairagi/Bairagi",
        "track_id": "59_Bairagi",
        "raga_idx": 4,  # bairagi
        "duration_str": "14:59",
        "duration_s": 899,
        "annotation_end_s": 899,
        "annotators": {"rj": "Bairagi_rj.csv"}
    },

    "Bhairavi Dadra": {
        "mbid": "f5d00c2d-5ca1-4a49-bb38-1af3496e6dfc",
        "artist": "Omkar Dadarkar",
        "audio_path": "Raag Bihag & Bhairavi Dadra by Omkar Dadarkar/Bhairavi Dadra/Bhairavi Dadra",
        "track_id": "80_Bhairavi_Dadra",
        "raga_idx": 8,  # bhairavi_dadra (track title in Saraga)
        "duration_str": "10:02",
        "duration_s": 602,
        "annotation_end_s": 602,
        "annotators": {"av": "BhairaviDadr_av.csv"}
    },

    "Irani Bhairavi": {
        "mbid": "9f4ba52c-935c-4066-ad79-daa709dfa6ec",
        "artist": "Brajeshwar Mukherjee",
        "audio_path": "New Signature by Brajeshwar Mukherjee/Irani Bhairavi Thumri/Irani Bhairavi Thumri",
        "track_id": "50_Irani_Bhairavi_Thumri",
        "raga_idx": 5,  # irani_bhairavi
        "duration_str": "6:00",
        "duration_s": 360,
        "annotation_end_s": 360,
        "annotators": {
            "sb": "Irani Bhairavi Thumri_sb.csv",
            "av": "IraniBhairaviThumri_av.csv"
        }
    },

    "Kalavati": {
        "mbid": "d7a57d81-a5dd-44d8-aabf-951cb8f73fe2",
        "artist": "Ajoy Chakrabarty",
        "audio_path": "Geetinandan : Part-3 by Ajoy Chakrabarty/Kalavati/Kalavati",
        "track_id": "73_Kalavati",
        "raga_idx": 52,  # kalavati
        "duration_str": "5:43",
        "duration_s": 343,
        "annotation_end_s": 343,
        "annotators": {"vi": "Kalavati_vi.csv"}
    },

    "Malkauns": {
        "mbid": "b77a3149-40fa-4e28-8205-cd4f006f60a1",
        "artist": "Ajoy Chakrabarty",
        "audio_path": "Geetinandan : Part-3 by Ajoy Chakrabarty/Malkauns/Malkauns",
        "track_id": "62_Malkauns",
        "raga_idx": 18,  # malkauns
        "duration_str": "5:33",
        "duration_s": 333,
        "annotation_end_s": 333,
        "annotators": {
            "sb": "Malkauns_sb.csv",
            "vh": "malkauns_vh.csv"
        }
    },

    "Chandrakauns": {
        "mbid": "49747cb6-7edc-43aa-bf15-44a6043f6472",
        "artist": "Satyasheel Deshpande",
        "audio_path": "Raag Malkauns, Chandrakauns & Majh Khamaj Thumri by Satyasheel Deshpande/Raag Chandrakauns/Raag Chandrakauns",
        "track_id": "54_Raag_Chandrakauns",
        "raga_idx": 14,  # chandrakauns
        "duration_str": "3:56",
        "duration_s": 236,
        "annotation_end_s": 236,
        "annotators": {
            "sb": "Raag Chandrakauns_SnehaBhat.csv",
            "av": "RaagChandrakauns_av.csv"
        }
    },

    "Majh Khamaj Thumri": {
        "mbid": "6bb27c39-a1d0-4775-8915-662044c43352",
        "artist": "Satyasheel Deshpande",
        "audio_path": "Raag Malkauns, Chandrakauns & Majh Khamaj Thumri by Satyasheel Deshpande/Majh Khamaj Thumri/Majh Khamaj Thumri",
        "track_id": "55_Majh_Khamaj_Thumri",
        "raga_idx": 17,  # majh_khamaj_thumri
        "duration_str": "5:19",
        "duration_s": 319,
        "annotation_end_s": 319,
        "annotators": {"vh": "manjkhamaj_vh.csv"}
    },

    "Nat Bhairon": {
        "mbid": "6a2c841d-5a0e-4886-a5c0-f856fccbb938",
        "artist": "Ajoy Chakrabarty",
        "audio_path": "Geetinandan : Part-3 by Ajoy Chakrabarty/Nat Bhairon/Nat Bhairon",
        "track_id": "65_Nat_Bhairon",
        "raga_idx": 32,  # nat_bhairav (closest match)
        "duration_str": "10:38",
        "duration_s": 638,
        "annotation_end_s": 638,
        "annotators": {"vh": "natbhairon_vh.csv"}
    },

    "Raageshree": {
        "mbid": "7704b0d2-eeac-47e4-ba05-c541a91eb621",
        "artist": "Ajoy Chakrabarty",
        "audio_path": "Geetinandan : Part-3 by Ajoy Chakrabarty/Raageshree/Raageshree",
        "track_id": "68_Raageshree",
        "raga_idx": 39,  # raageshree
        "duration_str": "8:12",
        "duration_s": 492,
        "annotation_end_s": 492,
        "annotators": {"vh": "raageshree_vh.csv"}
    },

    "Todi": {
        "mbid": "204008ee-89fe-44a4-a0c7-7bd7d64cf8a4",
        "artist": "Ajoy Chakrabarty",
        "audio_path": "Geetinandan : Part-3 by Ajoy Chakrabarty/Todi/Todi",
        "track_id": "64_Todi",
        "raga_idx": 28,  # todi
        "duration_str": "9:01",
        "duration_s": 541,
        "annotation_end_s": 541,
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
            map_duration = data.get("annotation_end_s", audio_duration)
            if map_duration > audio_duration:
                print(
                    f"  [WARN] Mapping duration ({map_duration:.2f}s) exceeds audio duration ({audio_duration:.2f}s); using audio duration for validation."
                )
                map_duration = audio_duration
            
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
                
                if max_time > map_duration:
                    print(f"  [ERROR] Annotator {annotator_id} ({ann_file}): Max annotation time ({max_time:.2f}s) EXCEEDS Mapping duration ({map_duration:.2f}s)")
                else:
                    print(f"  [OK] Annotator {annotator_id} ({ann_file}): Max annotation time ({max_time:.2f}s) fits within Mapping duration ({map_duration:.2f}s)")
        except Exception as e:
            print(f"  [!] Error processing {audio_path_prefix}: {e}")
