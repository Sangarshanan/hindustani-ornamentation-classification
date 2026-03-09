import os
import sys
import numpy as np
import pandas as pd
import librosa
import compiam

from algorithm.Raga import Raga
from algorithm.ornamentation import theWHERE
from algorithm.evaluation import eval_notes

from mapping import MAPPING

# TODO: change this to your dataset path
DATA_HOME = "/Users/sangarshananveera/Downloads/Datasets"
ANNO_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Ornamentation-In-Hindustani-Vocals-Dataset")

def load_pitch_data(track_id):
    """Loads pitch data using compiam to save Essentia extraction time."""
    saraga_hindustani = compiam.load_dataset("saraga_hindustani", data_home=DATA_HOME)
    st = saraga_hindustani.load_tracks()
    
    if track_id not in st:
        print(f"Error: Track {track_id} not found in compiam dataset.")
        return None, None
        
    track = st[track_id]
    pitch_path = track.pitch_path
    
    print(f"Loading pitch data from {pitch_path}...")
    df_pitch = pd.read_csv(pitch_path, sep="\t", header=None)
    df_pitch.columns = ["time", "f0"]
    return df_pitch, track.tonic

def run_evaluation_for_raga(raga_name):
    if raga_name not in MAPPING:
        print(f"Raga {raga_name} not found in MAPPING.")
        return

    print(f"\n--- Evaluating Raga: {raga_name} ---")
    data = MAPPING[raga_name]
    annotators = data["annotators"]
    track_id = data["track_id"]
    raga_idx = data["raga_idx"]

    if not annotators:
        print("No annotators defined — skipping evaluation.")
        return

    # 1. Load Pitch Data
    df_pitch, estimated_tonic = load_pitch_data(track_id)
    sa_freq = estimated_tonic 
    if df_pitch is None:
        return

    # 2. Setup Raga Object directly from in-memory data (no temp files needed)
    raga = Raga.from_dataframe(df_pitch, sa_freq, raga_idx)
    print(f"Found {len(raga.phrases)} phrases.")


    # 4. Run `theWHERE`
    print("Running ornamentation detection...")
    df_quantized_notes, df_ornamentations = theWHERE(
        raga,
        frame_len = 56,
        hop_len   = 25,
        y_thresh  = 0.04,
        x_thresh  = 2
    )
    
    # 5. Format predictions into start/end times & mapped classes
    # `theWHERE` returns `df_ornamentations` with `orn_type` (list of 1-5 integers) and `orn_samples`
    PRED_CLASS_MAP = {1: 'kan', 2: 'meend', 3: 'meend', 4: 'andolan', 5: 'murki'}
    
    pred_records = []
    for _, row in df_ornamentations.iterrows():
        if len(row['orn_type']) == 0:
            continue
            
        for i, orn_code in enumerate(row['orn_type']):
            if i >= len(row['orn_samples']):
                continue
                
            s, e = row['orn_samples'][i]
            # Convert frame indices to time in seconds
            t_start = raga.df_f0['time'].iloc[s]
            t_end   = raga.df_f0['time'].iloc[min(e, len(raga.df_f0)-1)]
            mapped_class = PRED_CLASS_MAP.get(orn_code, 'unknown')
            
            pred_records.append({
                'time_s': t_start,
                'time_e': t_end,
                'label': mapped_class
            })
            
    pred_df = pd.DataFrame(pred_records)
    print(f"Found {len(pred_df)} predicted ornaments.")
    if len(pred_df) == 0:
        print("No predictions found. Skipping evaluation.")
        return
        
    # 6. Evaluate against ground truth
    for ann_id, ann_file in annotators.items():
        ann_path = os.path.join(ANNO_BASE, ann_file)
        if not os.path.exists(ann_path):
            print(f"Annotation file not found: {ann_path}")
            continue
            
        print(f"\nEvaluating against Annotator: {ann_id}")
        # Load raw annotations — keep raw label strings for eval_notes::map_label
        raw_df = pd.read_csv(ann_path, header=None, names=["time", "label"])
        raw_df = raw_df[~raw_df["label"].isin(["none", "", "Q"])].dropna()
        raw_df = raw_df.reset_index(drop=True)

        # Build start/end pairs preserving raw labels (e.g. 'me', 'k', 'c_k_me')
        rows = []
        for i in range(0, len(raw_df) - 1, 2):
            l_s = str(raw_df["label"].iloc[i])
            l_e = str(raw_df["label"].iloc[i + 1])
            if l_s.endswith("_s") and l_e.endswith("_e"):
                rows.append({
                    "time_s": raw_df["time"].iloc[i],
                    "time_e": raw_df["time"].iloc[i + 1],
                    "label": l_s[:-2],  # strip '_s', e.g. 'me', 'c_k_me'
                })
        true_df = pd.DataFrame(rows)

        # Calculate metrics
        results_df = eval_notes(pred_df, true_df, window=0.5)
        print(results_df)

    return results_df

def print_final_report(all_results):
    """
    Combines results across all ragas and prints in the requested format.
    """
    classes = ['kan', 'meend', 'andolan', 'murki']
    totals = {cls: {'tp_rec': 0, 'tp_prec': 0, 'support': 0, 'pred': 0} for cls in classes}

    for raga_name, results_df in all_results.items():
        for cls in classes:
            if cls in results_df.index:
                row = results_df.loc[cls]
                totals[cls]['support'] += row['true_count']
                totals[cls]['pred'] += row['pred_count']
                # hits (recall) is used for recall numerator
                totals[cls]['tp_rec'] += row['hits (recall)']
                # tp_prec = precision * pred_count
                totals[cls]['tp_prec'] += row.get('precision', 0) * row['pred_count']

    print("\n" + "="*60)
    print("GLOBAL EVALUATION REPORT (Combined across all Ragas)")
    print("="*60)
    print(f"{'Ornament':<15} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'Support':<10}")

    report_data = []
    total_support = 0
    total_hits = 0
    
    for cls in classes:
        t = totals[cls]
        precision = t['tp_prec'] / t['pred'] if t['pred'] > 0 else 0
        recall = t['tp_rec'] / t['support'] if t['support'] > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = int(t['support'])
        
        print(f"{cls.capitalize():<15} {precision:<10.2f} {recall:<10.2f} {f1:<10.2f} {support:<10}")
        
        report_data.append({
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        })
        total_support += support
        total_hits += t['tp_rec']

    # Accuracy (Total Hits / Total Support)
    accuracy = total_hits / total_support if total_support > 0 else 0
    
    # Macro Average
    macro_prec = np.mean([d['precision'] for d in report_data])
    macro_rec = np.mean([d['recall'] for d in report_data])
    macro_f1 = np.mean([d['f1'] for d in report_data])
    
    # Weighted Average
    weighted_prec = np.sum([d['precision'] * d['support'] for d in report_data]) / total_support if total_support > 0 else 0
    weighted_rec = np.sum([d['recall'] * d['support'] for d in report_data]) / total_support if total_support > 0 else 0
    weighted_f1 = np.sum([d['f1'] * d['support'] for d in report_data]) / total_support if total_support > 0 else 0

    print("-" * 60)
    print(f"{'Accuracy':<37} {accuracy:<10.2f} {total_support:<10}")
    print(f"{'Macro Avg':<15} {macro_prec:<10.2f} {macro_rec:<10.2f} {macro_f1:<10.2f} {total_support:<10}")
    print(f"{'Weighted Avg':<15} {weighted_prec:<10.2f} {weighted_rec:<10.2f} {weighted_f1:<10.2f} {total_support:<10}")
    print("="*60)

def run_all():
    """Run evaluation for all ragas in MAPPING that have annotations."""
    ragas_to_evaluate = [name for name, data in MAPPING.items() if data.get("annotators")]
    print(f"Found {len(ragas_to_evaluate)} ragas with annotations: {ragas_to_evaluate}")
    all_results = {}
    for raga_name in ragas_to_evaluate:
        result = run_evaluation_for_raga(raga_name)
        if result is not None:
            all_results[raga_name] = result
    
    if all_results:
        print_final_report(all_results)
    
    return all_results


if __name__ == "__main__":
    run_all()
