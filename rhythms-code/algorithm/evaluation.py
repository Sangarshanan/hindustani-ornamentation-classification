import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re



#NOTES
# from https://pypi.org/project/strsimpy/#normalized-similarity-and-distance
# from strsimpy.weighted_levenshtein import WeightedLevenshtein

def insertion_cost(char):
    return 0.5


def deletion_cost(char):
    return 0.5


def substitution_cost(char_a, char_b):
    return 1.0

# weighted_levenshtein = WeightedLevenshtein(
#     substitution_cost_fn=substitution_cost,
#     insertion_cost_fn=insertion_cost,
#     deletion_cost_fn=deletion_cost)

#print(weighted_levenshtein.distance('String1', 'String'))

#modify the basic function (a) using above weights and (b) normalizing to account for string length
def w_levdist_norm(str1, str2):
    wd = weighted_levenshtein.distance(str1, str2)
    nwd = wd/len(str2)
    return(1 - nwd)


#ORNAMENTATION

def df_to_anno(df):
    all_index=[]
    for index, label in enumerate(df.label):
        #remove none, blanks, complex murkis
        
        if label=='none' or label == '' or label == 'Q':
            # print("Index:", index, "Label:", label, ", none, empty, Q ")
#             re.search('c_.', label) or  or label=='kh_s_e' or label =='k_s_e':
            all_index.append(index)


    df_new = df.drop(all_index)

    df_new.index = np.arange(0,len(df_new),1) 

    time_s=[]
    time_e=[]
    labels1=[]
    for i in range(df_new.index[0], df_new.index[-1], 2):

        s = i
        e = i+1
        time_s.append(df_new['time'].iloc[s])
        time_e.append(df_new['time'].iloc[e])
        labels1.append([df_new['label'].iloc[s], df_new['label'].iloc[e]])

    t={'time_s':time_s, 'time_e':time_e, 'labels':labels1}
    anno = pd.DataFrame(data = t)
    anno['duration'] = anno['time_e']- anno['time_s']
    # display(anno)
    return anno



def plot_fn(ph_id, df_what, df_pred):
    time = df_what.loc[ph_id]['time']
    pitch = df_what.loc[ph_id]['F0']
    plt.figure(figsize=(10, 4), dpi=80)

    # pitch = pitch.replace(0, np.nan)
    plt.plot(time, pitch)

    for i in range(len(df_what.loc[ph_id]['orn_samples'])):
        ornament_indices = df_what.loc[ph_id]['orn_samples'][i]

        time_s = time[ornament_indices[0]]
        time_e = time[ornament_indices[1]]
        # print("orn duration:", time_e - time_s)
        plt.axvspan(time_s, time_e
                    , color='green', alpha=0.25, lw=0)

    intersect_df = df_pred[np.logical_or(df_pred['time_s'].between(time[0], time[-1]), 
                            df_pred['time_e'].between(time[0], time[-1]))]
    display(intersect_df)
    for i in intersect_df.index:
        plt.axvspan(df_pred['time_s'].loc[i], df_pred['time_e'].loc[i]
                    , color='orange', alpha=0.55, lw=0)

    plt.grid()
    plt.title(f'Phrase {ph_id}')
    plt.xlabel('Time in sec')
    plt.ylabel('Pitch in log frequency values')
    plt.show()


def eval_notes(pred_df, true_df, window=0.5):
    """
    Evaluates predicted ornaments against ground truth annotations using a boundary hit rate metric.
    A hit is counted if a reference boundary is within `window` seconds of a predicted boundary.
    Classes are grouped into: kan, meend, andolan, murki.
    """
    # Grouping mapping for predictions and ground truth
    CLASS_MAP = {
        'k': 'kan', 'kh': 'kan',
        'me': 'meend',
        'a': 'andolan',
        'mu': 'murki', 'z': 'murki', 'g': 'murki'
    }

    # Helper to map labels
    def map_label(label):
        # Handle compound labels (e.g., 'c_k_me')
        if pd.isna(label) or not isinstance(label, str):
            return None
            
        # Clean label (from prediction it's numeric 1-5, handled previously before this function)
        # Assuming true_df has original string labels
        lab = str(label).lower().strip()
        
        # We need to map labels from `true_df` e.g. 'c_k_me' or 'me'
        if lab.startswith('c_'):
            # It's compound, count as murki for simplicity or split? 
            # Instruction said: murki, zumzuma, gamak -> murki. 
            parts = lab.split('_')[1:]
        else:
            parts = [lab]
            
        mapped_parts = [CLASS_MAP.get(p, 'none') for p in parts]
        
        # If any part maps to a valid class, we'll return the first one, 
        # or we can treat complex ones as 'murki' as instructed.
        valid_parts = [p for p in mapped_parts if p != 'none']
        if not valid_parts:
            return None
            
        # If it's complex (multiple valid parts), let's map it to 'murki' as a complex ornament
        if len(valid_parts) > 1:
            return 'murki'
            
        return valid_parts[0]

    # Map true labels — expects a 'label' column with strings (from utils.df_to_anno)
    true_df = true_df.copy()
    true_df['mapped_class'] = true_df['label'].apply(map_label)

    # Filter out anything that mapped to None
    true_df = true_df.dropna(subset=['mapped_class'])
    
    # Ensure pred mapping
    pred_df = pred_df.copy()
    
    results = {}
    classes = ['kan', 'meend', 'andolan', 'murki']
    
    for cls in classes:
        # Get ground truth boundaries for this class
        true_cls = true_df[true_df['mapped_class'] == cls]
        pred_cls = pred_df[pred_df['label'] == cls]
        
        true_starts = true_cls['time_s'].values
        true_ends = true_cls['time_e'].values
        
        pred_starts = pred_cls['time_s'].values
        pred_ends = pred_cls['time_e'].values
        
        # A true annotation is a "hit" if either its start OR end is within `window` of ANY pred start OR end.
        hits = 0
        for i in range(len(true_cls)):
            t_s = true_starts[i]
            t_e = true_ends[i]
            
            # Check if t_s is matched
            start_matched = np.any(np.abs(pred_starts - t_s) <= window) or np.any(np.abs(pred_ends - t_s) <= window)
            # Check if t_e is matched
            end_matched = np.any(np.abs(pred_starts - t_e) <= window) or np.any(np.abs(pred_ends - t_e) <= window)
            
            if start_matched or end_matched:
                hits += 1
                
        total_true = len(true_cls)
        total_pred = len(pred_cls)
        
        recall = hits / total_true if total_true > 0 else 0
        
        # Precision (hitting a predicted boundary)
        pred_hits = 0
        for i in range(len(pred_cls)):
            p_s = pred_starts[i]
            p_e = pred_ends[i]
            
            start_matched = np.any(np.abs(true_starts - p_s) <= window) or np.any(np.abs(true_ends - p_s) <= window)
            end_matched = np.any(np.abs(true_starts - p_e) <= window) or np.any(np.abs(true_ends - p_e) <= window)
            
            if start_matched or end_matched:
                pred_hits += 1
                
        precision = pred_hits / total_pred if total_pred > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[cls] = {
            'true_count': total_true,
            'pred_count': total_pred,
            'hits (recall)': hits,
            'recall': recall,
            'precision': precision,
            'f1': f1
        }
        
    return pd.DataFrame(results).T