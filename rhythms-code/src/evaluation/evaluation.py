#Python standard library imports
import pickle
import re

#Related third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def eval_notes():
    pass