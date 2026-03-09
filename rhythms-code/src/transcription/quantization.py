#Related third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def get_closest_note(log_freq, arr, sw):
    note_names = sw
    note_log_freq = log_freq

    phrase=[]
    for k in range(len(arr)):
        ids=np.where(abs(note_log_freq - arr[k]) == min(abs(note_log_freq - arr[k])))
        sur = note_names[ids[0][0]]
        phrase.append(sur)

    final= []
    final.append(phrase[0])
    for k in range(1,len(phrase)):
        if phrase[k-1] == phrase[k]:
            continue
        else:
            final.append(phrase[k])
    return final, phrase

def process_phrase(x, n_frame, n_hop):
    # x=x[x!=np.nan]

    frame_count = int(1 + (len(x) - n_frame) // n_hop)
    print(len(x), frame_count)
    notes_array = np.zeros(frame_count)
    notes_indices = []
    if frame_count!=0:
        notes_array[0] = 0
        for i in range(1, frame_count):
            # Slice the k'th frame
            start = (i * n_hop)
            end = (i* n_hop) + n_frame
            x_frame = x[start : end +1]
            diff = x_frame - np.median(x_frame)
            if (diff==0).all() == True:
                notes_array[i] = np.median(x_frame)
            else: 
                notes_array[i] = notes_array[i-1]
            
            notes_indices.append([start, end])
    # plt.plot(notes_array)

    # print("frame_count:", frame_count, "notesArr:", len(notes_array))
    return notes_array, notes_indices

def make_phrases(silences, df):
    print("df columns:", df.columns)
    raga_phrases=[]
    count=0
    for i in range(len(silences)-1):
        if df["f0"].iloc[silences[i]] - df["f0"].iloc[silences[i+1]] > 0.5: #>500ms for Breath phrases
            # print("diff:", df["time"].iloc[silences[i]] - df["time"].iloc[silences[i+1]])
            logfreq=df["log_freq"][silences[i]+1:silences[i+1]]
            raga_phrases.append(logfreq)
    return raga_phrases

def make_phrases_500ms_tresh(silences, df):
    phr = []
    trailing_silences = 0
    start=0
    end = 0
    all_silence_durations=[]
    time_delta = df["time"].iloc[1]-df["time"].iloc[0]
    for i in range(2, len(df["silences"])):
        if df["silences"].iloc[i-1]!=0 and df["silences"].iloc[i]==0:    # silence arr: |non-zero| 0 |. Thus, start of phrase region found, end of silence found
            duration = df["time"].loc[end] - df["time"].loc[start] 
            
            if duration != 0:
                # print("Found end of silence region :", duration)
                all_silence_durations.append(duration)
            if trailing_silences>0.5: #and (end-start) > 56 : #phrase length is at least 500ms, trailing_silences
                #valid phrase when inter phrase is >500ms and 
                #each phrase is at least a note's length long
                phr.append([start,end])
                start =  i
                
        if df["silences"].iloc[i-1]==0 and df["silences"].iloc[i]!=0: #CANDIDATE : phrase ends #  silence arr:| 0 |non-zero|  : Silence begins
            trailing_silences = 0
            end=i
        
        if df["silences"].iloc[i-1]!=0 and df["silences"].iloc[i]!=0: 
                trailing_silences += time_delta #silence's cumulative duration
    
    return phr, all_silence_durations

def get_df(phrasesL, phrase_num, df): 
    return df.iloc[phrasesL[phrase_num].index.start:phrasesL[phrase_num].index.stop]

def to_groundtruth_format(str_arr):
    '''
    Get the notes in the ground truth format. This is only for evaluation.
    '''
    str_arr=[x[0] for x in str_arr]
    res="".join(str_arr)
    return res

def plot_quant(DF, log_freq, sw):
#     print("here", sw)
    fig, ax = plt.subplots(2,1, figsize=(10,8))
    ax[0].plot(DF["time"],DF["log_freq"])
    ax[0].set_xlabel("Time in s")
    ax[0].set_ylabel("Frequency relative to tonic")
    ax[0].set_yticks(log_freq, sw)
    ax[0].grid()
    ax[0].set_title('original_f0s')

    ax[1].plot(DF["time"],DF["closest"])
    ax[1].set_xlabel("Time in s")
    ax[1].set_ylabel("Frequency relative to tonic")
    ax[1].set_yticks(log_freq, sw)
#     print("log_freq, sw:",log_freq, sw)
    ax[1].grid()
    ax[1].set_title('quantized_f0s')
    plt.tight_layout()
    plt.show()
import seaborn as sns

def new_plot_quant(DF, log_freq, sw):
    print("Within new_plot_quant:",len(log_freq), len(sw))       
    plt.yticks(log_freq, sw)
    # plt.plot(DF["time"].index,DF["log_freq"], label="Original")
    plt.xlabel("Time in s")
    plt.ylabel("Frequency relative to tonic")
    plt.plot(DF["time"].index,DF["closest_to_plot"],'+' ,label="Quantized")    
    plt.grid(linestyle='--', alpha=0.3)   
    plt.title('Original_f0s and Quantized_f0s overlay')
    plt.tight_layout()

    plt.plot(DF["time"].index,DF["log_freq"],'--' ,label="orig")    
    plt.legend()

    plt.show()

    
def closest(lf, notes, swara):
    '''
    Function finds the closest log frequency values to the log frequencies
    args lf: Pandas series containing log frequencies of a phrase in the raga 
    notes: log frequencies of the 36 allowed notes.

    returns frequencies and note names of the snapped pitch quantized phrase.
    '''
    closest_notes = []
    swara_array = []

    for j in range(len(lf)):

        if lf.iloc[j] == 0:
            closest_notes.append(0)
            swara_array.append('REST')
        else:
            diff = notes - lf.iloc[j]
            ids=np.where(abs(diff) == min(abs(diff)))[0][0]

            swara_array.append(swara[ids])
            closest_notes.append(notes[ids])
            
    #use binary search
    
    return closest_notes, swara_array

def time_dur_sw(df_closest_swaras):
    ''' 
    Gets the quantized notes in the format time<>duration<>swara 
    '''
    df_res = pd.DataFrame(columns = ['time_s', 'duration', 'swara'])

    time_s = df_closest_swaras.iloc[0].time

    min_dur = df_closest_swaras.iloc[1].time - df_closest_swaras.iloc[0].time
    dur = min_dur
    for i in range(len(df_closest_swaras)-1):
        # print("i:", i, " swara:", df_closest_swaras.iloc[i].closest_swaras)
        # if i == 223:
        # if df_closest_swaras.iloc[i].closest_swaras.isalnum() == False:
            # pdb.set_trace()
        if df_closest_swaras.iloc[i].closest_swaras == df_closest_swaras.iloc[i+1].closest_swaras :
            dur += min_dur
        else:
            entry = [time_s, dur, df_closest_swaras.iloc[i].closest_swaras]
            df_res.loc[len(df_res.index)] = entry
            time_s = df_closest_swaras.iloc[i+1].time

    return df_res

def get_pace(phrase_start, phrase_end):
    """
    extracts the pace of the phrase out of 0, 1, 2: being slow, medium and fast.
    This function is not being used anymore

    """
    print(phrase_start, phrase_end)
    if (phrase_start>= np.floor(df_tempo_arr[0]["Start(sec)"].iloc[0]) and
        phrase_end < np.ceil(df_tempo_arr[0]["End(sec)"].iloc[0])):
        return 0
    elif (phrase_start>= np.floor(df_tempo_arr[1]["Start(sec)"].iloc[1]) 
    and phrase_end < np.ceil(df_tempo_arr[1]["End(sec)"].iloc[1])):       
        return 1
    elif (phrase_start>= np.floor(df_tempo_arr[2]["Start(sec)"].iloc[2])
     and phrase_end < np.ceil(df_tempo_arr[2]["End(sec)"].iloc[2])):
        return 2
def quantize(ph, frame_len, hop_len, log_freq, sw):

    ph['closest']=closest(ph['log_freq'], log_freq)
    ph['closest_to_plot'] = ph['closest'].replace(0, np.nan) #maybe this is a incorrect place for this.
 
    time = ph['time']
    n_frames = frame_len
    hop_len = hop_len
    
    notesArr, _=process_phrase(np.array(ph['closest']), n_frames, hop_len) #returns the phrase's value. So, need to change within this for processing ornamentation
    if len(notesArr) == 0 and len(np.array(ph['closest']))>56: #unlikely because of the condition of minimum phrase length to be 56.
        print("here -1")
        return -1
    else:
        m = (np.array(notesArr))!=0
        notesArr=notesArr[m.argmax(): ]
        time=np.array(time)[m.argmax():]

        if len(time) == 0 or len(notesArr) == 0:
            print("here -2")
            return -2

        phrase=get_closest_note(log_freq, notesArr, sw)
        print(phrase)
        # duration = round(time[-1]-time[0], 2) 

        mod_phrase = to_groundtruth_format(phrase)
#         print(mod_phrase)
        # new_plot_quant(ph.iloc[m.argmax():], log_freq, sw)
        return mod_phrase

# def quantize_T(ph, frame_len, hop_len, log_freq, sw):

#     ph['closest']=closest(ph['log_freq'], log_freq)
#     ph['closest_to_plot'] = ph['closest'].replace(0, np.nan)
#     time = ph['time']

#     # pdb.set_trace()

#     stab, notesArr=stab_T(np.array(ph['closest']), frame_len, hop_len) #returns the phrase's value. So, need to change within this for processing ornamentation
    
#     print(stab)
#     normalized = stabs/np.linalg.norm(stabs)
#     peaks = find_peaks(normalized)
#     pdb.set_trace()
  
                
#     return stab
def stab_T(x, n_frame, n_hop):
    x=x[x!=0]
    # print("len(x):", len(x))
    frame_count = int(1 + (len(x) - n_frame) // n_hop)
    # print('total frames:', frame_count)
    notes_array = np.zeros(frame_count)

    # if frame_count!=0:
    #     notes_array[0] = 0
    #     for i in range(1, frame_count):
    #         # Slice the k'th frame
    #         start = (i * n_hop)
    #         end = (i* n_hop) + n_frame
    #         x_frame = x[start : end +1]
    #         diff = x_frame - np.median(x_frame)
    #         if (diff==0).all() == True:
    #             notes_array[i] = np.median(x_frame)
    #         else: 
    #             notes_array[i] = notes_array[i-1]

    
    plt.plot(x)
    
    # Initialize the output array
    # We have frame_count frames 
    stab = np.zeros(frame_count)
    
    # Populate each frame's results
    for k in range(frame_count):
        # Slice the k'th frame
        start = k * n_hop
        end = (k * n_hop) + n_frame
        x_frame = x[start : end+1]
        # Take the sum of squares between the median frame value and each point
        # pdb.set_trace() 
        ss = np.sum(x_frame - np.median(x_frame)) ** 2
        # print("Stability of ", k,"th frame:",ss)
        stab[k] = ss

    normalized = stab/np.linalg.norm(stab)
    peaks = find_peaks(normalized)

    if frame_count!=0:
        notes_array[0] = 0

    return stab, notes_array
