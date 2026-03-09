import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from quantization import *

import warnings
warnings.filterwarnings('ignore')
import pdb

'''
There are some hyperparameters here-
1. width of the convolution kernel within the function "moving_average"
2. peak prominence and
3. peak width in the function "get_peaks"
'''

def moving_average(x, w):
    '''
    Utility function to smoothen a curve.
    Args:
        x: Curve to be smoothened 
        w: convolution window size
    '''
    pad_width = int((w-1)/2)
    x=np.pad(x, pad_width)
    ma = np.convolve(x, np.ones(w), 'valid') / w

    return ma

def ma1(x,w):
    return np.array(pd.Series(x).rolling(w).mean().tolist())


def get_peaks(x, height=None, threshold=None, prominence=None, width=None, wlen=None):
    '''
    Utility function to get peaks of a curve.
    Args:
        x: Input curve
        height: vertical absolute threshold
        prominence: prominence value as per the scipy findpeaks definition
        width: horizontal absolute threshold
        
    '''
    peaks, properties = find_peaks(x, height = height, threshold=threshold, prominence=prominence, width=width , wlen=wlen)
    return peaks, properties

def get_notename_from_frequency(raga1, frequencies):
    '''
    Utility function to get note name from a list of frequency values.
    Args:
        frequencies: frequencies array
    '''
    for f in frequencies:
        # print("For f:",f, "\n", np.abs(np.log2(f) - raga1.log_freq), np.argmin(np.abs(f- raga1.log_freq)))
        return (raga1.sw[np.argmin(np.abs(np.log2(f)  - raga1.log_freq))])

def calc_instab(x_closest, frame_len, hop_len):
    '''
    Function to calculate instability scores on a given phrase
    Args:
        x_closest: quantized values of a phrase. Instability values are calculated for this array.
        frame_len: Frame length in samples
        hop_len: hop length in number of samples

    this function uses special numpy functions to exclude nan values from operations such as sum. 
    
    '''
    frame_count = int(1 + (len(x_closest) - frame_len) // hop_len)
    all_instab_values = []
    for f in range(frame_count):
        frame_start =  f*hop_len
        frame_end = f*hop_len + frame_len
        x_frame = x_closest[frame_start : frame_end + 1]
        squared_error = np.sum(np.square(x_frame-np.median(x_frame)))
        all_instab_values.append(squared_error)
    flat_instab_values = np.array(all_instab_values).flatten()
    # flat_instab_values = np.ma.array(flat_instab_values, mask=np.isnan(flat_instab_values)) #decide against masking because it didn't have impact on conv.

    #we should normalize over phrase level - see if this is necessary!!!
    if np.linalg.norm(flat_instab_values) == 0.0:
        normalized_instab_values = flat_instab_values
    else:
        normalized_instab_values = flat_instab_values/np.nansum(flat_instab_values)
    # print("normalized_instab_values:", normalized_instab_values)
    w=5
    # ma_curve = moving_average(normalized_instab_values, w)
    ma_curve = ma1(normalized_instab_values, w)    

    return ma_curve

def plot_peaks(i, ma_curve_frame, peaks, properties):
    '''
    Utility function to plot instability scores for a given curve corresponding to a raga phrase
    Args:
        i: index of the raga
        ma_curve_frame: instability score array of a phrase.
        peaks: indices of the sample peaks of ma_curve_frame as per the args provided in get_peaks
        properties: properties such as prominences, right and left bases of the peaks obtained from the get_peaks function.
    
    '''
    plt.figure(figsize=(11, 4), dpi=80)
    plt.title(f"Instability scores for {i}th phrase")
    plt.xlabel('Samples')
    plt.ylabel('Instability scores')

    plt.plot(ma_curve_frame)
    plt.plot(peaks, ma_curve_frame[peaks], "x")
    plt.vlines(x=peaks, ymin=ma_curve_frame[peaks] - properties["prominences"],
            ymax = ma_curve_frame[peaks], color = "C1")
#     plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
#               xmax=properties["right_ips"], color = "C1")
    plt.vlines(x=properties['left_bases'], ymin=ma_curve_frame[peaks] - properties["prominences"],
            ymax = ma_curve_frame[peaks], color = "C3")
    plt.vlines(x=properties['right_bases'], ymin=ma_curve_frame[peaks] - properties["prominences"],
            ymax = ma_curve_frame[peaks], color = "C3")
    plt.grid()

#1 sample in the original input curve: 4.44 ms

#Nov 4 TODO: tried out frame_len = 250 and hop_len = 5
#Nov 12 : 250 ms and 111 ms ie 56 and 25 samples as frame size are hyperparameters


def theWHERE(raga1, frame_len ,  hop_len,  y_thresh , x_thresh):
    '''
    Function to detect unstable regions for all the phrases of the raga. 
    - Iterates over each phrase of the raga, 
    - Calls calc_instab function to calculate instability values array, which returns the smooth curve
    - Calls get_peaks function to detect peaks 
    Args:
        raga object: Object containing all properties of a raga
        frame_len: Frame length in samples
        hop_len: hop length in number of samples
    Returns: 
        
    '''

    df_ornamentations = pd.DataFrame(columns = ['time', 'F0', 'orn_samples', 'dir_changes', 'unique_notes', 'orn_type'])
    df_quantized_notes = pd.DataFrame(columns = ['time_s', 'duration', 'swara'])

    # ornaments=dict()
    ma_curves_all_frames=[]
    
    for i in range(0,len(raga1.phrases)):
        if raga1.phrases[i][0]==0 and raga1.phrases[i][1] == 0:
#             print("Encountered (0,0), skipping")
            # ornaments[i] = {'starts': [] , 'ends': []} 
            df_ornamentations.loc[i] =  [[], [], [], [], [], []]
            continue
        curr_phrase = raga1.df_f0.iloc[raga1.phrases[i][0]:raga1.phrases[i][1]]
           
        curr_phrase['closest'], curr_phrase['closest_swaras'] = closest(curr_phrase['log_freq'], raga1.log_freq, raga1.sw)
        # print("i:", i,"\ncurr_phrase['closest']:", curr_phrase['closest'], "\ncurr_phrase[swaras]:",curr_phrase['closest_swaras'])

        df_temp = time_dur_sw(curr_phrase[['time','closest_swaras']])  

        df_quantized_notes=pd.concat([df_quantized_notes, df_temp], ignore_index=True)

        
            
        x_closest = np.array(curr_phrase['closest'].tolist())
        # print("x_closest", x_closest)
        # array_of_arrays_closest = list(curr_phrase['closest'])
        # x_closest = np.array([item for sublist in array_of_arrays_closest for item in sublist])
        

        #----------------------------------------------------------------------
    
        #----------------------------------------------------------------------
        ma_curve_frame = calc_instab(x_closest, frame_len, hop_len)
        # plt.plot(ma_curve_frame)
        # print("i:", i, "x_closest", x_closest, "ma_curve_frame:" , ma_curve_frame)
        ma_curves_all_frames.append(ma_curve_frame)

        '''
        Come back to this after implementing the what step for 1 file. 
        And change the hyperparameters prominence and width values.

        scipy find_peak's hyper parameters-
        Prominence : is the height beyond which to start looking for values
        Width : id the radial distance of the neighborhood, absolute width threshold
        height: absolute height threshold 
        wlen 
        threshold
        '''
        # peaks, properties= get_peaks(ma_curve_frame, prominence = 0.04, height = 0.0001) #1 NOT BEING USED ANYWHERE
        orn_samples = get_orn_samples(ma_curve_frame, y_thresh = y_thresh, x_thresh = x_thresh) #2 Maybe instead of 2 use 1.

        # if i%20 != 0:
        #     print(f"i={i}: {ma_curve_frame}, \npeaks: {peaks} \norn_samples: {orn_samples}")
        #     plot_peaks(i, ma_curve_frame, peaks, properties)

    #the WHAT
        dir_changes, original_orn_samples, unique_notes, orn_type = get_ornament_regions_and_type(x_closest, orn_samples, phrase_id=i,
                                                                                                  frame_len = frame_len ,  hop_len = hop_len, plots = False)
        df_ornamentations.loc[i] = [list(curr_phrase['time'].round(2)), 
                                    np.round(x_closest, 2), 
                                    original_orn_samples, 
                                    dir_changes,  
                                    unique_notes, 
                                    orn_type]

    # df_quantized_notes = pd.DataFrame(arr_quantized_notes, columns = ['time_s', 'duration', 'swara'])

    return df_quantized_notes, df_ornamentations

#The "WHAT"

def get_orn_samples(a, y_thresh, x_thresh):
    '''
    Function to calculate regions around the peak using vertical and horizontal threshold. 
    Currently NOT using the regions detected by the scipy peak function.
    Args:
        a: Instability curve to find unstable regions based on x_thresh and y_thresh.
        y_thresh: Absolute vertical threshold
        x_thresh: Absolute horizontal threshold, 
                default value :
                        2 if frame_size ~ 250ms or 56 samples (about 2 note) and
                        7 if frame_size ~ 100ms or 25 samples (about 1 note length).
    Returns: 
        
    '''
    if (a < y_thresh).all(): #if all values in the curve are below the vertical threshold, return no peak candidates.
        return []
#     plt.plot(a)

    res = []
    phrases=[]
    
    args = np.where(a>y_thresh)[0]
    if len(args)==1:
        res.append([args[0],args[0]])
        return res

    regions=[]
    for index in range(1, len(args)):
        if index == 1:
            region_s = args[index-1]
        
        one = args[index-1]
        two = args[index]
        diff = two - one
        
        if diff > x_thresh:            
            region_e = one
            regions.append([region_s, region_e])
            region_s = args[index]

        if index == (len(args) - 1):
                region_e = two
                regions.append([region_s, region_e])

    return regions


        

    

    # diff = np.diff(args) - 1

    # for i in range(len(diff)):
    #     curr_phrase = []
    #     # print(f"diff[{i}] = {diff[i]}")
    #     if diff[i] < x_thresh:
    #         curr_phrase.append(args[i])
    #         curr_phrase.append(args[i+1])
    #     # if curr_phrase != []:
    #     phrases.append(curr_phrase)

    # print("Args:", args, "diff", diff, "Phrases:", phrases)

    # index = 0

    # #Skip non zero phrases
    # # if (phrases[index] == []):
    # #     continue
    # if index < len(phrases): #there is at least one 
    #     start = phrases[index][0]
    #     start_id = index + 1
    # else: 
    #     start = phrases[0][0]
    #     start_id = 1
    # # print("phrases:", phrases)

    # for i in range(start_id, len(phrases)):
    #     if phrases[i]==[]: 
    #         print(f"phrases[{i}]==[]. start_id: {start_id}. phrases[{i-1}][1]: {phrases[i-1][1]}, phrases[{i+1}]: {phrases[i+1]}")
    #         res.append([start, phrases[i-1][1]])
    #         start = phrases[i+1][0]
    #     if i == len(phrases)-1:
    #         print(f"start_id : {start_id}")
    #         if len(phrases[i])>0:
    #             res.append([start, phrases[i][1]])

    # print("Res:", res)
        
    # return res

#the WHAT
def get_ornament_regions_and_type(x_closest, res, phrase_id, frame_len = 25, hop_len = 5, plots = False):
    
    #rename "res"
    
    dir_changes=[]
    og_samples = []
    unique_notes = []
    orn_type = []

    for orn_id in range(len(res)):
        
        ur_start = res[orn_id][0]
        ur_end = res[orn_id][-1]
        s = ur_start*hop_len
        e = ur_end*hop_len + frame_len
        
        chunk = x_closest[s:e] #x_closest is already quantized, not smoothened though!
        # plt.plot(chunk)
        # plt.title('chunk')
        # plt.show()

        unique = np.unique(chunk)
        

        '''
        Smoothing it here to detect number of direction changes.
        Don't want it to disrupt the unique number of notes (because 
        it was going till log frequency of ~4)
        '''
        smooth_chunk =moving_average(chunk, 5)
        
        dir_change = np.count_nonzero(np.diff(np.sign(np.diff(smooth_chunk)[np.diff(smooth_chunk)!=0])))
        
        dir_changes.append(dir_change)
        og_samples.append([s,e])
        unique_notes.append(len(unique)) 
        
        if dir_change <= 2:
            if len(smooth_chunk)<=56: #checking number of consecutive unstable frames. 
                                     #Note this is not = 7 because 7 was calculated in the 
                                     #unstable frame scale.
                orn_type.append(1)  #kan
            else:
                orn_type.append(2)  #meend1
        else:
            if len(unique) == 1: #for andolan, we probably should use the original curve
                                 #not the quantized:curr_phrase instead of x_closest
                orn_type.append(4)  #andolan
            elif len(unique) == 2 or len(unique) == 3:
                orn_type.append(3)  #meend2
            elif len(unique) > 3:
                orn_type.append(5)  #murki

    # non_overlap_orn_samples = remove_overlap_orn_samples(og_samples)

    return dir_changes , og_samples, unique_notes, orn_type

def remove_overlap_orn_samples(og_samples):
   
    if len(og_samples) <= 1:    
        return og_samples
    og_new = []
    tentative = og_samples[0] 
    for ind in range(1, len(og_samples)):
        one = tentative
        two = og_samples[ind]
    #     print(one, two)
        if two[0] <= one[1]:
            tentative = [one[0], two[1]]
    #         print(two[0], "<=", one[1], "new tentative:", tentative)
        else:
    #         print(two[0], ">", one[1], "appending tentative:", tentative, "new tentative:", og_samples[ind])
            og_new.append(tentative)
            tentative = og_samples[ind]
        
        if ind == (len(og_samples) - 1):
            og_new.append(tentative)
    return og_new


def ornament_plot(orig_x, orig_y, quantized_y, orn_x):
        
        plt.figure(figsize=(4, 2), dpi=80)
        plt.title("Original F0")
        plt.plot(orig_x, orig_y)
        plt.show()
        
        plt.figure(figsize=(4, 2), dpi=80)
        plt.title("Quantized F0")
        plt.plot(orig_x, quantized_y)
        low_id = np.where(raga1.log_freq == min(quantized_y))[0][0]
        high_id = np.where(raga1.log_freq == max(quantized_y))[0][0] 

        plt.yticks(raga1.log_freq[low_id -4 :high_id + 4])
        plt.show()
        
        plt.figure(figsize=(4, 2), dpi=80)
        plt.title("Instability")
        plt.plot(orn_x)
        plt.tight_layout()
        plt.show()