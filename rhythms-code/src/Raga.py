#Python standard library imports
import os
import re
import time
import warnings
warnings.filterwarnings('ignore')

#Related third party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#local application imports
# import preprocessing.ragas_and_swaras 
# from ornamentation import *
# import quantization as quant
# from ground_truth_preprocessing import *
 
 
class Raga:
	"""
	Defines the attributes of a Hindustani raga and segments the vocal pitch
	contour	into phrases.
	"""

	def __init__(self, idx, pitches_path):
		self.idx = idx #index of the raga
		self.sa = 0 #frequency value of the key of the composition
		
		self.raga_name = pitches_path[idx]
		self.sw = None #set of note names permitted in the raga  
		self.phrases = None #array of arrays of the phrases in the composition
		self.df_f0 = None #f0s in Hz from pitch.txt
		self.log_freq = None #log frequencies for ease of plotting
		self.silences = None #array of silences in the whole file
		self.ornamentation=None
		self.eval_df=[]


	def set_raga_object(self, pitches_path, ctonic_path):
		# xxx: make public
	    
	    self.get_sa(ctonic_path)
	    self.get_raga_notes(self.idx)
	    self.get_phrases(self.idx, pitches_path)
	    pattern = r"(?:.(?!\/))+$"
	    self.raga_name = re.search(pattern, pitches_path[self.idx])[0] 
	                                        
	
	def get_raga_object(self):
		# xxx: make public
		print("Raga Name:", self.raga_name)
		print("Sa:", self.sa) #frequency value of the key of the composition 
		print("Raga Swaras:", self.sw)  #set of note names permitted in the raga  
	

	def _get_sa(self, ctonic_path):
		# xxx: make private
		with open(ctonic_path[self.idx]) as f:
			self.sa = float(f.readlines()[0].strip())
	

	def get_raga_notes(self, idx):
		# xxx: make private
		#define the vocabulary of the raga
	    self.sw = [swaras[it] for it in range(len(swaras)) if raga_allowed_notes[raga_order[self.idx]][it]!=0] 
	    allowed_notes = raga_allowed_notes[raga_order[self.idx]]
	    '''
	    Here 
	    self.sa : frequency of the root note of the composition, 
	    allowed notes : allowed notes in the raga, acts like a mask on the allowed and omitted notes,
	    ratio : relationship of other notes with respect to the root note of the composition
	    '''
	    freq = allowed_notes*ratio*self.sa
	    final_freq = freq[freq!=0] #removing 0s here for the notes which were 0 in the mask "allowed_notes"
	    self.log_freq = np.log2(final_freq) #no need to do a try-catch here since the argument will not be 0
	        

	def get_phrases(self, idx, pitches_path):
		# xxx: make private
		# pdb.set_trace()
		
		self.df_f0=pd.read_csv(pitches_path[idx], delimiter="\t", header=None)
		self.df_f0.columns=["time", "f0"]
		self.df_f0["log_freq"]=[np.log2(x) if x!=0 else 0 for x in self.df_f0["f0"]]
		self.df_f0["silences"]=self.silence_column(self.df_f0["f0"], self.df_f0["time"].iloc[1] - self.df_f0["time"].iloc[0])
		silences = np.where(self.df_f0["f0"]==0.0)[0]
		self.phrases, self.silences = make_phrases_500ms_tresh(silences, self.df_f0)
		


	def silence_column(self, arr, delta):
		# xxx: make private
	    column_silences=np.zeros(len(arr))
	    column_silences[0]=0
	    for i in range(1, len(arr)):
	        if arr[i]==0:
	            column_silences[i] = column_silences[i-1]+delta
	        else:
	            column_silences[i] = 0
	    return column_silences  

