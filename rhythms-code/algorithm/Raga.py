import warnings
warnings.filterwarnings('ignore')
import pdb
import re

import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from algorithm.ragas_and_swaras import *
from algorithm.ornamentation import *
import algorithm.quantization as quant
from algorithm.ground_truth_preprocessing import *
 
class Raga:

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


	@classmethod
	def from_dataframe(cls, df_pitch, sa_freq, raga_idx):
		"""
		Alternative constructor that accepts in-memory pitch data directly,
		avoiding the need to write/read intermediate pitch.txt and ctonic.txt files.

		Args:
			df_pitch: DataFrame with 'time' and 'f0' columns (from compiam pitch CSV).
			sa_freq: Tonic frequency in Hz.
			raga_idx: Index of the raga in ragas_and_swaras.
		"""
		# Create a minimal dummy pitches_path just to satisfy __init__ signature.
		# We won't actually read from these paths.
		dummy_paths = [""] * (raga_idx + 1)
		raga = cls(idx=raga_idx, pitches_path=dummy_paths)

		# Set tonic directly
		raga.sa = sa_freq

		# Set raga notes (swaras, log_freq) based on index
		raga.get_raga_notes(raga_idx)

		# Populate df_f0 directly from the provided DataFrame
		raga.df_f0 = df_pitch.copy()
		raga.df_f0["log_freq"] = raga.df_f0["f0"].apply(lambda x: np.log2(x) if x != 0 else 0)
		raga.df_f0["silences"] = raga.silence_column(
			raga.df_f0["f0"],
			raga.df_f0["time"].iloc[1] - raga.df_f0["time"].iloc[0]
		)

		# Detect phrases using the same logic as get_phrases
		silences = np.where(raga.df_f0["f0"] == 0.0)[0]
		raga.phrases, raga.silences = make_phrases_500ms_tresh(silences, raga.df_f0)

		raga.raga_name = f"raga_{raga_idx}"
		return raga

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
	

	def get_sa(self, ctonic_path):
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

