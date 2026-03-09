#This file contains functions to preprocess and some generic utility functions

import re
import os
import pandas as pd
import numpy as np

class InputData:
    def __init__(self, path):
        self.ground_truth_notes, self.pitches_path, self.ctonic_path, self.tempo_path = self.initialize(path)

    def initialize(self, path):
        reg=r"[.+?].+"
        mphrases=[]
        pitches=[]
        ctonic=[]
        tempo=[]
        dire =[x[2] for x in os.walk(path)]
        for files in dire:
            for file in files:
                
                match = re.search(reg, file)
                if match.group(0) == '.mphrases-manual.txt':
                    mphrases.append(file)
                if match.group(0) == '.pitch.txt':
                    pitches.append(file)
                if match.group(0) == '.ctonic.txt':
                    ctonic.append(file)
                if match.group(0) == '.tempo-manual.txt':
                    tempo.append(file)

        ground_truth_notes=[]
        pitches_path=[]
        ctonic_path=[]
        tempo_path=[]
        for i,x in enumerate(os.walk(path)):
            if i>0 and i <54:

                
                a=x[0]+"/"+mphrases[i-1]
                df_gt=pd.read_csv(a, delimiter="\t", header=None)
                df_gt.columns=["time", "type", "duration", "phrase"]
                ground_truth_notes.append(df_gt)
                pitches_path.append(x[0]+"/"+pitches[i-1])
                ctonic_path.append(x[0]+"/"+ctonic[i-1])
                tempo_path.append(x[0]+"/"+tempo[i-1])
        df_tempo_arr=[]
        for file in tempo_path:
            each_df = pd.read_csv(file,delimiter=',', names=['Tempo(bpm)', 'a','b','c','Start(sec)', 'End(sec)'])
            each_df=each_df.drop(['a','b','c'], axis=1)    
            df_tempo_arr.append(each_df)
        return ground_truth_notes, pitches_path, ctonic_path, tempo_path
