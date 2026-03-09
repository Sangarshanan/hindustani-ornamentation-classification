#Python standard library imports
# import os
import pdb

#Related third party imports
# import numpy as np

#local imports
from Raga import Raga
from preprocessing.ground_truth_preprocessing import InputData

def main():
	print("imported everything")
	
	#load saraga f0 annotations
	path = "../Saraga/f0-annotations"
	f0input = InputData(path)

	args = define_args()
	print("args:", args.idx, args.frame, args.jump, args.xthresh, args.ythresh, 'chosen raga: ', input_data_obj.pitches_path[args.idx[0]])
	
	
	#make raga object for the raga index
	raga = Raga(args.idx[0], input_data_obj.pitches_path)
	pdb.set_trace()




if __name__ == '__main__':
	main()