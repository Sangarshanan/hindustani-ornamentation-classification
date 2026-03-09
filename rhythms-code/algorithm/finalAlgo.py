import sys
import argparse
import numpy as np
import pdb

# import Raga
from Raga import *
# from ornamentation import *
# from quantization import *
# import quantization 
# import ornamentation
from evaluation import *
from ground_truth_preprocessing import *


parser = argparse.ArgumentParser()

# Custom arguments are added here. By default, argparse provides a Help argument. It can be accessed using:
# python finalsynth.py -h
def define_args():
	#File Name
	parser.add_argument('-i', '--idx',  nargs='+',type=int,
		help='Provide the index of the file to be processed.'
              # 'Ex: finalAlgo -p RaagBhairav.mp3')
              'Ex: finalAlgo -i 5')

	#File processing arguments
	parser.add_argument('-f', '--frame', nargs='+',
		 default=56, type=int,
		help='Choose the frame size to be used for processing the file.'
              'Ex: finalAlgo -f 56', required=False)

	parser.add_argument('-j', '--jump',  nargs='+',type=int,
		 default=5, 
		help='Choose the hop size to be used for processing the file.'
              'Ex: finalAlgo -j 5', required=False)

	parser.add_argument('-x', '--xthresh', nargs='+',type=int,
		default=3,
		help='Choose the x-threshold to be used for detecting ornamentations.'
              'Ex: finalAlgo -x 3', required=False)

	parser.add_argument('-y', '--ythresh', nargs='+',type=float,
		 default=0.001,
		help='Choose the y-threshold to be used for detecting ornamentations.'
              'Ex: finalAlgo -y 0.001', required=False)
	# Output file
	parser.add_argument('-o', '--output', nargs='+',
		help='Provide the name of the output file in txt.'
			  'Ex: finalAlgo -o my_output.txt')
	
	return parser.parse_args()


# Get the list of effects chosen by the user.
# def get_argument_list():
#     # sys.argv contains the full command in list form that was called by the user. 
#     # We can loop through it looking for effects flags.
#     # It's not elegant, but it works.

#     arg_list = []
#     for arg in sys.argv:
#         if arg == '-p' or arg == '--path':
#             arg_list.append('path')
#         elif arg == '-f' or arg == '--frame':
#             arg_list.append('frame')
#         elif arg == '-h' or arg == '--hop':
#             arg_list.append('hop')
#         elif arg == '-x' or arg == '--xthresh':
#             arg_list.append('xthresh')
#         elif arg == '-y' or arg == '--ythresh':
#             arg_list.append('ythresh')
#         elif arg == '-o' or arg == '--output':
#             arg_list.append('output')
        

#     return arg_list


if __name__ == '__main__':
	
	input_data_obj = InputData('../f0-annotations/input') #Pre-processing 

	args = define_args()

	# args_list = get_argument_list()
	print("args:", args.idx, args.frame, args.jump, args.xthresh, args.ythresh, 'chosen raga: ', input_data_obj.pitches_path[args.idx[0]])


	raga = Raga(args.idx[0], input_data_obj.pitches_path)
	raga.set_raga_object(input_data_obj.pitches_path, input_data_obj.ctonic_path)
	raga.ornamentation =  theWHERE(raga, frame_len =  args.frame[0], hop_len = args.jump[0] , y_thresh = args.ythresh[0] , x_thresh = args.xthresh[0])
	# notes, orns =  theWHERE(raga, frame_len =  args.frame[0], hop_len = args.jump[0] , y_thresh = args.ythresh[0] , x_thresh = args.xthresh[0])
	# np.savetxt(args.output[0], raga.ornamentation.values)
	# print("swara:\n",raga.ornamentation[0])
	raga.ornamentation[0].round({'time_s': 2, 'duration': 2}).to_csv(args.output[0], index=None, sep='\t', mode='a')
	raga.ornamentation[1].round({'time': 2, 'F0': 2}).to_csv(args.output[0], index=None, sep='\t', mode='a')
	# 
	print("File has been written")
