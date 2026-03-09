#Python standard library imports
import argparse


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