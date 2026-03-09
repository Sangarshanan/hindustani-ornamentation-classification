This directory contains the code for the algorithm to transcribe Hindustani vocals. Following describes the purpose of each file.
- ragas_and_swaras.py : containing dictionaries with the swara vocabulary (i.e. 3 octave in our case), allowed notes and the sequence of ragas to be read by the algorithm.

- Raga.py : File containing the abstraction of a raga. Each 'raga' has the following attributes-
	- idx : index of the raga
	- sa : frequency of the F0 of the raga rendition
	- sw : list of names of the permitted swaras in the raga
	- phrases : array of arrays of the phrases in the composition
	- df_f0 : f0s in Hz from pitch.txt file of the raga
	- log_freq : log frequencies for ease of plotting
	- silences : array of silences in the whole file

	Each raga also has a setter and getter function to access these attributes.

- quantization.py : this file contains the code for quantizing the F0 pitch contour into swaras.

- ornamentation.py : this file contains the code for detecting peaks and regions of instability and their amounts to then classify them into each type of ornament.

- evaluation.py : this file contains the code for the evaluation of the quantization and ornamentation detection part of the algorithm.

- ground_truth_preprocessing.py : this file contains functions to preprocess and some generic utility functions.

