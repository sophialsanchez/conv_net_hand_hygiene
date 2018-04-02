import tensorflow as tf 
import os
import numpy as np
from more_itertools import chunked,ncycles
import random
from sklearn.model_selection import train_test_split


def get_filenames():
	print("Getting filenames...")
	# Note: Example only includes sensor 2 for
	# live demo purposes
	# Note: PAC data not included in github repo
	cur_dir = 'pac_data/02/'
	filenames = []

	subdirs = [x[0] for x in os.walk(cur_dir)]                                                                            
	for subdir in subdirs:
		files = os.walk(subdir).next()[2]     
		if (len(files) > 0):
			for file in files:
				if not file.startswith('.'):     
					curr_file = subdir + '/' + file
					filenames.append(curr_file)
	return filenames

batch_size = 32
filenames_all = get_filenames()
filenames_all = filenames_all[len(filenames_all)%batch_size:]
training_filenames, test_filenames = train_test_split(filenames_all, test_size = .1, shuffle=True)

def gen_imgs_unchunked(filenames):
	for filename in filenames:
		A = np.load(filename)
		out = np.expand_dims(A['x'].reshape(76800),0)
		yield out

gen_imgs = (np.asarray(chunk) for chunk in chunked(gen_imgs_unchunked(training_filenames),batch_size))

def labels_unchunked(filenames):
	for filename in filenames:
		output = filename.split("/")[2]
		if int(output) == 0:
			out = np.array([1,0])
		else:
			out = np.array([0,1])
		yield out

labels = (np.asarray(chunk) for chunk in chunked(labels_unchunked(training_filenames),batch_size))

def full_gen():
	print("Running full gen...")
	for x in gen_imgs: 
		y = labels.next() 
		if x.shape[0] != batch_size:
			pass
		yield np.squeeze(x),np.squeeze(y)

gen = ncycles(full_gen(),10000)

def get_test_labels():
	print("Generating test labels...")
	test_labels = []
	# Note: Example only includes subset of files
	# for live demo. To use all fies, comment
	# out the line below and uncomment the line two below
	for i in range(0, 50):
	#for i in range(0, len(test_filenames)):
		output = test_filenames[i].split("/")[2]
		if int(output) == 0:
			out = np.array([1,0])
		else:
			out = np.array([0,1])
		test_labels.append(out)
	test_labels = np.asarray(test_labels)
	return np.squeeze(test_labels)

def get_test_imgs():
	print("Generating test images..")
	test_imgs = []
	# Note: Example only includes subset of files
	# for live demo. To use all fies, comment
	# out the line below and uncomment the line two below
	for i in range(0, 50):
	#for i in range(0, len(test_filenames)):
		A = np.load(test_filenames[i])
		out = np.expand_dims(A['x'].reshape(76800),0)
		test_imgs.append(out)
	test_imgs = np.asarray(test_imgs)
	return np.squeeze(test_imgs)


