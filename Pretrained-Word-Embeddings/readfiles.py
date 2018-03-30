import os
import re
import numpy as np 


def read_files():

	with open('Dataset/rt-polarity.pos','r',encoding='latin1') as f:
		pos = f.readlines()

	with open('Dataset/rt-polarity.neg','r',encoding='latin1') as f:
		neg = f.readlines()

	labels = np.zeros(10662)
	labels[0:5331] = 1

	return pos,neg,labels