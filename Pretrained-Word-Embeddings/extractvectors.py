import os
import re
import numpy as np 
import pandas as pd 
import nltk
import sklearn

import gensim
from gensim.models import Word2Vec

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

	

def makevectors(sent,vocab):
	vecs = np.zeros(100)

	new_vecs = np.zeros(100)

	count1 = 0
	count = 0

	senti = sent.split()
	for i in range(len(senti)):
		if(senti[i] in model.wv.vocab and senti[i] in vocab):
			count+=1
			vecs += model[senti[i]]
		if(senti[i] in model.wv.vocab):
			new_vecs+=model[senti[i]]
			count1+=1
	
	if(count>0):
		vecs = vecs/count
	if(count1>0 and count==0):
		new_vecs = new_vecs/count1
		vecs = new_vecs
	if(count1==0 and count==0):
		vecs = new_vecs
	return vecs,new_vecs



def getvectors(train_corpus,test_corpus,vocab):

	word2vec_file = "test_word2vec.txt"

	model = KeyedVectors.load_word2vec_format(word2vec_file)


	train_corpus_vecs = np.zeros((len(train_corpus),100))
	train_vecs = np.zeros((len(train_corpus),100))

	test_corpus_vecs = np.zeros((len(test_corpus),100))
	test_vecs = np.zeros((len(test_corpus),100))

	for i in range(len(train_corpus)):
		train_corpus_vecs[i],train_vecs[i] = makevectors(train_corpus[i],vocab)

	for i in range(len(test_corpus)):
		test_corpus_vecs[i],test_vecs[i] = makevectors(test_corpus[i],vocab)

	return train_corpus_vecs,test_corpus_vecs,train_vecs,test_vecs


	
