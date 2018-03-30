import os
import re
import numpy as np 
import pandas as pd 
import nltk
import sklearn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim
from gensim.models import Word2Vec

import tensorflow as tf 
import keras

from readfiles import *
from preprocess import *
from featureextraction import *
from extractvectors import *
from train import *


if __name__ == '__main__':
	pos,neg,labels = read_files()

	train_corpus,test_corpus,label_train,label_test = preprocessing(pos,neg,labels) 

	train_corpus_tf_idf,test_corpus_tf_idf,vocab = features(train_corpus,test_corpus)

	#print(train_corpus_tf_idf[0:2])
	#print(test_corpus_tf_idf[0:2])

	train_corpus_vecs,test_corpus_vecs,train_vecs,test_vecs = getvectors(train_corpus,test_corpus,vocab)

	#print(train_corpus_vecs[0:2])

	#print(test_corpus_vecs)

	classify(train_corpus_vecs,test_corpus_vecs,train_vecs,test_vecs,label_train,label_test)






