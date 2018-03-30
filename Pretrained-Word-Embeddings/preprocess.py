import os
import re
import nltk
import numpy as np 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split


def make_sentence(sentence):
	stopword = stopwords.words('english')

	
	sents = word_tokenize(sentence)
	new_sents = " "

	for i in range(len(sents)):
		if(sents[i].lower() not in stopword and len(sents[i])>1):
			new_sents+=sents[i].lower()+" "


	return new_sents

def preprocessing(pos,neg,labels):

	corpus = []

	for i in range(5331):
		corpus.append(make_sentence(pos[i]))

	for i in range(5331):
		corpus.append(make_sentence(neg[i]))

	train_corpus,test_corpus,label_train,label_test = train_test_split(corpus,labels,test_size=0.3,random_state=42)

	return train_corpus,test_corpus,label_train,label_test