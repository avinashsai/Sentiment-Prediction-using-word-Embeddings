import os
import re
import numpy as np 
import pandas as pd 
import nltk
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer


def features(train_corpus,test_corpus):

	vectorizer =  TfidfVectorizer(min_df=1,max_df=1.0,use_idf=True,sublinear_tf=True,stop_words='english')

	train_corpus_tf_idf = vectorizer.fit_transform(train_corpus)

	test_corpus_tf_idf = vectorizer.transform(test_corpus)

	vocab = vectorizer.vocabulary_

	return train_corpus_tf_idf,test_corpus_tf_idf,vocab