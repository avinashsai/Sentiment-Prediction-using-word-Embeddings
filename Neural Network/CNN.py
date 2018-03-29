import os
import re
import pandas as pd
import numpy as np
import sklearn
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stopword = stopwords.words('english')

with open('finalpolarity.txt','r',encoding='latin1') as f:
    data = f.readlines()

def preprocess(sentence):
    sentence = re.sub(r'[^\w\s]'," ",sentence)
    sentence = re.sub(r'[^a-zA-Z]'," ",sentence)
    sents = sentence.split()
    return sents

labels = np.zeros(2000)
labels[0:1000] = 1

corpus_tokens = []

corpus = []

for i in range(2000):
    cor_to = preprocess(data[i])
    corpus_tokens.append(cor_to)
    cor = data[i]
    cor = cor[0:len(cor)-1]
    corpus.append(cor)

train_corpus,test_corpus,label_train,label_test = train_test_split(corpus,labels,test_size=0.3,random_state=42)

train_length = len(train_corpus)
test_length = len(test_corpus)

import gensim
from gensim.models import Word2Vec

model_word2vec = Word2Vec(sg=1,seed=1,size=100, min_count=2, window=10,sample=1e-4)

model_word2vec.build_vocab(corpus_tokens)

model_word2vec.train(corpus_tokens,total_examples=model_word2vec.corpus_count,epochs=50)

model_word2vec.save('word2vec_vectors_2000.txt')

model_word2vec = Word2Vec.load('word2vec_vectors_2000.txt')

print(len(model_word2vec.wv.vocab))

vectorizer = TfidfVectorizer(min_df=3,max_df=0.8,use_idf=True,sublinear_tf=True,stop_words='english')

train_corpus_tf_idf = vectorizer.fit_transform(train_corpus)

test_corpus_tf_idf = vectorizer.transform(test_corpus)

print(train_corpus_tf_idf.shape)

vocab = vectorizer.vocabulary_

print(model_word2vec.wv.most_similar('good'))

corpus_train_vecs = np.zeros((train_length,100))

for i in range(train_length):
    su = np.zeros(100)
    count = 0
    sents = word_tokenize(train_corpus[i])
    for j in range(len(sents)):
        if(sents[j] in vocab and sents[j] in model_word2vec.wv.vocab):
            su+=model_word2vec[sents[j]]
            count+=1
        else:
            continue
    if(count>0):
        corpus_train_vecs[i] = su/count

corpus_test_vecs = np.zeros((test_length,100))

for i in range(test_length):
    su = np.zeros(100)
    count = 0
    sents = word_tokenize(test_corpus[i])
    for j in range(len(sents)):
        if(sents[j] in vocab and sents[j] in model_word2vec.wv.vocab):
            su+=model_word2vec[sents[j]]
            count+=1
        else:
            continue
    if(count>0):
        corpus_test_vecs[i] = su/count

import tensorflow as tf
import random as rn

os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(42)
rn.seed(12345)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten,Conv1D,MaxPooling1D
from keras import metrics

corpus_train_vecs = corpus_train_vecs.reshape((corpus_train_vecs.shape[0],100,1))

corpus_test_vecs = corpus_test_vecs.reshape((test_length,100,1))

model = Sequential()
model.add(Conv1D(64,kernel_size=3,input_shape=(100,1)))
model.add(Conv1D(32,kernel_size=3))
model.add(Dropout(0.24))
model.add(Conv1D(16,kernel_size=3))
model.add(Conv1D(8,kernel_size=3))
model.add(Conv1D(4,kernel_size=3))
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32,activation='elu'))
model.add(Dropout(0.25))
model.add(Dense(8,activation='elu'))
model.add(Dense(4,activation='elu'))
model.add(Dense(1,activation='sigmoid'))

print(model.summary())

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

model.fit(corpus_train_vecs,label_train,batch_size=4,epochs=50)

pred = model.predict(corpus_test_vecs)

y_test = np.zeros(test_length)

for i in range(test_length):
  if(pred[i]>0.5):
    y_test[i] = 1

print(sum(y_test==label_test)/test_length)

