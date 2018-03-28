get_ipython().system(u'conda install gensim -y')

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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier


nltk.download('punkt')
nltk.download('stopwords')


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


with open('finalpolarity.txt','r',encoding='latin1') as f:
    data = f.readlines()


stopword = stopwords.words('english')


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

model = Word2Vec(sg=1,seed=1,size=100, min_count=2, window=10,sample=1e-4)

model.build_vocab(corpus_tokens)

model.train(corpus_tokens,total_examples=model.corpus_count,epochs=50)


print(len(model.wv.vocab))


vectorizer = TfidfVectorizer(min_df=3,max_df=0.8,use_idf=True,sublinear_tf=True,stop_words='english')

train_corpus_tf_idf = vectorizer.fit_transform(train_corpus)

test_corpus_tf_idf = vectorizer.transform(test_corpus)

print(train_corpus_tf_idf.shape)

vocab = vectorizer.vocabulary_

print(model.most_similar('good'))

corpus_train_vecs = np.zeros((train_length,100))

for i in range(train_length):
    su = np.zeros(100)
    count = 0
    sents = word_tokenize(train_corpus[i])
    for j in range(len(sents)):
        if(sents[j] in vocab and sents[j] in model.wv.vocab):
            su+=model[sents[j]]
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
        if(sents[j] in vocab and sents[j] in model.wv.vocab):
            su+=model[sents[j]]
            count+=1
        else:
            continue
    if(count>0):
        corpus_test_vecs[i] = su/count


classifier_list = [LogisticRegression(),SVC(gamma=0.9),SVC(kernel='linear',gamma=0.5),BernoulliNB(),DecisionTreeClassifier()]


def Classifier(classifier,train_data,test_data,train_labels,test_labels):
    classifier = classifier
    classifier.fit(train_data,train_labels)
    predict = classifier.predict(test_data)
    acc = accuracy_score(test_labels,predict)
    cm = confusion_matrix(test_labels,predict)
    f1 = f1_score(test_labels,predict)
    return acc,cm,f1

classifiers = ["LR","SVC-RBF","SVC-L","BNB","DT"]

accuracy = []
F1_score = []

for classifier in classifier_list:
    acc,cm,f1 = Classifier(classifier,corpus_train_vecs,corpus_test_vecs,label_train,label_test)
    accuracy.append(acc)
    F1_score.append(f1)


print(accuracy)

print(F1_score)


import matplotlib.pyplot as plt

plt.xlabel("classifiers")
plt.ylabel("Accuracy Scores")
plt.bar(classifiers,accuracy,align='center',alpha=0.8,color='r')
plt.show()


plt.xlabel("classifiers")
plt.ylabel("F1 Scores")
plt.bar(classifiers,F1_score,align='center',alpha=0.8,color='g')
plt.show()

