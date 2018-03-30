from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,confusion_matrix


def classify_sentiment(classifier,train_corpus,test_corpus,label_train,label_test):
    clf   = classifier
    clf.fit(train_corpus,label_train)
    pred = clf.predict(test_corpus)
    accuracy = clf.score(test_corpus,label_test)
    cm = confusion_matrix(pred,label_test)
    f1 = f1_score(pred,label_test)
    return accuracy,f1,cm 


def classify(train_corpus_tf_idf,test_corpus_tf_idf,train_corpus_vecs,test_corpus_vecs,label_train,label_test):
	classifiers = [LogisticRegression(random_state=0),SVC(),DecisionTreeClassifier(random_state=0),
                   SVC(kernel='linear',gamma=0.9),BernoulliNB(),KNeighborsClassifier(n_neighbors=7)]
    

    for i in range(len(classifiers)):
    	acc_tf,f1_tf,cm_tf = classify_sentiment(classifiers[i],train_corpus_tf_idf,test_corpus_tf_idf,label_train,label_test)
    	acc_w,f1_w,cm_w = classify_sentiment(classifiers[i],train_corpus_vecs,test_corpus_vecs,label_train,label_test)
    	print(acc_tf,end=" ")
    	print(acc_w)
    	print(f1_tf,end=" ")
    	print(f1_w)
    	print(cm_tf)
    	print(cm_w)
    	print("---------------------------------------")   
