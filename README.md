# Sentiment-Prediction-using-word-Embeddings

Word Embeddings proved to be effective features in sentiment classification. During Sentiment Prediction Word Embeddings are averaged across each sentence to derive feature for a single sentence.This repository consists of new approach for calculating word vectors for each sentence. Instead of averaging all the word vectors, we use **TF-IDF** to select top most words for each sentence and average those vectors for each sentence. The word vectors calculated for each sentence are trained using Machine Learning Algorithms and Neural Networks. The word vectors are calculated without any pretrained word Embeddings like Glove and Google Word2vec corpus. Pre-trained Word Embeddings folder contains comparison of both methods using Pre trained Word vectors.

# Getting Started

The approach is tested on the **Polarity v2.0** dataset[http://www.cs.cornell.edu/people/pabo/movie-review-data]. It consists of 1000 positive review and 1000 negative review documents.We used 70% for training and 30% for testing. Machine Learning folder consists of code for proposed method trained on Machine Learning Algorithms and CNN.

# Prerequisites 
This new approach requires knowledge about word Embeddings. If you haven't learnt about word Embeddings I advise you to go through https://arxiv.org/abs/1301.3781 this paper. The word Embeddings are trained on entire dataset. TF-IDF scores are calculated and words with high TF-IDF scores are kept in vocabulary. For each sentence if words are present in vocabulary the word vectors are added and finally averaged for each sentence. The word vectors obtained are then trained using Machine Learning Algorithms and CNN.

The approach is also tested on Pre Trained Word Emeddings Glove. The code for the approach is in Pre trained Word Embedding folder.

# Installation

These are the packages needed to be installed 

**1. Gensim**  
```
pip3 install gensim

```
**2. NLTK** 
```
sudo pip3 install -U nltk

```
**3.Tensorflow**

Install Tensorflow using this https://www.tensorflow.org/install/

**Keras**

Install Keras using this https://keras.io/#installation

# Running the Tests

Machine Learning folder consists of code trained using Machine Learning Classifiers and Neural networks folder consists of code trained using  ConvNets.

To run the files without Pre-trained Word Embeddings

```
git clone https://github.com/avinashsai/Sentiment-Prediction-using-word-Embeddings.git 

```

To execute WordEmbeddings with TF-IDF using  Machine Learning Classifiers

```
cd Machine Learning/Scripts
python3 main.py

```
To execute WordEmbeddings with TF-IDF using ConvNets

```
cd Neural Network
python3 CNN.py

```
Embeddings after training:

```
('well', 0.8155489563941956), ('really', 0.7915444374084473), ('bad', 0.7796602845191956), ('movie', 0.7706460952758789), ('pretty', 0.7576977014541626), ('acting', 0.7528560757637024), ('even', 0.751783549785614), ('like', 0.7510882616043091), ('isnt', 0.7507451176643372), ('much', 0.7412222027778625)

```
Word Embeddings showed much similar words that are closely related. These Word Embeddings are calculated on the Dataset and Pre trained word Embeddings are not used



To run files with pre trained Word Embeddings:

Install Glove using this https://pypi.python.org/pypi/glove/1.0.2.

```
cd Pretrained Word Embeddings
python3 main.py

```
You will get results tested with all machine learning classifiers.
