'''
The code is tested on Keras 2.0.0 using Tensorflow backend, and Python 3.5
'''
#from string import punctuation
import csv
import codecs
import pandas as pd
import numpy as np
from tqdm import tqdm
#import spacy
#nlp = spacy.load('en')
np.set_printoptions(threshold=400000)

#import itertools as it
from os.path import isfile
from collections import Counter
from pprint import pprint
from itertools import islice

#from gensim.models import KeyedVectors
#from gensim.models import Phrases
#from gensim.models.word2vec import LineSentence

from keras.models import Sequential
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input, Activation, PReLU
from keras.layers import LSTM, Embedding, Bidirectional, GRU
from keras.layers import Dropout, Merge, BatchNormalization
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

import time
import datetime
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.15
MAX_SEQUENCE_LENGTH = 40
MAX_NB_WORDS = 200000

GLOVE840 = 'F:/DS-main/BigFiles/glove.840B.300d/glove.840B.300d.txt'
TRAIN_DATA_FILE = 'F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/finalTrain.csv'
TEST_DATA_FILE  = 'F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/finalTest.csv'

STAMP = 'Quora - '+ st
print(STAMP)
del(ts)

##############################
###First Steps
##############################
#calculate eng features
#calculate scapy features
#not done yet: preprocess sentences for nlp (gensim)


##############################
###Getting data
##############################
#train data
print('\nGetting training data')
dataALL = pd.read_csv(TRAIN_DATA_FILE, sep='\t', encoding="utf-8")
y = dataALL['0'].values
texts_1 = dataALL['1'].values.astype(str)
texts_2 = dataALL['2'].values.astype(str)
dataDense = dataALL.drop(['0', '1', '2'], axis=1).values
print(len(y),' training "lines"')
del(dataALL)

#test data
print('\nGetting testing data')
dataALL = pd.read_csv(TEST_DATA_FILE, sep='\t', encoding="utf-8")
print('test_ids')
test_ids = dataALL['0'].values
print('test_texts_1')
test_texts_1 = dataALL['1'].values.astype(str)
print('test_texts_2')
test_texts_2 = dataALL['2'].values.astype(str)
test_dataDense = dataALL.drop(['0', '1', '2'], axis=1)
print(len(test_ids),' testing "lines"')
del(dataALL)

##############################
###Tokenizing
##############################
print('\nTokenizing Started')
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(list(texts_1) + list(texts_2) + list(test_texts_1) + list(test_texts_2))

print('x1')
x1 = tokenizer.texts_to_sequences(texts_1)
x1 = pad_sequences(x1, maxlen=MAX_SEQUENCE_LENGTH)
del(texts_1)

print('x2')
x2 = tokenizer.texts_to_sequences(texts_2)
x2 = pad_sequences(x2, maxlen=MAX_SEQUENCE_LENGTH)
del(texts_2)

print('test_x1')
test_x1 = tokenizer.texts_to_sequences(test_texts_1)
test_x1 = pad_sequences(test_x1, maxlen=MAX_SEQUENCE_LENGTH)
del(test_texts_1)

print('test_x2')
test_x2 = tokenizer.texts_to_sequences(test_texts_2)
test_x2 = pad_sequences(test_x2, maxlen=MAX_SEQUENCE_LENGTH)
del(test_texts_2)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

#ytrain_enc = np_utils.to_categorical(y)
#y = np.array(y)
#test_ids = np.array(test_ids)

print('Shape of data tensor x1:', x1.shape)
print('Shape of data tensor x2:', x2.shape)
print('Shape of data tensor test_x1:', test_x1.shape)
print('Shape of data tensor test_x2:', test_x2.shape)
print('Shape of label tensor:', y.shape)
print('Tokenizing Done')


##############################
###prepare embeddings glove840
##############################
print('Preparing embedding matrix')
nb_words = min(MAX_NB_WORDS, len(word_index))+1

embeddings_index = {}
f = open(GLOVE840, encoding='utf8')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))




#ValueError: could not convert string to float: '.'





