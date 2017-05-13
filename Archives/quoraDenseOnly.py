# -*- coding: utf-8 -*-

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
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %Hh%Mm%Ss')

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
dataDense = dataALL.drop(['0', '1', '2'], axis=1).values
print(len(y),' training "lines"')
del(dataALL)

#test data
print('\nGetting testing data')
dataALL = pd.read_csv(TEST_DATA_FILE, sep='\t', encoding="utf-8")
print('test_ids')
test_ids = dataALL['0'].values
test_dataDense = dataALL.drop(['0', '1', '2'], axis=1)
print(len(test_ids),' testing "lines"')
del(dataALL)





merged_model = Sequential()

merged_model.add(Dense(300, input_shape = (dataDense.shape[1],)))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(1))
merged_model.add(Activation('sigmoid'))

merged_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ["accuracy"])

#Fitting the ANN to the Training Set

checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc', save_best_only=True, verbose=2)

merged_model.fit(dataDense, y, batch_size=4096, epochs=200,
                 verbose=2, validation_split=0.1, shuffle=True, callbacks=[checkpoint])





























