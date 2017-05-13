# -*- coding: utf-8 -*-

import tensorflow as tf 
from sklearn.model_selection import train_test_split
import csv
import codecs
import pandas as pd
import numpy as np
from tqdm import tqdm
np.set_printoptions(threshold=400000)


TRAIN_DATA_FILE = 'F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/finalTrain.csv'

y_pd = pd.read_csv(TRAIN_DATA_FILE, sep='\t', encoding="utf-8", usecols=[0])
preg1_pd = pd.read_csv(TRAIN_DATA_FILE, sep='\t', encoding="utf-8", usecols=[1])
preg2_pd = pd.read_csv(TRAIN_DATA_FILE, sep='\t', encoding="utf-8", usecols=[2])

y = y_pd.values
preg1 = preg1_pd.values
preg2 = preg2_pd.values

y = [ligne[0] for _,ligne in enumerate(y)]
preg1 = [str(ligne[0]) for _,ligne in enumerate(preg1)]
preg2 = [str(ligne[0]) for _,ligne in enumerate(preg2)]
y[0:8]

preg1temp = preg1
print(len(y))
print(len(preg1))
print(len(preg2))
y.extend(y)
preg1 = preg1 + preg2
preg2.extend(preg1temp)
print(len(y))
print(len(preg1))
print(len(preg2))
del(preg1temp)

print(preg1[:3])
print(preg2[:3])
print(y[:3])
print(preg1[404290:404293])
print(preg2[404290:404293])
print(y[404290:404293])

all_words = ' '.join(preg1)
words = all_words.split()

all_words[:1000]

words[:20]

from collections import Counter
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii-1 for ii, word in enumerate(vocab, 1)}

preg1_ints = []
preg2_ints = []
for each in preg1:
    preg1_ints.append([vocab_to_int[word]+1 for word in each.split()])
for each in preg2:
    preg2_ints.append([vocab_to_int[word]+1 for word in each.split()])
###+1 because I pad with 0 and I had a word 0

vocab[:5]

y = np.array(y)

print(len(preg1_ints))
print(type(preg1_ints))

preg1_lens = Counter([len(x) for x in preg1_ints])
print("Zero-length preg1: {}".format(preg1_lens[0]))
print("Maximum preg1 length: {}".format(max(preg1_lens)))

preg2_lens = Counter([len(x) for x in preg2_ints])
print("Zero-length preg2: {}".format(preg2_lens[0]))
print("Maximum preg2 length: {}".format(max(preg2_lens)))

seq_len = 40
features1 = np.zeros((len(preg1_ints), seq_len), dtype=int)
for i, row in enumerate(preg1_ints):
    features1[i, -len(row):] = np.array(row)[:seq_len]

features1[0:10]

seq_len = 40
features2 = np.zeros((len(preg2_ints), seq_len), dtype=int)
for i, row in enumerate(preg2_ints):
    features2[i, -len(row):] = np.array(row)[:seq_len]

features2[0:5]

type(features1)

features = np.concatenate((features1,np.zeros((len(features1), 5), dtype=int),features2), axis=1)

features[0:5]

del(y_pd, preg1_pd, preg2_pd, preg1, preg2, preg1_ints, preg2_ints, features1, features2)


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Merge
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D, SpatialDropout1D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text


# Number of Epochs
epochs = 60
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 50
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 13
decoding_embedding_size = 13
# Learning Rate
learning_rate = 0.001

encoding_dim = 32

model = Sequential()
#model.add(Embedding(len(vocab) + 1, 300, input_length=85))
model.add(Embedding(90000 + 1, 300, input_length=85))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(300))
model.add(PReLU())
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint('wweights.h5', monitor='val_acc', save_best_only=True, verbose=1)

model.fit(features, y=y, batch_size=128, epochs=20,
                 verbose=1, validation_split=0.1, shuffle=True, callbacks=[checkpoint])


########################## test data processing
del(y, features)

TEST_DATA_FILE = 'F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/finalTest.csv'

test_ids_pd = pd.read_csv(TEST_DATA_FILE, sep='\t', encoding="utf-8", usecols=[0])
preg1_pd = pd.read_csv(TEST_DATA_FILE, sep='\t', encoding="utf-8", usecols=[1])
preg2_pd = pd.read_csv(TEST_DATA_FILE, sep='\t', encoding="utf-8", usecols=[2])

test_ids_pd = test_ids_pd.values
preg1 = preg1_pd.values
preg2 = preg2_pd.values
del(preg1_pd, preg2_pd)

test_ids_pd = [ligne[0] for _,ligne in enumerate(test_ids_pd)]
preg1 = [str(ligne[0]) for _,ligne in enumerate(preg1)]
preg2 = [str(ligne[0]) for _,ligne in enumerate(preg2)]
test_ids_pd[0:8]

preg1temp = preg1
print(len(test_ids_pd))
print(len(preg1))
print(len(preg2))


print(preg1[:3])
print(preg2[:3])
print(test_ids_pd[:3])
print(preg1[404290:404293])
print(preg2[404290:404293])
print(test_ids_pd[404290:404293])


#==============================================================================
# preg1_ints = []
# preg1_ints.append('1')
# preg1_ints.append('')
# preg1_ints.append('1')
# print(preg1_ints)
#==============================================================================
print(len(vocab))
print(vocab[231655])
preg1_ints = []
preg2_ints = []
for each in preg1:
    preg1_ints.append([vocab_to_int[word]+1 if word in vocab_to_int else 231656 for word in each.split()])#missing words from training becomes 0
for each in preg2:
    preg2_ints.append([vocab_to_int[word]+1 if word in vocab_to_int else 231656 for word in each.split()])
###+1 because I pad with 0 and I had a word 0


#==============================================================================
# 
# preg1_ints = []
# preg2_ints = []
# count1 = 0
# count2 = 0
# for each in preg1:
#     for word in each.split():
#         if word in vocab:
#             preg1_ints.append(vocab_to_int[word]+1)#missing words from training are removes (I never trained on them)
#         else:
#             count1 += 1
# print("count1 missing word in training that are in test ", count1)
# for each in preg2:
#     for word in each.split():
#         if word in vocab:
#             preg2_ints.append(vocab_to_int[word]+1)
#         else:
#             count2 += 1
# print("count2 missing word in training that are in test ", count2)
# 
# ###+1 because I pad with 0 and I had a word 0
# 
#==============================================================================

del(preg1, preg2)

vocab[:5]

test_ids = np.array(test_ids_pd)

print(len(preg1_ints))
print(type(preg1_ints))

preg1_lens = Counter([len(x) for x in preg1_ints])
print("Zero-length preg1: {}".format(preg1_lens[0]))
print("Maximum preg1 length: {}".format(max(preg1_lens)))

preg2_lens = Counter([len(x) for x in preg2_ints])
print("Zero-length preg2: {}".format(preg2_lens[0]))
print("Maximum preg2 length: {}".format(max(preg2_lens)))



#manual fix, but could be done by looking the ids of the index in an array
for i, row in enumerate(preg1_ints):
    if len(row) == 0:
        #row = [" "]
        print(i," - ",row)
preg1_ints[94646] =  [231656]
preg1_ints[714289] =  [231656]


for i, row in enumerate(preg2_ints):
    if len(row) == 0:
        #row = [" "]
        print(i," - ",row)
preg2_ints[2164990] =  [231656]





seq_len = 40
features1 = np.zeros((len(preg1_ints), seq_len), dtype=int)
for i, row in enumerate(preg1_ints):
    features1[i, -len(row):] = np.array(row)[:seq_len]

features1[0:10]

features2 = np.zeros((len(preg2_ints), seq_len), dtype=int)
for i, row in enumerate(preg2_ints):
    features2[i, -len(row):] = np.array(row)[:seq_len]

features2[0:5]

type(features1)

features = np.concatenate((features1,np.zeros((len(features1), 5), dtype=int),features2), axis=1)

features[0:5]

del(preg1_ints, preg2_ints, features1, features2)








#model.load_weights(bst_model_path)
#bst_val_score = min(model.history['val_loss'])
print(bst_val_score)
########################################
## make the submission
########################################
print('Start making the submission before fine-tuning')

#==============================================================================
# preds1 = model.predict(features, batch_size=128, verbose=1)
# preds2 = model.predict([test_data_2, test_data_1], batch_size=128, verbose=1)
# predsMerge = (preds1+preds2)/2
# 
# submission1 = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds.ravel()})
# submission1.to_csv('%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)
# 
#==============================================================================


preds = model.predict(features, batch_size=128, verbose=1)

submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds.ravel()})
submission.to_csv('rnnStyle1 20170507 21h.csv', index=False)
































