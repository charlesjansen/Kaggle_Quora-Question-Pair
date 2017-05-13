'''
The code is tested on Keras 2.0.0 using Tensorflow backend, and Python 3.5
'''
#from string import punctuation
import csv
import codecs
import numpy as np
import pandas as pd
import spacy
nlp = spacy.load('en')
np.set_printoptions(threshold=400000)

#import itertools as it
from os.path import isfile
from collections import Counter
from pprint import pprint
from itertools import islice

from gensim.models import KeyedVectors
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Bidirectional, GRU, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

import helperQuoraSpacy
from helperQuoraSpacy import preprocess_training_data
from helperQuoraSpacy import loading_training_data
from helperQuoraSpacy import preprocess_test_data
from helperQuoraSpacy import loading_test_data
from helperQuoraSpacy import scapyCounts
from helperQuoraSpacy import wordCompare
from helperQuoraSpacy import text_to_wordlist

MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.15

GOOGLE_DIR = 'F:/DS-main/BigFiles/'
EMBEDDING_FILE  =  GOOGLE_DIR + 'GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILE = 'F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/input/train.csv'
TEST_DATA_FILE  = 'F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/input/test.csv'
PREPROCESSED_TRAIN_DATA_FILE = 'F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/input/preprocessed_train.csv'
PREPROCESSED_TEST_DATA_FILE  = 'F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/input/preprocessed_test.csv'
SEQUENCED_TRAIN_DATA_FILE = 'F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/input/sequenced_train.csv'
SEQUENCED_TEST_DATA_FILE  = 'F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/input/sequenced_test.csv'

num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)

rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

act = 'relu'
re_weight = True # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'Spacy-gru_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
        rate_drop_dense)
print(STAMP)

########################################
## text preprocessing
########################################
if not isfile(PREPROCESSED_TRAIN_DATA_FILE):
    texts_1, texts_2, labels = preprocess_training_data(TRAIN_DATA_FILE, PREPROCESSED_TRAIN_DATA_FILE)
else:
    texts_1, texts_2, labels = loading_training_data(PREPROCESSED_TRAIN_DATA_FILE)
  
####Scapy
if (1==1):##process only once, from original text. Saved into file after
    scapyData1 = scapyCounts(texts_1)
    scapyData2 = scapyCounts(texts_2)
    wordCompare = wordCompare(texts_1, texts_2)
    derivedData = np.concatenate((scapyData1, scapyData2), axis=1)
    derivedData = np.concatenate((derivedData,wordCompare), axis=1)
else:
    pass #load from file
    

###Test
if not isfile(PREPROCESSED_TEST_DATA_FILE):
    test_texts_1, test_texts_2, test_ids = preprocess_test_data(TEST_DATA_FILE, PREPROCESSED_TEST_DATA_FILE)
else:
    test_texts_1, test_texts_2, test_ids = loading_test_data(PREPROCESSED_TEST_DATA_FILE)

if (1==1):##process only once, from original text. Saved into file after
    scapyTestData1 = scapyCounts(test_texts_1)
    scapyTestData2 = scapyCounts(test_texts_2)
    wordCompareTest = wordCompare(test_texts_1, test_texts_2)
    derivedTestData = np.concatenate((scapyTestData1, scapyTestData2), axis=1)
    derivedTestData = np.concatenate((derivedData, wordCompareTest), axis=1)
else:
    pass #load from file
    

########################################
## Texts sequence of int
########################################
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)

sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

print('Ex of sequences', sequences_1[:5])

########################################
## Size of sequences picking "MAX_SEQUENCE_LENGTH"
## Sequence preprocessing, removing 0 lenght sequences
########################################

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

all_sequences = sequences_1 + sequences_2

#Udacity Style
sequences_lens = Counter([len(x) for x in all_sequences])
print("Zero-length sentence: {}".format(sequences_lens[0]))
print("Maximum sentence length: {}".format(max(sequences_lens)))

print("Removing 0 length")
non_zero_idx1 = [ii for ii, seq in enumerate(sequences_1) if len(seq) != 0 ]
sequences_1 = [sequences_1[ii] for ii in non_zero_idx1]
sequences_2 = [sequences_2[ii] for ii in non_zero_idx1]
labels = np.array([labels[ii] for ii in non_zero_idx1])

all_sequences = sequences_1 + sequences_2
sequences_lens = Counter([len(x) for x in all_sequences])
print(len(all_sequences))
print("Zero-length sentence: {}".format(sequences_lens[0]))
print("Maximum sentence length: {}".format(max(sequences_lens)))

#MAX_SEQUENCE_LENGTH = max(sequences_lens)
MAX_SEQUENCE_LENGTH = 30
print(MAX_SEQUENCE_LENGTH)

########################################
## Padding
########################################

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_ids = np.array(test_ids)

########################################
## Cleaning RAM
########################################

del(texts_1, texts_2, test_texts_1, test_texts_2, sequences_1, sequences_2, test_sequences_1, test_sequences_2,
    all_sequences, sequences_lens)

########################################
## index word vectors
########################################
print('Indexing word vectors')

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
        binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))



########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))


del(word_index)
########################################
## sample train/validation data
########################################
#np.random.seed(1234)
perm = np.random.permutation(len(data_1))
idx_train = perm[:int(len(data_1)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(data_1)*(1-VALIDATION_SPLIT)):]

data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

weight_val = np.ones(len(labels_val))
if re_weight:
    weight_val *= 0.472001959
    weight_val[labels_val==0] = 1.309028344

print("train/validation data split done")
########################################
## define the model structure
########################################
#What is the correct way for stacking Bidirectional RNNs using Keras?
#output1 = Bidirectional(GRU(64, return_sequences=True))(input)
#output2 = Bidirectional(GRU(64))(output1)
#https://github.com/fchollet/keras/issues/5022
#
#


embedding_layer = Embedding(nb_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)
gru_layer = Bidirectional(GRU(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=True))

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = gru_layer(embedded_sequences_1)
x1 = gru_layer(x1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = gru_layer(embedded_sequences_2)
y1 = gru_layer(y1)

merged = concatenate([x1, y1])

merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

preds = Dense(1, activation='sigmoid')(merged)

########################################
## add class weight
########################################
if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None

########################################-------------------------------------------------------------------
## train the model
########################################


model = Model(inputs=[sequence_1_input, sequence_2_input], \
        outputs=preds)
model.compile(loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['acc'])

#model.load_weights(bst_model_path)#************************reprendre les derniers weights entrainées


#model.summary()
print(STAMP)

early_stopping =EarlyStopping(monitor='val_loss', patience=4)
bst_model_path = "2gru" + STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([data_1_train, data_2_train], labels_train, \
        validation_data=([data_1_val, data_2_val], labels_val, weight_val), \
        epochs=100, batch_size=384, shuffle=True, \
        class_weight=class_weight, callbacks=[early_stopping, model_checkpoint], verbose=2)

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])



########################################
## make the submission
########################################
print('Start making the submission before fine-tuning')

preds = model.predict([test_data_1, test_data_2], batch_size=512, verbose=2)
preds += model.predict([test_data_2, test_data_1], batch_size=512, verbose=2)
preds /= 2

submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds.ravel()})
submission.to_csv('%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)