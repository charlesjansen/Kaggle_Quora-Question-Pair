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

from gensim.models import KeyedVectors
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence

import helperQuoraSpacy
from helperQuoraSpacy import preprocess_training_data
from helperQuoraSpacy import loading_training_data
from helperQuoraSpacy import preprocess_test_data
from helperQuoraSpacy import loading_test_data
from helperQuoraSpacy import scapyCounts
from helperQuoraSpacy import wordCompare
from helperQuoraSpacy import text_to_wordlist


TRAIN_DATA_FILE = 'F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/input/train.csv'
TEST_DATA_FILE  = 'F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/input/test.csv'



########################################
## text preprocessing
########################################

texts_1, texts_2, labels = loading_training_data(TRAIN_DATA_FILE)
  
####Training set
scapyData1 = scapyCounts(texts_1)
scapyData2 = scapyCounts(texts_2)
wordCompareData = wordCompare(texts_1, texts_2)
derivedData = np.concatenate((scapyData1, scapyData2), axis=1)
derivedData = np.concatenate((derivedData,wordCompareData), axis=1)
    
featuresScapyCountTrain = pd.DataFrame(dict(enumerate(derivedData.T)))
featuresScapyCountTrain.to_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/input/featuresScapyCountTrain.csv'
                               , index=False, encoding="utf-8")
del(featuresScapyCountTrain)

###Test
test_texts_1, test_texts_2, test_ids = loading_test_data(TEST_DATA_FILE)

scapyTestData1 = scapyCounts(test_texts_1)
scapyTestData2 = scapyCounts(test_texts_2)
wordCompareTest = wordCompare(test_texts_1, test_texts_2)
derivedTestData = np.concatenate((scapyTestData1, scapyTestData2), axis=1)
derivedTestData = np.concatenate((derivedTestData, wordCompareTest), axis=1)
    
featuresScapyCountTest = pd.DataFrame(dict(enumerate(derivedTestData.T)))
featuresScapyCountTest.to_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/input/featuresScapyCountTest.csv'
                              , index=False, encoding="utf-8")
del(featuresScapyCountTest)


























