# -*- coding: utf-8 -*-
'''
The code is tested on Keras 2.0.0 using Tensorflow backend, and Python 3.5
'''
#from string import punctuation
import csv
import codecs
import numpy as np
import pandas as pd
np.set_printoptions(threshold=400000)

#import itertools as it
from os.path import isfile

dataTrainAb = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/train_featuresAbhis.csv' , encoding = "ISO-8859-1")
dataTestAb = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/test_featuresAbhis.csv' , encoding = "ISO-8859-1")

dataTrainScapy = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/featuresScapyCountTrain.csv' , encoding = "utf-8")
dataTestScapy = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/featuresScapyCountTest.csv' , encoding = "utf-8")

dataTrain = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/train.csv' , encoding = "utf-8")
dataTest = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/test.csv' , encoding = "utf-8")
dataTrain = dataTrain.drop(['id', 'qid1', 'qid2',"question1","question2"], axis=1)
dataTest = dataTest.drop(["question1","question2"], axis=1)

dataTrain = np.array(dataTrain)
dataTrainAb = np.array(dataTrainAb)
dataTrainScapy = np.array(dataTrainScapy)
dataTrainFinal = np.concatenate((dataTrain, dataTrainAb, dataTrainScapy), axis=1)

dataTrainFinal = pd.DataFrame(dict(enumerate(dataTrainFinal.T)))
dataTrainFinal.to_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/finalTrain.csv'  , index=False, encoding="utf-8", sep='\t')
#del(data, dataTrainScapy, dataTrainAb)


dataTest = np.array(dataTest)
dataTestAb = np.array(dataTestAb)
dataTestScapy = np.array(dataTestScapy)
dataTestFinal = np.concatenate((dataTest, dataTestAb, dataTestScapy), axis=1)

dataTestFinal = pd.DataFrame(dict(enumerate(dataTestFinal.T)))
dataTestFinal.to_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/finalTest.csv'  , index=False, encoding="utf-8", sep='\t')
#del(dataTest, dataTestScapy, dataTestAb)