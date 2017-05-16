# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import datetime
import operator
from collections import Counter
from nltk.corpus import stopwords
from pylab import plot, show, subplot, specgram, imshow, savefig


train = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/train.csv', header=0) 
test = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/test.csv', header=0) 

questions = pd.concat([train.question1, train.question2, test.question1, test.question2]).apply(str).tolist()

from gensim.models import Phrases

#==============================================================================
# sentence_stream = [num for num, doc in enumerate(questions) if type(doc)==float]
# print(type(questions[606131]))
#==============================================================================

#print(questions[606132])
sentence_stream_q = [question.split(" ") for question in questions]
bigram_model = Phrases(sentence_stream_q)
bigram_model.save("bigrame")
bigram_model = Phrases.load("bigrame")


questionsGensim =  [bigram_model[question.split(" ")] for question in questions]

bigram_model[u'What is the step by step guide to invest in share market in india?'] 

sent = questions[3].split(" ")
print(bigram_model[sent])




from gensim.models import Phrases
documents = ["the mayor of new york was there", "machine learning can be useful sometimes","new york mayor was present"]

sentence_stream = [doc.split(" ") for doc in documents]
bigram = Phrases(sentence_stream, min_count=1, threshold=2)

sent = [u'the', u'mayor', u'of', u'new', u'york', u'was', u'there']
print(bigram[sent])