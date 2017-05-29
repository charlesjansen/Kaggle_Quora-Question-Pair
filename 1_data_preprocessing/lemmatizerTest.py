# -*- coding: utf-8 -*-
"""
Created on Tue May 16 20:17:47 2017

@author: Charles
"""
laptop = 0
if laptop == 1:
    drive = "C"
else:
    drive = "F"

import spacy
import pandas as pd
import numpy as np
nlp = spacy.load('en')

def lemmatizer(text):        
    sent = []
    doc = nlp(text)
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)


TEST_DATA_FILE  = drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/test.csv'

dataTest = pd.read_csv(TEST_DATA_FILE, sep=',', encoding='utf-8')

dataTest['question1'] = dataTest['question1'].fillna("empty")
print("lemma spacyQ1_test")
dataTest["spacyQ1_test"] =  dataTest.apply(lambda x: lemmatizer(x['question1']), axis=1)

dataTest['question2'] = dataTest['question2'].fillna("empty")
print("lemma spacyQ2_test")
dataTest["spacyQ2_test"] =  dataTest.apply(lambda x: lemmatizer(x['question2']), axis=1)


dataTest['question1'] = dataTest["spacyQ1_test"]
dataTest['question2'] = dataTest["spacyQ2_test"]

dataTest = dataTest.drop(['spacyQ1_test', 'spacyQ2_test'], axis=1)

dataTest.to_csv(drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/test_spacyLemma.csv', index=False, encoding="utf-8")



























#==============================================================================
# 
# from spacy.en import English
# parser = English()
# 
# multiSentence = u"this is spacy lemmatize testing. programming books are more better than others"
# 
# parsedData = parser(multiSentence)
# 
# for i, token in enumerate(parsedData):
#     print("original:", token.orth, token.orth_)
#     print("lowercased:", token.lower, token.lower_)
#     print("lemma:", token.lemma, token.lemma_)
#     print("shape:", token.shape, token.shape_)
#     print("prefix:", token.prefix, token.prefix_)
#     print("suffix:", token.suffix, token.suffix_)
#     print("log probability:", token.prob)
#     print("Brown cluster id:", token.cluster)
#     print("----------------------------------------")
#     if i > 1:
#         break
#     
#==============================================================================

#==============================================================================
# 
# from nltk.stem.wordnet import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
# lemmatizer.lemmatize("is", 'v')
# 
# print(lemmatizer.lemmatize('provided', 'v'))
# 
# print(lemmatizer.lemmatize('using', 'v'))
# 
# print(lemmatizer.lemmatize('a', 'v'))
# 
# print(lemmatizer.lemmatize('contracting', 'v'))
# 
# print(lemmatizer.lemmatize('gonna', 'v'))
# 
# print(lemmatizer.lemmatize("ain't", 'v'))
# 
# 
#==============================================================================
