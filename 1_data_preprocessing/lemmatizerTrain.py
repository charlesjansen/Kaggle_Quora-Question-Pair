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


TRAIN_DATA_FILE = drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/train.csv'

data = pd.read_csv(TRAIN_DATA_FILE, sep=',', encoding='utf-8')

data['question1'] = data['question1'].fillna("empty")
print("lemma spacyQ1_train")
data["spacyQ1_train"] =  data.apply(lambda x: lemmatizer(x['question1']), axis=1)

data['question2'] = data['question2'].fillna("empty")
print("lemma spacyQ2_train")
data["spacyQ2_train"] =  data.apply(lambda x: lemmatizer(x['question2']), axis=1)

data['question1'] = data["spacyQ1_train"]
data['question2'] = data["spacyQ2_train"]

data = data.drop(['spacyQ1_train', 'spacyQ2_train'], axis=1)

data.to_csv(drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/train_spacyLemma.csv', index=False, encoding="utf-8")























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
