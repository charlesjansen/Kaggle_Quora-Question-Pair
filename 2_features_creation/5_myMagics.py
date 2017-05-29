# -*- coding: utf-8 -*-
laptop = 0
if laptop == 1:
    drive = "C"
else:
    drive = "F"

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import defaultdict

TRAIN_DATA_FILE = drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/train.csv'
TEST_DATA_FILE  = drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/test.csv'

print("loading train")
train_orig =  pd.read_csv(TRAIN_DATA_FILE, header=0)
print("loading test")
test_orig =  pd.read_csv(TEST_DATA_FILE, header=0)
print("data loaded")

ques = pd.concat([train_orig['question1'], train_orig['question2'], 
        test_orig['question1'], test_orig['question2']], axis=0).reset_index(drop='index')
ques.shape

words = ques.astype(str).values
all_words = " ".join(words)
words = all_words.split()

del all_words, ques

from collections import Counter
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii-1 for ii, word in enumerate(vocab, 1)}

#plus un mot est rare plus la question est unique, moins elle a de chance d'etre une dupliqué
#donner pour chaque question le mot le plus rare, son id
#donner pour chaque question la somme des mots (plus rare sera plus grand)
#donner la soustraction de ces sommes au carré, pour une meme ligne.

train_orig = train_orig.astype(str).fillna("empty")
test_orig = test_orig.astype(str).fillna("empty")


def rarest_word(text):
    text = set(text.split())
    max_int = 0
    for word in text:
        word_int = vocab_to_int[word]
        if word_int >= max_int:
            max_int = word_int
    return max_int

def averaged_sum_words(text):
    text = set(text.split())
    sum_int = 0
    for word in text:
        sum_int += vocab_to_int[word]
    return sum_int/len(set(text))



#==============================================================================
# 
# text = "hello all ? charles slaves"
# text2 = "why should  do that ?"
# rarest_word(text)
# averaged_sum_words(text)
# lineDiff_of_averaged_sum_words(text2, text)
# 
#==============================================================================
print("rarestWordID1")
train_orig['rarestWordID1'] = train_orig.question1.apply(lambda x: rarest_word(x))
print("rarestWordID2")
train_orig['rarestWordID2'] = train_orig.question2.apply(lambda x: rarest_word(x))

print("avgWordID1")
train_orig['avgWordID1'] = train_orig.question1.apply(lambda x: averaged_sum_words(x))
print("avgWordID2")
train_orig['avgWordID2'] = train_orig.question2.apply(lambda x: averaged_sum_words(x))

print("diffAvgWordID")
train_orig['diffAvgWordID'] = abs(train_orig.avgWordID1 - train_orig.avgWordID2)

print("diffRarestWordID")
train_orig['diffRarestWordID'] = abs(train_orig.rarestWordID1 - train_orig.rarestWordID2)


train_orig = train_orig.drop(['id', 'is_duplicate', 'qid1', 'qid2'], axis=1)
train_orig = train_orig.drop(['question1', 'question2'], axis=1)





print("rarestWordID1")
test_orig['rarestWordID1'] = test_orig.question1.apply(lambda x: rarest_word(x))
print("rarestWordID2")
test_orig['rarestWordID2'] = test_orig.question2.apply(lambda x: rarest_word(x))

print("avgWordID1")
test_orig['avgWordID1'] = test_orig.question1.apply(lambda x: averaged_sum_words(x))
print("avgWordID2")
test_orig['avgWordID2'] = test_orig.question2.apply(lambda x: averaged_sum_words(x))

print("diffAvgWordID")
test_orig['diffAvgWordID'] = abs(test_orig.avgWordID1 - test_orig.avgWordID2)

print("diffRarestWordID")
test_orig['diffRarestWordID'] = abs(test_orig.rarestWordID1 - test_orig.rarestWordID2)


test_orig = test_orig.drop(['test_id'], axis=1)
test_orig = test_orig.drop(['question1', 'question2'], axis=1)


train_orig.to_csv(drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/train_myMagic.csv') 
test_orig.to_csv(drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/test_myMagic.csv')

























