#https://www.kaggle.com/tour1st/magic-feature-v2-0-045-gain

laptop = 0
if laptop == 1:
    drive = "C"
else:
    drive = "F"

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import defaultdict

TRAIN_DATA_FILE = drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/train_spacyLemma.csv'
TEST_DATA_FILE  = drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/test_spacyLemma.csv'

train_orig =  pd.read_csv(TRAIN_DATA_FILE, header=0)
test_orig =  pd.read_csv(TEST_DATA_FILE, header=0)

ques = pd.concat([train_orig[['question1', 'question2']], \
        test_orig[['question1', 'question2']]], axis=0).reset_index(drop='index')
ques.shape

q_dict = defaultdict(set)
for i in range(ques.shape[0]):
        q_dict[ques.question1[i]].add(ques.question2[i])
        q_dict[ques.question2[i]].add(ques.question1[i])

def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

train_orig['q1_q2_intersect'] = train_orig.apply(q1_q2_intersect, axis=1, raw=True)
test_orig['q1_q2_intersect'] = test_orig.apply(q1_q2_intersect, axis=1, raw=True)

train_orig[['q1_q2_intersect']].to_csv(drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/train_magic2_spacyLemma.csv') 
test_orig[['q1_q2_intersect']].to_csv(drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/test_magic2_spacyLemma.csv')


