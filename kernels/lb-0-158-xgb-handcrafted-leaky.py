# -*- coding: utf-8 -*-
#https://www.kaggle.com/act444/lb-0-158-xgb-handcrafted-leaky?scriptVersionId=1188812

import argparse
import functools
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb

from nltk.corpus import stopwords
from collections import Counter
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split

from xgboost import XGBClassifier


def word_match_share(row, stops=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        if word not in stops:
            q1words[word] = 1
    for word in row['question2']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

def jaccard(row):
    wic = set(row['question1']).intersection(set(row['question2']))
    uw = set(row['question1']).union(row['question2'])
    if len(uw) == 0:
        uw = [1]
    return (len(wic) / len(uw))

def common_words(row):
    return len(set(row['question1']).intersection(set(row['question2'])))

def total_unique_words(row):
    return len(set(row['question1']).union(row['question2']))

def total_unq_words_stop(row, stops):
    return len([x for x in set(row['question1']).union(row['question2']) if x not in stops])

def wc_diff(row):
    return abs(len(row['question1']) - len(row['question2']))

def wc_ratio(row):
    l1 = len(row['question1'])*1.0 
    l2 = len(row['question2'])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def wc_diff_unique(row):
    return abs(len(set(row['question1'])) - len(set(row['question2'])))

def wc_ratio_unique(row):
    l1 = len(set(row['question1'])) * 1.0
    l2 = len(set(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def wc_diff_unique_stop(row, stops=None):
    return abs(len([x for x in set(row['question1']) if x not in stops]) - len([x for x in set(row['question2']) if x not in stops]))

def wc_ratio_unique_stop(row, stops=None):
    l1 = len([x for x in set(row['question1']) if x not in stops])*1.0 
    l2 = len([x for x in set(row['question2']) if x not in stops])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def same_start_word(row):
    if not row['question1'] or not row['question2']:
        return np.nan
    return int(row['question1'][0] == row['question2'][0])

def char_diff(row):
    return abs(len(''.join(row['question1'])) - len(''.join(row['question2'])))

def char_ratio(row):
    l1 = len(''.join(row['question1'])) 
    l2 = len(''.join(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def char_diff_unique_stop(row, stops=None):
    return abs(len(''.join([x for x in set(row['question1']) if x not in stops])) - len(''.join([x for x in set(row['question2']) if x not in stops])))


def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)
    
def tfidf_word_match_share_stops(row, stops=None, weights=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        if word not in stops:
            q1words[word] = 1
    for word in row['question2']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

def tfidf_word_match_share(row, weights=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        q1words[word] = 1
    for word in row['question2']:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


def build_features(data, stops, weights):
    X = pd.DataFrame()
    f = functools.partial(word_match_share, stops=stops)
    print("word_match")
    X['word_match'] = data.apply(f, axis=1, raw=True) #1

    print("tfidf_wm")
    f = functools.partial(tfidf_word_match_share, weights=weights)
    X['tfidf_wm'] = data.apply(f, axis=1, raw=True) #2

    print("tfidf_wm_stops")
    f = functools.partial(tfidf_word_match_share_stops, stops=stops, weights=weights)
    X['tfidf_wm_stops'] = data.apply(f, axis=1, raw=True) #3

    print("jaccard")
    X['jaccard'] = data.apply(jaccard, axis=1, raw=True) #4
    print("wc_diff")
    X['wc_diff'] = data.apply(wc_diff, axis=1, raw=True) #5
    print("wc_ratio")
    X['wc_ratio'] = data.apply(wc_ratio, axis=1, raw=True) #6
    print("wc_diff_unique")
    X['wc_diff_unique'] = data.apply(wc_diff_unique, axis=1, raw=True) #7
    print("wc_ratio_unique")
    X['wc_ratio_unique'] = data.apply(wc_ratio_unique, axis=1, raw=True) #8

    print("wc_diff_unq_stop")
    f = functools.partial(wc_diff_unique_stop, stops=stops)    
    X['wc_diff_unq_stop'] = data.apply(f, axis=1, raw=True) #9
    print("wc_ratio_unique_stop")
    f = functools.partial(wc_ratio_unique_stop, stops=stops)    
    X['wc_ratio_unique_stop'] = data.apply(f, axis=1, raw=True) #10

    print("same_start")
    X['same_start'] = data.apply(same_start_word, axis=1, raw=True) #11
    print("char_diff")
    X['char_diff'] = data.apply(char_diff, axis=1, raw=True) #12

    print("char_diff_unq_stop")
    f = functools.partial(char_diff_unique_stop, stops=stops) 
    X['char_diff_unq_stop'] = data.apply(f, axis=1, raw=True) #13

    print("common_words")
    #X['common_words'] = data.apply(common_words, axis=1, raw=True)  #14
    print("total_unique_words")
    X['total_unique_words'] = data.apply(total_unique_words, axis=1, raw=True)  #15

    print("total_unq_words_stop")
    f = functools.partial(total_unq_words_stop, stops=stops)
    X['total_unq_words_stop'] = data.apply(f, axis=1, raw=True)  #16
    
    print("char_ratio")
    X['char_ratio'] = data.apply(char_ratio, axis=1, raw=True) #17    

    return X




laptop = 0
if laptop == 1:
    drive = "C"
else:
    drive = "F"

preprocessing = "_spacy_cleaned"
#preprocessing = "_MyMagic3_spacyLemma_trigram"


import numpy as np
import pandas as pd
import operator
from sklearn.model_selection import train_test_split
from collections import Counter
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from pylab import plot, show, subplot, specgram, imshow, savefig
from sklearn.model_selection import KFold
K = 5
kf = KFold(n_splits = K)
#==============================================================================
# print(list(enumerate(kf.split([1,2,3,4,5,6,7,8,9]))))   
# X= np.array(["a","b","c","d","e","f","g","h","i"])
# for kth, (train_index, test_index) in enumerate(kf.split(X)):
#     print(test_index)
#     max = np.amax(test_index)+1
#     min = np.amin(test_index)
#     print(X[min:max])
#==============================================================================

np.set_printoptions(threshold=400000)
pd.set_option('display.max_rows', 2000, 'display.max_columns', 2000,  'display.show_dimensions', 'truncate')


RS = 12357
np.random.seed(RS)

print("Loading data")
input_folder = drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/'
df_train = pd.read_csv(input_folder + 'train.csv')
df_test  = pd.read_csv(input_folder + 'test.csv')
df_test.drop(["question1", "question2"], axis=1, inplace=True)

#adding final x features
print("Loading x final features")
#x = pd.read_csv(drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/x_final_features.csv', header=0) 
#x = pd.read_csv(drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/x_final_features_expended.csv', header=0) 
#x = pd.read_csv(drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/x_final_features_lowRemoved.csv', header=0) 
x = pd.read_csv(drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/x_final_features' + preprocessing  + '.csv', header=0) 
#x = x.drop(["rarestWordID1", "rarestWordID2", 'max_kcore', "tfidf_word_match", "wordMatchShare"], axis=1)

#corr = np.corrcoef(x)
print(x.columns)
print(x.describe())

feature_names = list(x.columns.values)
print("Features: {}".format(feature_names))

x_train = x[:df_train.shape[0]].astype("float64").reset_index()
x_test_real  = x[df_train.shape[0]:].astype("float64").reset_index()
labels = df_train['is_duplicate'].values
x_train_real =  x_train
labels_real =  labels
del df_train, x

print("setting parser")
parser = argparse.ArgumentParser(description='XGB with Handcrafted Features')
parser.add_argument('--save', type=str, default='XGB_leaky',
                    help='save_file_names')
args = parser.parse_args()

X_train_ab = x_train
X_train_ab = X_train_ab.drop('euclidean_distance', axis=1)
X_train_ab = X_train_ab.drop('jaccard_distance', axis=1)
X_train_ab = X_train_ab.drop('q2_freq', axis=1)
X_train_ab = X_train_ab.drop('q1_freq', axis=1)
X_train_ab = X_train_ab.drop('q1_q2_intersect', axis=1)
X_train_ab = X_train_ab.drop("word_match", axis=1)

print("load train")
df_train = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/train_spacy_cleaned.csv')
df_train = df_train.fillna(' ')

print("load test")
df_test = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/test_spacy_cleaned.csv')
ques = pd.concat([df_train[['question1', 'question2']], \
    df_test[['question1', 'question2']]], axis=0).reset_index(drop='index')

print("qdict")
q_dict = defaultdict(set)
for i in range(ques.shape[0]):
        q_dict[ques.question1[i]].add(ques.question2[i])
        q_dict[ques.question2[i]].add(ques.question1[i])

def q1_freq(row):
    return(len(q_dict[row['question1']]))
    
def q2_freq(row):
    return(len(q_dict[row['question2']]))
    
def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

print("q1_q2_intersect")
df_train['q1_q2_intersect'] = df_train.apply(q1_q2_intersect, axis=1, raw=True)
print("q1_freq")
df_train['q1_freq'] = df_train.apply(q1_freq, axis=1, raw=True)
print("q2_freq")
df_train['q2_freq'] = df_train.apply(q2_freq, axis=1, raw=True)

print("q1_q2_intersect")
df_test['q1_q2_intersect'] = df_test.apply(q1_q2_intersect, axis=1, raw=True)
print("q1_freq")
df_test['q1_freq'] = df_test.apply(q1_freq, axis=1, raw=True)
print("q2_freq")
df_test['q2_freq'] = df_test.apply(q2_freq, axis=1, raw=True)

test_leaky = df_test.loc[:, ['q1_q2_intersect','q1_freq','q2_freq']]
del df_test

train_leaky = df_train.loc[:, ['q1_q2_intersect','q1_freq','q2_freq']]

# explore
stops = set(stopwords.words("english"))

print("lower split 1")
df_train['question1'] = df_train['question1'].map(lambda x: str(x).lower().split())
print("lower split 2")
df_train['question2'] = df_train['question2'].map(lambda x: str(x).lower().split())

print("train_qs")
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist())

words = [x for y in train_qs for x in y]
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

print('Building Features')
X_train_full = build_features(df_train, stops, weights)
X_train_full = pd.concat((X_train_full, X_train_ab, train_leaky), axis=1)
y_train = df_train['is_duplicate'].values

X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train, test_size=0.1, random_state=4242)

#UPDownSampling
print("updownsampling")
pos_train = X_train[y_train == 1]
neg_train = X_train[y_train == 0]
X_train = pd.concat((neg_train, pos_train.iloc[:int(0.8*len(pos_train))], neg_train))
y_train = np.array([0] * neg_train.shape[0] + [1] * pos_train.iloc[:int(0.8*len(pos_train))].shape[0] + [0] * neg_train.shape[0])
print(np.mean(y_train))
del pos_train, neg_train

pos_valid = X_valid[y_valid == 1]
neg_valid = X_valid[y_valid == 0]
X_valid = pd.concat((neg_valid, pos_valid.iloc[:int(0.8 * len(pos_valid))], neg_valid))
y_valid = np.array([0] * neg_valid.shape[0] + [1] * pos_valid.iloc[:int(0.8 * len(pos_valid))].shape[0] + [0] * neg_valid.shape[0])
print(np.mean(y_valid))
del pos_valid, neg_valid


params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 7
params['subsample'] = 0.6
params['base_score'] = 0.2
#params['scale_pos_weight'] = 0.36

print(list(X_train))


d_train = xgb.DMatrix(X_train, label=y_train)
d_valid = xgb.DMatrix(X_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 2500, watchlist, early_stopping_rounds=50, verbose_eval=10)
print(log_loss(y_valid, bst.predict(d_valid)))
bst.save_model(args.save + '.mdl')


print('Building Test Features')
x_test_real = x_test_real.drop('euclidean_distance', axis=1)
x_test_real = x_test_real.drop('jaccard_distance', axis=1)
x_test_real = x_test_real.drop('q2_freq', axis=1)
x_test_real = x_test_real.drop('q1_freq', axis=1)
x_test_real = x_test_real.drop('q1_q2_intersect', axis=1)
x_test_real = x_test_real.drop("word_match", axis=1)

df_test = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/test_spacy_cleaned.csv')
df_test = df_test.fillna(' ')

df_test['question1'] = df_test['question1'].map(lambda x: str(x).lower().split())
df_test['question2'] = df_test['question2'].map(lambda x: str(x).lower().split())

x_test_build = build_features(df_test, stops, weights)
x_test = pd.concat((x_test_build, x_test_real, test_leaky), axis=1)
d_test = xgb.DMatrix(x_test)
p_test = bst.predict(d_test)
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = p_test
sub.to_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/handcraftKernel0158_spacy_cleaned.csv', index=False)


#saving variables
X_train_full.to_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/train_4_spacy_cleaned.csv') 
x_test.to_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/test_4_spacy_cleaned.csv')