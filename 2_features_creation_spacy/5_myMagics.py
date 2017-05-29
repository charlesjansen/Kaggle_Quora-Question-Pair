# -*- coding: utf-8 -*-
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
print("rarestWordID1 train")
train_orig['rarestWordID1'] = train_orig.question1.apply(lambda x: rarest_word(x))
print("rarestWordID2 train")
train_orig['rarestWordID2'] = train_orig.question2.apply(lambda x: rarest_word(x))

print("avgWordID1 train")
train_orig['avgWordID1'] = train_orig.question1.apply(lambda x: averaged_sum_words(x))
print("avgWordID2 train")
train_orig['avgWordID2'] = train_orig.question2.apply(lambda x: averaged_sum_words(x))

print("diffAvgWordID train")
train_orig['diffAvgWordID'] = abs(train_orig.avgWordID1 - train_orig.avgWordID2)
print("diffRarestWordID train")
train_orig['diffRarestWordID'] = abs(train_orig.rarestWordID1 - train_orig.rarestWordID2)





print("rarestWordID1 test")
test_orig['rarestWordID1'] = test_orig.question1.apply(lambda x: rarest_word(x))
print("rarestWordID2 test")
test_orig['rarestWordID2'] = test_orig.question2.apply(lambda x: rarest_word(x))

print("avgWordID1 test")
test_orig['avgWordID1'] = test_orig.question1.apply(lambda x: averaged_sum_words(x))
print("avgWordID2 test")
test_orig['avgWordID2'] = test_orig.question2.apply(lambda x: averaged_sum_words(x))

print("diffAvgWordID test")
test_orig['diffAvgWordID'] = abs(test_orig.avgWordID1 - test_orig.avgWordID2)
print("diffRarestWordID test")
test_orig['diffRarestWordID'] = abs(test_orig.rarestWordID1 - test_orig.rarestWordID2)




#==============================================================================
# #https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb
#==============================================================================
from nltk.corpus import stopwords

stops = set(stopwords.words("english"))

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

print("wordMatchShare train")
train_orig['wordMatchShare'] = train_orig.apply(word_match_share, axis=1, raw=True)
print("wordMatchShare test")
test_orig['wordMatchShare'] = test_orig.apply(word_match_share, axis=1, raw=True)






#td-idf
from collections import Counter

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)
train_qs = pd.Series(train_orig['question1'].tolist() + train_orig['question2'].tolist()).astype(str)
test_qs = pd.Series(test_orig['question1'].tolist() + test_orig['question2'].tolist()).astype(str)

eps = 5000 
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}
def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

print("tfidf_word_match train")
train_orig['tfidf_word_match'] = train_orig.apply(tfidf_word_match_share, axis=1, raw=True)
print("tfidf_word_match test")
test_orig['tfidf_word_match'] = test_orig.apply(tfidf_word_match_share, axis=1, raw=True)





#==============================================================================
# https://www.kaggle.com/sudalairajkumar/simple-leaky-exploration-notebook-quora
#==============================================================================
from nltk.corpus import stopwords
from nltk import word_tokenize#, ngrams

eng_stopwords = set(stopwords.words('english'))

def get_unigrams(que):
    return [word for word in word_tokenize(que.lower()) if word not in eng_stopwords]

def get_common_unigrams(row):
    return len( set(row["unigrams_ques1"]).intersection(set(row["unigrams_ques2"])) )

def get_common_unigram_ratio(row):
    return float(row["unigrams_common_count"]) / max(len( set(row["unigrams_ques1"]).union(set(row["unigrams_ques2"])) ),1)

print("unigrams_ques1 train")
train_orig["unigrams_ques1"] = train_orig['question1'].apply(lambda x: get_unigrams(str(x)))
print("unigrams_ques2 train")
train_orig["unigrams_ques2"] = train_orig['question2'].apply(lambda x: get_unigrams(str(x)))
print("unigrams_common_count train")
train_orig["unigrams_common_count"] = train_orig.apply(lambda row: get_common_unigrams(row),axis=1)
print("unigrams_common_ratio train")
train_orig["unigrams_common_ratio"] = train_orig.apply(lambda row: get_common_unigram_ratio(row), axis=1)

print("unigrams_ques1 test")
test_orig["unigrams_ques1"] = test_orig['question1'].apply(lambda x: get_unigrams(str(x)))
print("unigrams_ques2 test")
test_orig["unigrams_ques2"] = test_orig['question2'].apply(lambda x: get_unigrams(str(x)))
print("unigrams_common_count test")
test_orig["unigrams_common_count"] = test_orig.apply(lambda row: get_common_unigrams(row),axis=1)
print("unigrams_common_ratio test")
test_orig["unigrams_common_ratio"] = test_orig.apply(lambda row: get_common_unigram_ratio(row), axis=1)

train_orig = train_orig.drop(['unigrams_ques1', 'unigrams_ques2'], axis=1)
test_orig = test_orig.drop(['unigrams_ques1', 'unigrams_ques2'], axis=1)



#==============================================================================
# https://www.kaggle.com/c/quora-question-pairs/discussion/33371
#==============================================================================
test_orig["qid1"] = 0
test_orig["qid2"] = 0
counter = 0
for index, row in test_orig.iterrows():
    counter += 1
    row["qid1"] = counter
    counter += 1
    row["qid2"] = counter
    if (index%100000 == 0 ):
            print ('test',index)
    

import networkx as nx
df_train = train_orig[["qid1", "qid2"]] 
df_test = test_orig[["qid1", "qid2"]] 
df_all = pd.concat([df_train, df_test])
print("df_all.shape:", df_all.shape) # df_all.shape: (2750086, 2)
df = df_all
g = nx.Graph()
g.add_nodes_from(df.qid1)
edges = list(df[['qid1', 'qid2']].to_records(index=False))
g.add_edges_from(edges)
g.remove_edges_from(g.selfloop_edges())
print(len(set(df.qid1)), g.number_of_nodes()) # 4789604
print(len(df), g.number_of_edges()) # 2743365 (after self-edges)

df_output = pd.DataFrame(data=g.nodes(), columns=["qid"])
print("df_output.shape:", df_output.shape)
NB_CORES = 20
for k in range(2, NB_CORES + 1):
    fieldname = "kcore{}".format(k)
    print("fieldname = ", fieldname)
    ck = nx.k_core(g, k=k).nodes()
    print("len(ck) = ", len(ck))
    df_output[fieldname] = 0
    df_output.ix[df_output.qid.isin(ck), fieldname] = k
df_output.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/question_kcores.csv", index=None)

df_cores = pd.read_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/question_kcores.csv", index_col="qid")
df_cores = df_output
df_cores.index.names = ["qid"]
df_cores['max_kcore'] = df_cores.apply(lambda row: max(row), axis=1)
df_cores[['max_kcore']].to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/question_max_kcores.csv") # with index


cores_dict = pd.read_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/question_max_kcores.csv", index_col="qid").to_dict()["max_kcore"]
def gen_qid1_max_kcore(row):
    return cores_dict[row["qid1"]]
def gen_qid2_max_kcore(row):
    return cores_dict[row["qid2"]]
def gen_max_kcore(row):
    return max(row["qid1_max_kcore"], row["qid2_max_kcore"])

print("qid1_max_kcore train")
train_orig["qid1_max_kcore"] = train_orig.apply(gen_qid1_max_kcore, axis=1)
print("qid2_max_kcore train")
train_orig["qid2_max_kcore"] = train_orig.apply(gen_qid2_max_kcore, axis=1)
print("max_kcore train")
train_orig["max_kcore"] = train_orig.apply(gen_max_kcore, axis=1)

print("qid1_max_kcore test")
test_orig["qid1_max_kcore"] = test_orig.apply(gen_qid1_max_kcore, axis=1)
print("qid2_max_kcore test")
test_orig["qid2_max_kcore"] = test_orig.apply(gen_qid2_max_kcore, axis=1)
print("max_kcore test")
test_orig["max_kcore"] = test_orig.apply(gen_max_kcore, axis=1)



##############################################################################
train_orig = train_orig.drop(['id', 'is_duplicate', 'qid1', 'qid2'], axis=1)
train_orig = train_orig.drop(['question1', 'question2'], axis=1)

test_orig = test_orig.drop(['test_id'], axis=1)
test_orig = test_orig.drop(['question1', 'question2'], axis=1)
#saving
train_orig.to_csv(drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/train_myMagic3_spacyLemma.csv') 
test_orig.to_csv(drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/test_myMagic3_spacyLemma.csv')

