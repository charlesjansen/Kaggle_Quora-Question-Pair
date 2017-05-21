# -*- coding: utf-8 -*-
#https://www.kaggle.com/dasolmar/xgb-with-whq-jaccard/code/code
laptop = 0
if laptop == 1:
    drive = "C"
else:
    drive = "F"

import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords


input_folder = drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/'


print("all imported, starting to process")


def add_word_count(x, df, word):
	x['q1_' + word] = df['question1'].apply(lambda x: (word in str(x).lower())*1)
	x['q2_' + word] = df['question2'].apply(lambda x: (word in str(x).lower())*1)
	x[word + '_both'] = x['q1_' + word] * x['q2_' + word]




df_train = pd.read_csv(input_folder + 'train.csv')
df_test  = pd.read_csv(input_folder + 'test.csv')

###
#	df_train['question1'] = df_train['question1'].apply(lambda x:str(x).replace("?",""))
#	df_train['question2'] = df_train['question2'].apply(lambda x:str(x).replace("?",""))
#	df_test['question1'] = df_test['question1'].apply(lambda x:str(x).replace("?",""))
#	df_test['question2'] = df_test['question2'].apply(lambda x:str(x).replace("?",""))
###
###	
#	df_train['question1'] = df_train['question1'].apply(lambda x:str(x).replace(".",""))
#	df_train['question2'] = df_train['question2'].apply(lambda x:str(x).replace(".",""))
#	df_test['question1'] = df_test['question1'].apply(lambda x:str(x).replace(".",""))
#	df_test['question2'] = df_test['question2'].apply(lambda x:str(x).replace(".",""))

#	df_train['question1'] = df_train['question1'].apply(lambda x:str(x).replace(",",""))
#	df_train['question2'] = df_train['question2'].apply(lambda x:str(x).replace(",",""))
#	df_test['question1'] = df_test['question1'].apply(lambda x:str(x).replace(",",""))
#	df_test['question2'] = df_test['question2'].apply(lambda x:str(x).replace(",",""))
###

print("Original data: X_train: {}, X_test: {}".format(df_train.shape, df_test.shape))

print("Features processing, be patient...")

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=2):
	return 0 if count < min_count else 1 / (count + eps)

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

stops = set(stopwords.words("english"))

def word_shares(row):
	q1_list = str(row['question1']).lower().split()
	q1 = set(q1_list)
	q1words = q1.difference(stops)
	if len(q1words) == 0:
		return '0:0:0:0:0:0:0:0'

	q2_list = str(row['question2']).lower().split()
	q2 = set(q2_list)
	q2words = q2.difference(stops)
	if len(q2words) == 0:
		return '0:0:0:0:0:0:0:0'

	words_hamming = sum(1 for i in zip(q1_list, q2_list) if i[0]==i[1])/max(len(q1_list), len(q2_list))

	q1stops = q1.intersection(stops)
	q2stops = q2.intersection(stops)

	q1_2gram = set([i for i in zip(q1_list, q1_list[1:])])
	q2_2gram = set([i for i in zip(q2_list, q2_list[1:])])

	shared_2gram = q1_2gram.intersection(q2_2gram)

	shared_words = q1words.intersection(q2words)
	shared_weights = [weights.get(w, 0) for w in shared_words]
	q1_weights = [weights.get(w, 0) for w in q1words]
	q2_weights = [weights.get(w, 0) for w in q2words]
	total_weights = q1_weights + q2_weights
	
	R1 = np.sum(shared_weights) / np.sum(total_weights) #tfidf share
	R2 = len(shared_words) / (len(q1words) + len(q2words) - len(shared_words)) #count share
	R31 = len(q1stops) / len(q1words) #stops in q1
	R32 = len(q2stops) / len(q2words) #stops in q2
	Rcosine_denominator = (np.sqrt(np.dot(q1_weights,q1_weights))*np.sqrt(np.dot(q2_weights,q2_weights)))
	Rcosine = np.dot(shared_weights, shared_weights)/Rcosine_denominator
	if len(q1_2gram) + len(q2_2gram) == 0:
		R2gram = 0
	else:
		R2gram = len(shared_2gram) / (len(q1_2gram) + len(q2_2gram))
	return '{}:{}:{}:{}:{}:{}:{}:{}'.format(R1, R2, len(shared_words), R31, R32, R2gram, Rcosine, words_hamming)

df = pd.concat([df_train, df_test])
df['word_shares'] = df.apply(word_shares, axis=1, raw=True)

x = pd.DataFrame()
print("stacking original features")
print("x word_match")
x['word_match']       = df['word_shares'].apply(lambda x: float(x.split(':')[0]))
print("x word_match_2root")
x['word_match_2root'] = np.sqrt(x['word_match'])
print("x tfidf_word_match")
x['tfidf_word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[1]))
print("x shared_count")
x['shared_count']     = df['word_shares'].apply(lambda x: float(x.split(':')[2]))

print("x stops1_ratio")
x['stops1_ratio']     = df['word_shares'].apply(lambda x: float(x.split(':')[3]))
print("x stops2_ratio")
x['stops2_ratio']     = df['word_shares'].apply(lambda x: float(x.split(':')[4]))
print("x shared_2gram")
x['shared_2gram']     = df['word_shares'].apply(lambda x: float(x.split(':')[5]))
print("x cosine")
x['cosine']           = df['word_shares'].apply(lambda x: float(x.split(':')[6]))
print("x words_hamming")
x['words_hamming']    = df['word_shares'].apply(lambda x: float(x.split(':')[7]))
print("x diff_stops_r")
x['diff_stops_r']     = x['stops1_ratio'] - x['stops2_ratio']

print("x len_q1")
x['len_q1'] = df['question1'].apply(lambda x: len(str(x)))
print("x len_q2")
x['len_q2'] = df['question2'].apply(lambda x: len(str(x)))
print("x diff_len")
x['diff_len'] = x['len_q1'] - x['len_q2']

print("x caps_count_q1")
x['caps_count_q1'] = df['question1'].apply(lambda x:sum(1 for i in str(x) if i.isupper()))
print("x caps_count_q2")
x['caps_count_q2'] = df['question2'].apply(lambda x:sum(1 for i in str(x) if i.isupper()))
print("x diff_caps")
x['diff_caps'] = x['caps_count_q1'] - x['caps_count_q2']

print("x len_char_q1")
x['len_char_q1'] = df['question1'].apply(lambda x: len(str(x).replace(' ', '')))
print("x len_char_q2")
x['len_char_q2'] = df['question2'].apply(lambda x: len(str(x).replace(' ', '')))
print("x diff_len_char")
x['diff_len_char'] = x['len_char_q1'] - x['len_char_q2']

print("x len_word_q1")
x['len_word_q1'] = df['question1'].apply(lambda x: len(str(x).split()))
print("x len_word_q2")
x['len_word_q2'] = df['question2'].apply(lambda x: len(str(x).split()))
print("x diff_len_word")
x['diff_len_word'] = x['len_word_q1'] - x['len_word_q2']

print("x avg_world_len1")
x['avg_world_len1'] = x['len_char_q1'] / x['len_word_q1']
print("x avg_world_len2")
x['avg_world_len2'] = x['len_char_q2'] / x['len_word_q2']
print("x diff_avg_word")
x['diff_avg_word'] = x['avg_world_len1'] - x['avg_world_len2']

print("x exactly_same")
x['exactly_same'] = (df['question1'] == df['question2']).astype(int)
print("x duplicated")
x['duplicated'] = df.duplicated(['question1','question2']).astype(int)

print("x how")
add_word_count(x, df,'how')
print("x what")
add_word_count(x, df,'what')
print("x which")
add_word_count(x, df,'which')
print("x who")
add_word_count(x, df,'who')
print("x where")
add_word_count(x, df,'where')
print("x when")
add_word_count(x, df,'when')
print("x why")
add_word_count(x, df,'why')



x.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/xgb_whq_jaccard_features.csv", index=False)





















