# -*- coding: utf-8 -*-
#https://www.kaggle.com/dasolmar/xgb-with-whq-jaccard/code/code

import numpy as np
import pandas as pd
import xgboost as xgb
import datetime
import operator
from sklearn.model_selection import train_test_split
from collections import Counter
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from pylab import plot, show, subplot, specgram, imshow, savefig

RS = 12357
ROUNDS = 315

print("Started")
np.random.seed(RS)
input_folder = 'F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/'

#https://www.kaggle.com/c/quora-question-pairs/discussion/32819
#adding magic features
train_combine = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/train_comb.csv', header=0) 
test_combine = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/test_comb.csv', header=0) 
df_combine = pd.concat([train_combine, test_combine]) 

#https://www.kaggle.com/c/quora-question-pairs/discussion/31284
#https://www.kaggle.com/c/quora-question-pairs/discussion/30224
#adding Abhishek features
train_abhis = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/train_featuresAbhis.csv', encoding = "ISO-8859-1", header=0) 
test_abhis = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/test_featuresAbhis.csv', encoding = "ISO-8859-1", header=0) 
df_abhis = pd.concat([train_abhis, test_abhis]) 

#==============================================================================
# #rnn 1 lstm
# train_rnn1Lstm = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/nn 1 lstm best results/0.2647_lstm_300_200_0.30_0.30Train.csv', header=0) 
# test_rnn1Lstm = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/nn 1 lstm best results/0.2647_lstm_300_200_0.30_0.30.csv', header=0) 
# df_rnn1Lstm = pd.concat([train_rnn1Lstm, test_rnn1Lstm]) 
#==============================================================================

#==============================================================================
# #rnn 1 GRU
# train_rnn1GRU = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/nn 1 gru/0.3143_lstm_300_200_0.50_0.50_gru_Train.csv', header=0) 
# test_rnn1GRU = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/nn 1 gru/0.3143_lstm_300_200_0.50_0.50_gru.csv', header=0) 
# df_rnn1GRU = pd.concat([train_rnn1GRU, test_rnn1GRU]) 
#==============================================================================

print("all imported, starting to process")




def train_xgb(X, y, params):
	print("Will train XGB for {} rounds, RandomSeed: {}".format(ROUNDS, RS))
	x, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RS)

	xg_train = xgb.DMatrix(x, label=y_train)
	xg_val = xgb.DMatrix(X_val, label=y_val)

	watchlist  = [(xg_train,'train'), (xg_val,'eval')]
	return xgb.train(params, xg_train, ROUNDS, watchlist)

def predict_xgb(clr, X_test):
	return clr.predict(xgb.DMatrix(X_test))

def create_feature_map(features):
	outfile = open('xgb.fmap', 'w')
	i = 0
	for feat in features:
		outfile.write('{0}\t{1}\tq\n'.format(i, feat))
		i = i + 1
	outfile.close()

def add_word_count(x, df, word):
	x['q1_' + word] = df['question1'].apply(lambda x: (word in str(x).lower())*1)
	x['q2_' + word] = df['question2'].apply(lambda x: (word in str(x).lower())*1)
	x[word + '_both'] = x['q1_' + word] * x['q2_' + word]


params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.11
params['max_depth'] = 5
params['silent'] = 1
params['seed'] = RS

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

x['word_match']       = df['word_shares'].apply(lambda x: float(x.split(':')[0]))
x['word_match_2root'] = np.sqrt(x['word_match'])
x['tfidf_word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[1]))
x['shared_count']     = df['word_shares'].apply(lambda x: float(x.split(':')[2]))

x['stops1_ratio']     = df['word_shares'].apply(lambda x: float(x.split(':')[3]))
x['stops2_ratio']     = df['word_shares'].apply(lambda x: float(x.split(':')[4]))
x['shared_2gram']     = df['word_shares'].apply(lambda x: float(x.split(':')[5]))
x['cosine']           = df['word_shares'].apply(lambda x: float(x.split(':')[6]))
x['words_hamming']    = df['word_shares'].apply(lambda x: float(x.split(':')[7]))
x['diff_stops_r']     = x['stops1_ratio'] - x['stops2_ratio']

x['len_q1'] = df['question1'].apply(lambda x: len(str(x)))
x['len_q2'] = df['question2'].apply(lambda x: len(str(x)))
x['diff_len'] = x['len_q1'] - x['len_q2']

x['caps_count_q1'] = df['question1'].apply(lambda x:sum(1 for i in str(x) if i.isupper()))
x['caps_count_q2'] = df['question2'].apply(lambda x:sum(1 for i in str(x) if i.isupper()))
x['diff_caps'] = x['caps_count_q1'] - x['caps_count_q2']

x['len_char_q1'] = df['question1'].apply(lambda x: len(str(x).replace(' ', '')))
x['len_char_q2'] = df['question2'].apply(lambda x: len(str(x).replace(' ', '')))
x['diff_len_char'] = x['len_char_q1'] - x['len_char_q2']

x['len_word_q1'] = df['question1'].apply(lambda x: len(str(x).split()))
x['len_word_q2'] = df['question2'].apply(lambda x: len(str(x).split()))
x['diff_len_word'] = x['len_word_q1'] - x['len_word_q2']

x['avg_world_len1'] = x['len_char_q1'] / x['len_word_q1']
x['avg_world_len2'] = x['len_char_q2'] / x['len_word_q2']
x['diff_avg_word'] = x['avg_world_len1'] - x['avg_world_len2']

x['exactly_same'] = (df['question1'] == df['question2']).astype(int)
x['duplicated'] = df.duplicated(['question1','question2']).astype(int)

#https://www.kaggle.com/c/quora-question-pairs/discussion/32819
#adding magic features
print("Magic features")
x['q1_freq'] = df_combine['q1_freq'] 
x['q2_freq'] = df_combine['q2_freq']

#https://www.kaggle.com/c/quora-question-pairs/discussion/31284
#https://www.kaggle.com/c/quora-question-pairs/discussion/30224
#adding Abhishek features
print("Abhis features")
x['common_words'] = df_abhis['common_words'] 
x['fuzz_qratio'] = df_abhis['fuzz_qratio'] 
x['fuzz_WRatio'] = df_abhis['fuzz_WRatio'] 
x['fuzz_partial_ratio'] = df_abhis['fuzz_partial_ratio'] 
x['fuzz_partial_token_set_ratio'] = df_abhis['fuzz_partial_token_set_ratio'] 
x['fuzz_partial_token_sort_ratio'] = df_abhis['fuzz_partial_token_sort_ratio'] 
x['fuzz_token_set_ratio'] = df_abhis['fuzz_token_set_ratio'] 
x['fuzz_token_sort_ratio'] = df_abhis['fuzz_token_sort_ratio'] 
x['wmd'] = df_abhis['wmd'] 
x['norm_wmd'] = df_abhis['norm_wmd'] 
x['cosine_distance'] = df_abhis['cosine_distance'] 
x['cityblock_distance'] = df_abhis['cityblock_distance'] 
x['jaccard_distance'] = df_abhis['jaccard_distance'] 
x['canberra_distance'] = df_abhis['canberra_distance'] 
x['euclidean_distance'] = df_abhis['euclidean_distance'] 
x['minkowski_distance'] = df_abhis['minkowski_distance'] 
x['braycurtis_distance'] = df_abhis['braycurtis_distance'] 
x['skew_q1vec'] = df_abhis['skew_q1vec'] 
x['skew_q2vec'] = df_abhis['skew_q2vec'] 
x['kur_q1vec'] = df_abhis['kur_q1vec'] 
x['kur_q2vec'] = df_abhis['kur_q2vec'] 

#==============================================================================
# ##rnn 1 lstm
# print("rnn features")
# x['rnn1Lstm'] = df_rnn1Lstm['is_duplicate'] 
#==============================================================================


#==============================================================================
# ##rnn 1 gru
# print("rnn gru features")
# x['rnn1GRU'] = df_rnn1GRU['is_duplicate'] 
# 
#==============================================================================


add_word_count(x, df,'how')
add_word_count(x, df,'what')
add_word_count(x, df,'which')
add_word_count(x, df,'who')
add_word_count(x, df,'where')
add_word_count(x, df,'when')
add_word_count(x, df,'why')

print(x.columns)
print(x.describe())

feature_names = list(x.columns.values)
create_feature_map(feature_names)
print("Features: {}".format(feature_names))

x_train = x[:df_train.shape[0]]
x_test  = x[df_train.shape[0]:]
y_train = df_train['is_duplicate'].values
del x, df_train

if 1: # Now we oversample the negative class - on your own risk of overfitting!
	pos_train = x_train[y_train == 1]
	neg_train = x_train[y_train == 0]

	print("Oversampling started for proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))
	p = 0.165
	scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
	while scale > 1:
		neg_train = pd.concat([neg_train, neg_train])
		scale -=1
	neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
	print("Oversampling done, new proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))

	x_train = pd.concat([pos_train, neg_train])
	y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
	del pos_train, neg_train

print("Training data: X_train: {}, Y_train: {}, X_test: {}".format(x_train.shape, len(y_train), x_test.shape))
clr = train_xgb(x_train, y_train, params)
preds = predict_xgb(clr, x_test)

print("Writing output...")
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = preds *.75
sub.to_csv("xgb_seed{}_n{}.csv".format(RS, ROUNDS), index=False)

print("Features importances...")
importance = clr.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))
ft = pd.DataFrame(importance, columns=['feature', 'fscore'])

ft.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
plt.gcf().savefig('features_importance.png')

print("Done.")
