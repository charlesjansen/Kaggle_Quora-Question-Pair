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

#https://www.kaggle.com/c/quora-question-pairs/discussion/32819
#adding magic features
print("loading magic features")
train_combine = pd.read_csv(drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/train_comb.csv', header=0) 
test_combine = pd.read_csv(drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/test_comb.csv', header=0) 
df_combine = pd.concat([train_combine, test_combine]) 

#https://www.kaggle.com/c/quora-question-pairs/discussion/31284
#https://www.kaggle.com/c/quora-question-pairs/discussion/30224
#adding Abhishek features
#train_abhis = pd.read_csv(drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/train_featuresAbhis.csv', encoding = "ISO-8859-1", header=0) 
#test_abhis = pd.read_csv(drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/test_featuresAbhis.csv', encoding = "ISO-8859-1", header=0) 
print("loading my extended abhi whq features")
train_abhis = pd.read_csv(drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/featuresAbhishekkrthakurTrain.csv', header=0) 
test_abhis = pd.read_csv(drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/featuresAbhishekkrthakurTest.csv', header=0) 
df_abhis = pd.concat([train_abhis, test_abhis]) 

#magic2
print("loading magic2")
train_mgc2 = pd.read_csv(drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/train_magic2.csv', header=0) 
test_mgc2  = pd.read_csv(drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/test_magic2.csv', header=0)
df_mgc2 = pd.concat([train_mgc2, test_mgc2]) 


#myMagic3
print("loading myMagic")
train_mgc3 = pd.read_csv(drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/train_myMagic.csv', header=0) 
test_mgc3  = pd.read_csv(drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/test_myMagic.csv', header=0)
df_mgc3 = pd.concat([train_mgc3, test_mgc3])



del train_combine, test_combine, train_abhis, test_abhis, laptop, train_mgc2, test_mgc2, train_mgc3, test_mgc3

print("all imported, starting to process")


def add_word_count(x, df, word):
	x['q1_' + word] = df['question1'].apply(lambda x: (word in str(x).lower())*1)
	x['q2_' + word] = df['question2'].apply(lambda x: (word in str(x).lower())*1)
	x[word + '_both'] = x['q1_' + word] * x['q2_' + word]




df_train = pd.read_csv(input_folder + 'train.csv')
df_test  = pd.read_csv(input_folder + 'test.csv')

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



#x[x.index.duplicated()]
#x.reset_index()
#df_combine.reset_index()


#https://www.kaggle.com/c/quora-question-pairs/discussion/32819
#adding magic features
print("Magic features")
print("x q1_freq")
x['q1_freq'] = df_combine['q1_freq'] 
print("x q2_freq")
x['q2_freq'] = df_combine['q2_freq']


#https://www.kaggle.com/c/quora-question-pairs/discussion/31284
#https://www.kaggle.com/c/quora-question-pairs/discussion/30224
#adding Abhishek features
print("Abhis features")
print("x common_words")
x['common_words'] = df_abhis['common_words'] 
print("x fuzz_qratio")
x['fuzz_qratio'] = df_abhis['fuzz_qratio'] 
print("x fuzz_WRatio")
x['fuzz_WRatio'] = df_abhis['fuzz_WRatio'] 
print("x fuzz_partial_ratio")
x['fuzz_partial_ratio'] = df_abhis['fuzz_partial_ratio'] 
print("x fuzz_partial_token_set_ratio")
x['fuzz_partial_token_set_ratio'] = df_abhis['fuzz_partial_token_set_ratio'] 
print("x fuzz_partial_token_sort_ratio")
x['fuzz_partial_token_sort_ratio'] = df_abhis['fuzz_partial_token_sort_ratio'] 
print("x fuzz_token_set_ratio")
x['fuzz_token_set_ratio'] = df_abhis['fuzz_token_set_ratio'] 
print("x fuzz_token_sort_ratio")
x['fuzz_token_sort_ratio'] = df_abhis['fuzz_token_sort_ratio'] 

print("x wmd")
x['wmd'] = df_abhis['wmd'] 
print("x norm_wmd")
x['norm_wmd'] = df_abhis['norm_wmd'] 

print("x cosine_distance")
x['cosine_distance'] = df_abhis['cosine_distance'] 
print("x cityblock_distance")
x['cityblock_distance'] = df_abhis['cityblock_distance'] 
print("x jaccard_distance")
x['jaccard_distance'] = df_abhis['jaccard_distance'] 
print("x canberra_distance")
x['canberra_distance'] = df_abhis['canberra_distance'] 
print("x euclidean_distance")
x['euclidean_distance'] = df_abhis['euclidean_distance'] 
print("x minkowski_distance")
x['minkowski_distance'] = df_abhis['minkowski_distance'] 
print("x braycurtis_distance")
x['braycurtis_distance'] = df_abhis['braycurtis_distance'] 
#moi
print("x chebyshev_distance")
x['chebyshev_distance'] = df_abhis['chebyshev_distance'] 
print("x correlation")
x['correlation'] = df_abhis['correlation'] 
print("x sqeuclidean")
x['sqeuclidean'] = df_abhis['sqeuclidean'] 
print("x levene")
x['levene'] = df_abhis['levene'] 
print("x bartlett")
x['bartlett'] = df_abhis['bartlett'] 
print("x ranksums")
x['ranksums'] = df_abhis['ranksums'] 
print("x ansari")
x['ansari'] = df_abhis['ansari'] 


print("x skew_q1vec")
x['skew_q1vec'] = df_abhis['skew_q1vec'] 
print("x skew_q2vec")
x['skew_q2vec'] = df_abhis['skew_q2vec'] 
print("x kur_q1vec")
x['kur_q1vec'] = df_abhis['kur_q1vec'] 
print("x kur_q2vec")
x['kur_q2vec'] = df_abhis['kur_q2vec'] 
#moi
print("x kur_Pearson_q1vec")
x['kur_Pearson_q1vec'] = df_abhis['kur_Pearson_q1vec'] 
print("x kur_Pearson_q2vec")
x['kur_Pearson_q2vec'] = df_abhis['kur_Pearson_q2vec'] 
print("x tmean_q1vec")
x['tmean_q1vec'] = df_abhis['tmean_q1vec'] 
print("x tmean_q2vec")
x['tmean_q2vec'] = df_abhis['tmean_q2vec'] 



#==============================================================================
# #adding magic2
#==============================================================================
print("x q1_q2_intersect")
x['q1_q2_intersect'] = df_mgc2['q1_q2_intersect']


#==============================================================================
# #adding my magic
#==============================================================================
print("rarestWordID1")
x['rarestWordID1'] = df_mgc3['rarestWordID1']
print("rarestWordID2")
x['rarestWordID2'] = df_mgc3['rarestWordID2']
print("avgWordID1")
x['avgWordID1'] = df_mgc3['avgWordID1']
print("avgWordID2")
x['avgWordID2'] = df_mgc3['avgWordID2']
print("diffAvgWordID")
x['diffAvgWordID'] = df_mgc3['diffAvgWordID']
print("diffRarestWordID")
x['diffRarestWordID'] = df_mgc3['diffRarestWordID']






#low removed (bad idea)
#x = x.drop(['where_both', 'sqeuclidean', 'euclidean_distance', 'when_both', "q2_where", "which_both", "q1_when", "q2_when", "who_both", "q1_who", "q1_where", "q2_who", "q2_which", "q1_which", "why_both", "how_both", "q2_why", "what_both", "q1_why", "q2_what"], axis=1)

#x.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/x_final_features_extendedMagic2.csv", index=False)
x.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/x_final_features_MyMagic3.csv", index=False)



















