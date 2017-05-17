"""
Detecting duplicate quora questions
feature engineering
@author: Abhishek Thakur
"""
laptop = 0
if laptop == 1:
    drive = "C"
else:
    drive = "F"
    
#import cPickle
import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
nltk.download('stopwords')
#nltk.download('all')
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis, chebyshev, correlation, sqeuclidean
stop_words = stopwords.words('english')

TRAIN_DATA_FILE = drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/train.csv'
TEST_DATA_FILE  = drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/test.csv'

GOOGLE_DIR = drive + ':/DS-main/BigFiles/'
EMBEDDING_FILE  =  GOOGLE_DIR + 'GoogleNews-vectors-negative300.bin'

def wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def norm_wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)


def sent2vec(s):
    words = str(s).lower()#.decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


data = pd.read_csv(TRAIN_DATA_FILE, sep=',', encoding='utf-8')
data = data.drop(['id', 'qid1', 'qid2'], axis=1)

print("len_q1")
data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
print("len_q2")
data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
print("diff_len")
data['diff_len'] = data.len_q1 - data.len_q2
print("len_char_q1")
data['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
print("len_char_q2")
data['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
print("len_word_q1")
data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))
print("len_word_q2")
data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))
print("common_words")
data['common_words'] = data.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
print("fuzz_qratio")
data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
print("fuzz_WRatio")
data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
print("fuzz_partial_ratio")
data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
print("fuzz_partial_token_set_ratio")
data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
print("fuzz_partial_token_sort_ratio")
data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
print("fuzz_token_set_ratio")
data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
print("fuzz_token_sort_ratio")
data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)


print("wmd")
model = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
data['wmd'] = data.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)


print("norm_wmd")
norm_model = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
norm_model.init_sims(replace=True)
data['norm_wmd'] = data.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)

question1_vectors = np.zeros((data.shape[0], 300))
error_count = 0

for i, q in tqdm(enumerate(data.question1.values)):
    question1_vectors[i, :] = sent2vec(q)

question2_vectors  = np.zeros((data.shape[0], 300))
for i, q in tqdm(enumerate(data.question2.values)):
    question2_vectors[i, :] = sent2vec(q)

print("cosine_distance")
data['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

print("cityblock_distance")
data['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

print("jaccard_distance")
data['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

print("canberra_distance")
data['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

print("euclidean_distance")
data['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

print("minkowski_distance")
data['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

print("braycurtis_distance")
data['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]
###moi
print("chebyshev")
data['chebyshev_distance'] = [chebyshev(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

print("correlation")
data['correlation'] = [correlation(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

print("sqeuclidean")
data['sqeuclidean'] = [sqeuclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]     
    
    
#Abhis    
print("skew_q1vec")
data['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
print("skew_q2vec")
data['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
print("kur_q1vec")
data['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
print("kur_q2vec")
data['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]
#moi
print("kur_Pearson_q1vec")
data['kur_Pearson_q1vec'] = [kurtosis(x, fisher=False) for x in np.nan_to_num(question1_vectors)]
print("kur_Pearson_q2vec")
data['kur_Pearson_q2vec'] = [kurtosis(x, fisher=False) for x in np.nan_to_num(question2_vectors)]





#cPickle.dump(question1_vectors, open('data/q1_w2v.pkl', 'wb'), -1)
#cPickle.dump(question2_vectors, open('data/q2_w2v.pkl', 'wb'), -1)

data.to_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/featuresAbhishekkrthakurTrain.csv', index=False, encoding="utf-8")




#######################################################################Test File
print("****************************** TEST *********************************")

dataTest = pd.read_csv(TEST_DATA_FILE, sep=',', encoding='utf-8')


print("len_q1")
dataTest['len_q1'] = dataTest.question1.apply(lambda x: len(str(x)))
print("len_q2")
dataTest['len_q2'] = dataTest.question2.apply(lambda x: len(str(x)))
print("diff_len")
dataTest['diff_len'] = dataTest.len_q1 - dataTest.len_q2
print("len_char_q1")
dataTest['len_char_q1'] = dataTest.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
print("len_char_q2")
dataTest['len_char_q2'] = dataTest.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
print("len_word_q1")
dataTest['len_word_q1'] = dataTest.question1.apply(lambda x: len(str(x).split()))
print("len_word_q2")
dataTest['len_word_q2'] = dataTest.question2.apply(lambda x: len(str(x).split()))
print("common_words")
dataTest['common_words'] = dataTest.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
print("fuzz_qratio")
dataTest['fuzz_qratio'] = dataTest.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
print("fuzz_WRatio")
dataTest['fuzz_WRatio'] = dataTest.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
print("fuzz_partial_ratio")
dataTest['fuzz_partial_ratio'] = dataTest.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
print("fuzz_partial_token_set_ratio")
dataTest['fuzz_partial_token_set_ratio'] = dataTest.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
print("fuzz_partial_token_sort_ratio")
dataTest['fuzz_partial_token_sort_ratio'] = dataTest.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
print("fuzz_token_set_ratio")
dataTest['fuzz_token_set_ratio'] = dataTest.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
print("fuzz_token_sort_ratio")
dataTest['fuzz_token_sort_ratio'] = dataTest.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)


print("wmd")
dataTest['wmd'] = dataTest.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)


print("norm_wmd")
norm_model.init_sims(replace=True)
dataTest['norm_wmd'] = dataTest.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)

question1_vectors = np.zeros((dataTest.shape[0], 300))
error_count = 0

for i, q in tqdm(enumerate(dataTest.question1.values)):
    question1_vectors[i, :] = sent2vec(q)

question2_vectors  = np.zeros((dataTest.shape[0], 300))
for i, q in tqdm(enumerate(dataTest.question2.values)):
    question2_vectors[i, :] = sent2vec(q)

print("cosine_distance")
dataTest['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

print("cityblock_distance")
dataTest['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

print("jaccard_distance")
dataTest['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

print("canberra_distance")
dataTest['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

print("euclidean_distance")
dataTest['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

print("minkowski_distance")
dataTest['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

print("braycurtis_distance")
dataTest['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

print("skew_q1vec")
dataTest['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
print("skew_q2vec")
dataTest['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
print("kur_q1vec")
dataTest['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
print("kur_q2vec")
dataTest['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

#cPickle.dump(question1_vectors, open('dataTest/q1_w2v.pkl', 'wb'), -1)
#cPickle.dump(question2_vectors, open('dataTest/q2_w2v.pkl', 'wb'), -1)


#cPickle.dump(question1_vectors_test, open('dataTest/q1_w2v.pkl', 'wb'), -1)
#cPickle.dump(question2_vectors_test, open('dataTest/q2_w2v.pkl', 'wb'), -1)
dataTest.to_csv(drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/featuresAbhishekkrthakurTest.csv', index=False, encoding="utf-8")
