
laptop = 0
if laptop == 1:
    drive = "C"
else:
    drive = "F"

preprocessing = "_MyMagic3_spacyLemma"

import numpy as np
import pandas as pd
import xgboost as xgb
import operator
from sklearn.model_selection import train_test_split
from collections import Counter
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from pylab import plot, show, subplot, specgram, imshow, savefig
from sklearn.model_selection import KFold
K = 10
kf = KFold(n_splits = K)
np.set_printoptions(threshold=400000)
pd.set_option('display.max_rows', 2000, 'display.max_columns', 2000,  'display.show_dimensions', 'truncate')

preprocessing = "_spacy_cleaned"


input_folder = 'F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/'
df_train = pd.read_csv(input_folder + 'train.csv')
df_test  = pd.read_csv(input_folder + 'test.csv')
y = df_train.is_duplicate.values



#xg result
print("loading xg")
xgTraining = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/xgb_gbtree_train_MyMagic3_kf.csv', header=0) 
xgTest = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/xgb_gbtree_test_MyMagic3_kf.csv', header=0) #xg result
#ANN result
print("loading ANN")
ANNTraining = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/ANN_train_MyMagic3_kf.csv', header=0) 
ANNTest = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/ANN_test_MyMagic3_kf.csv', header=0) #ANN result


#gru result
print("loading gru")
rnnGRUTraining = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/nn 1 gru/0.3143_lstm_300_200_0.50_0.50_gru_Train.csv', header=0) 
rnnGRUTest = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/nn 1 gru/0.3143_lstm_300_200_0.50_0.50_gru.csv', header=0) 
#lstm sin x kf3 result
print("loading lstm kf3 sin x")
rnnLSTMTraining = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/newLystdo_train_kf3_spacy_cleaned_kf3_sinX_epochsPatience3.csv', header=0) 
rnnLSTMTest = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/newLystdo_test_kf3_spacy_cleaned_kf3_sinX_epochsPatience3.csv', header=0) 
#xg result
print("loading xg_spacy")
xgTraining_spacy = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/xgb_gbtree_train' + preprocessing  + '.csv', header=0) 
xgTest_spacy = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/xgb_gbtree_test' + preprocessing  + '.csv', header=0) #xg result
#ANN result
print("loading ANN_spacy")
ANNTraining_spacy = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/ANN_train' + preprocessing  + '.csv', header=0) 
ANNTest_spacy = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/ANN_test' + preprocessing  + '.csv', header=0) #ANN result
print("loading lightGBM_spacy")#LB  0.20513
gbmTraining_spacy = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/gbm_train' + preprocessing  + '.csv', header=0) 
gbmTest_spacy = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/gbm_test' + preprocessing  + '.csv', header=0) #ANN result

preprocessing = "spacy_cleaned_7_"



x = pd.DataFrame()
x['xg'] = xgTraining.is_duplicate
x['ANN'] = ANNTraining.is_duplicate
x['gru'] = rnnGRUTraining.is_duplicate
x['xg_spacy'] = xgTraining_spacy.is_duplicate
x['ANN_spacy'] = ANNTraining_spacy.is_duplicate
x['gbm_spacy'] = gbmTraining_spacy.is_duplicate
x['LSTM'] = rnnLSTMTraining.is_duplicate



x_test = pd.DataFrame()
x_test['xg'] = xgTest.is_duplicate
x_test['ANN'] = ANNTest.is_duplicate
x_test['gru'] = rnnGRUTest.is_duplicate
x_test['xg_spacy'] = xgTest_spacy.is_duplicate
x_test['ANN_spacy'] = ANNTest_spacy.is_duplicate
x_test['gbm_spacy'] = gbmTest_spacy.is_duplicate 
x_test['LSTM'] = rnnLSTMTest.is_duplicate 



gru = x['gru'].values
xg = x['xg'].values
ANN = x['ANN'].values
gru = x['gru'].values
xg_spacy = x['xg_spacy'].values
ANN_spacy = x['ANN_spacy'].values
gbm_spacy = x['gbm_spacy'].values
LSTM = x['LSTM'].values




corrcoef = np.corrcoef(ANN,[LSTM, gru, xg, gbm_spacy, ANN_spacy, xg_spacy])



print(x.columns)
print(x.describe())

feature_names = list(x.columns.values)
print("Features: {}".format(feature_names))

ROUNDS = 570
RS = 12357
params = {}
params['scale_pos_weight'] = 0.36 #https://www.kaggle.com/c/quora-question-pairs/discussion/31179   same LB as if = 1 above
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.01
params['max_depth'] = 5#4
params['silent'] = 1
params['seed'] = RS
#params['booster'] = 'dart'


#training
def create_feature_map(features):
	outfile = open('xgb.fmap', 'w')
	i = 0
	for feat in features:
		outfile.write('{0}\t{1}\tq\n'.format(i, feat))
		i = i + 1
	outfile.close()
create_feature_map(feature_names)

###KFold
stacking_train = np.empty(shape=(len(x)))
stacking_test = np.empty(shape=(len(x_test),K))
for kth, (train_index, test_index) in enumerate(kf.split(x)):
    print("\n********************** FOLD ", kth+1, " ******************\n")
    max_idx_test  = np.amax(test_index)+1
    min_idx_test  = np.amin(test_index)
    max_idx_train = np.amax(train_index)+1
    min_idx_train = np.amin(train_index)
    X_training, X_val, labels_training, labels_val = train_test_split(x[min_idx_train:max_idx_train], y[min_idx_train:max_idx_train], test_size=0.2)#, random_state=RS)
    xg_train = xgb.DMatrix(X_training, label=labels_training)
    xg_val = xgb.DMatrix(X_val, label=labels_val)
    watchlist  = [(xg_train,'train'), (xg_val,'eval')]
    clr = xgb.train(params, xg_train, ROUNDS, watchlist, early_stopping_rounds=20, verbose_eval=10)
    #predict training ids of the testing fold
    stacking_train[min_idx_test:max_idx_test] = clr.predict(xgb.DMatrix(x[min_idx_test:max_idx_test]))
    #pred test
    stacking_test[:,kth] = clr.predict(xgb.DMatrix(x_test))
del max_idx_test, min_idx_test, max_idx_train, min_idx_train, X_training, X_val, labels_training, labels_val
    

#Saving 
#pred average
preds = (stacking_test[:,0] + stacking_test[:,1] + stacking_test[:,2] + stacking_test[:,3] + stacking_test[:,4] + stacking_test[:,5] + stacking_test[:,6] + stacking_test[:,7] + stacking_test[:,8] + stacking_test[:,9])/10


print("Features importances...")
importance = clr.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))
ft = pd.DataFrame(importance, columns=['feature', 'fscore'])
ft.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
plt.gcf().savefig("F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer2/xg_gbm_vieuxGru_ann_byXG_kf5_7.png")
print("Writing output...")
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = preds #(*0.85 best)
sub.to_csv("F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer2/xg_gbm_vieuxGru_ann_byXG_kf5_7.csv".format(RS, ROUNDS), index=False)


print("Done.")

