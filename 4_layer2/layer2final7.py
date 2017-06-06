
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



#gru result
print("loading gru")
rnnGRUTraining = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/nn 1 gru/0.3143_lstm_300_200_0.50_0.50_gru_Train.csv', header=0) 
rnnGRUTest = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/nn 1 gru/0.3143_lstm_300_200_0.50_0.50_gru.csv', header=0) 


#xg result
print("loading xg_spacy")
xgTraining_spacy = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/xgb_gbtree_train_spacy_cleaned_kf10_1000normal.csv', header=0) 
xgTest_spacy = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/xgb_gbtree_test_spacy_cleaned_kf10_1000normal.csv', header=0) #xg result
print("loading lightGBM_spacy")#LB  0.20513
gbmTraining_spacy = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/lightgbm_train_spacy_cleaned_kf10_kf10.csv', header=0) 
gbmTest_spacy = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/lightgbm_test_spacy_cleaned_kf10_kf10.csv', header=0) #ANN result

#ANN result
print("ANN_spacy50")
ANNTraining_spacy = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/ANN_train_spacy_cleaned_kf10_neurones_50_kf10.csv', header=0) 
ANNTest_spacy = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/ANN_test_spacy_cleaned_kf10_neurones_50_kf10.csv', header=0) #ANN result
#ANN result
print("ANN_spacy100")
ANNTraining_spacy100 = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/ANN_train_spacy_cleaned_kf10_neurones_100_kf10.csv', header=0) 
ANNTest_spacy100 = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/ANN_test_spacy_cleaned_kf10_neurones_100_kf10.csv', header=0) #ANN result
#ANN result
print("ANN_spacy200")
ANNTraining_spacy200 = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/ANN_train_spacy_cleaned_kf10_neurones_200_kf10.csv', header=0) 
ANNTest_spacy200 = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/ANN_test_spacy_cleaned_kf10_neurones_200_kf10.csv', header=0) #ANN result
#ANN result
print("ANN_spacy400")
ANNTraining_spacy400 = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/ANN_train_spacy_cleaned_kf10_neurones_400_kf10.csv', header=0) 
ANNTest_spacy400 = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/ANN_test_spacy_cleaned_kf10_neurones_400_kf10.csv', header=0) #ANN result
#ANN result
print("ANN_spacy500")
ANNTraining_spacy500 = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/ANN_train1layer_spacy_cleaned_kf10_neurones_500_kf10.csv', header=0) 
ANNTest_spacy500 = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/ANN_test1layer_spacy_cleaned_kf10_neurones_500_kf10.csv', header=0) #ANN result

#newLystdoLSTM result
print("LSTM")
LSTMTraining = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/newLystdo_train_kf2__spacy_cleaned_kf10_epochsPatience3.csv', header=0) 
LSTMTest = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/newLystdo_test_kf2__spacy_cleaned_kf10_epochsPatience3.csv', header=0) #ANN result




preprocessing = "spacy_cleaned_j-1_"



x = pd.DataFrame()
x['xg'] = xgTraining_spacy.is_duplicate
x['gru'] = rnnGRUTraining.is_duplicate
x['gbm_spacy'] = gbmTraining_spacy.is_duplicate
x['ANN_spacy50'] = ANNTraining_spacy.is_duplicate
x['ANN_spacy100'] = ANNTraining_spacy100.is_duplicate
x['ANN_spacy200'] = ANNTraining_spacy200.is_duplicate
x['ANN_spacy400'] = ANNTraining_spacy400.is_duplicate
x['ANN_spacy500'] = ANNTraining_spacy500.is_duplicate
x['LSTM'] = LSTMTraining.is_duplicate



x_test = pd.DataFrame()
x_test['xg'] = xgTest_spacy.is_duplicate
x_test['gru'] = rnnGRUTest.is_duplicate
x_test['gbm_spacy'] = gbmTest_spacy.is_duplicate 
x_test['ANN_spacy50'] = ANNTraining_spacy.is_duplicate
x_test['ANN_spacy100'] = ANNTest_spacy100.is_duplicate
x_test['ANN_spacy200'] = ANNTest_spacy200.is_duplicate
x_test['ANN_spacy400'] = ANNTest_spacy400.is_duplicate
x_test['ANN_spacy500'] = ANNTest_spacy500.is_duplicate
x_test['LSTM'] = LSTMTest.is_duplicate


print(len(x['xg']))

#==============================================================================
# xg = x['xg'].values
# gru = x['gru'].values
# gbm_spacy = x['gbm_spacy'].values
# ANN50 = x['ANN_spacy50'].values
# ANN100 = x['ANN_spacy100'].values
# ANN200 = x['ANN_spacy200'].values
# ANN400 = x['ANN_spacy400'].values
# ANN500 = x['ANN_spacy500'].values
# LSTM = x['LSTM'].values
# 
# 
# corrcoef = np.corrcoef(xg,[gru, gbm_spacy, ANN50, ANN100, ANN200, ANN400, ANN500, LSTM])
#==============================================================================








print(x.columns)
print(x.describe())

feature_names = list(x.columns.values)
print("Features: {}".format(feature_names))

ROUNDS = 570
RS = 12357
params = {}
#•params['scale_pos_weight'] = 0.36 #https://www.kaggle.com/c/quora-question-pairs/discussion/31179   same LB as if = 1 above
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.01
params['max_depth'] = 5#4
params['silent'] = 1
params['seed'] = RS


####################################################################
# preprocessing Imputer
####################################################################
print("imputer")
from sklearn.preprocessing import Imputer
x = Imputer().fit_transform(x)
x_test = Imputer().fit_transform(x_test)



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
    X_training, X_val, labels_training, labels_val = train_test_split(x[train_index], y[train_index], test_size=0.2)
    xg_train = xgb.DMatrix(X_training, label=labels_training)
    xg_val = xgb.DMatrix(X_val, label=labels_val)
    watchlist  = [(xg_train,'train'), (xg_val,'eval')]
    clr = xgb.train(params, xg_train, ROUNDS, watchlist, early_stopping_rounds=20, verbose_eval=10)
    #predict training ids of the testing fold
    stacking_train[test_index] = clr.predict(xgb.DMatrix(x[test_index]))
    #pred test
    stacking_test[:,kth] = clr.predict(xgb.DMatrix(x_test))
    del test_index, train_index
    

#Saving 
#pred average
preds = (stacking_test[:,0] + stacking_test[:,1] + stacking_test[:,2] + stacking_test[:,3] + stacking_test[:,4] + stacking_test[:,5] + stacking_test[:,6] + stacking_test[:,7] + stacking_test[:,8] + stacking_test[:,9])/10


print("Features importances...")
importance = clr.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))
ft = pd.DataFrame(importance, columns=['feature', 'fscore'])
ft.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
plt.gcf().savefig("F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer2/xg_gbm_vieuxGru_ann_byXG_kf5_j-1sansweight.png")

print("Writing output...")
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = preds #(*0.85 best)
sub.to_csv("F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer2/xg_gbm_vieuxGru_ann_byXG_kf5_j-1sansweight.csv".format(RS, ROUNDS), index=False)


print("Done.")

