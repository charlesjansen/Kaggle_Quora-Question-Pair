import numpy as np
import pandas as pd
import xgboost as xgb
import operator
from sklearn.model_selection import train_test_split
from collections import Counter
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from pylab import plot, show, subplot, specgram, imshow, savefig


input_folder = 'F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/'
df_train = pd.read_csv(input_folder + 'train.csv')
df_test  = pd.read_csv(input_folder + 'test.csv')
y = df_train.is_duplicate.values

#rnn result
print("loading gru")
rnnGRUTraining = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/nn 1 gru/0.3143_lstm_300_200_0.50_0.50_gru_Train.csv', header=0) 
rnnGRUTest = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/nn 1 gru/0.3143_lstm_300_200_0.50_0.50_gru.csv', header=0) 
#xg result
print("loading xg")
xgTraining = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/xgb_gbtree_train_extendedMagic2_kf.csv', header=0) 
xgTest = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/xgb_gbtree_test_extendedMagic2_kf.csv', header=0) #xg result

#==============================================================================
# #rm result
# print("loading rm")
# rmTraining = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/RM_train_kf.csv', header=0) 
# rmTest = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/RM_test_kf.csv', header=0) 
# #linReg result
# print("loading linReg")
# linRegTraining = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/linReg_train_kf.csv', header=0) 
# linRegTest = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/linReg_test_kf.csv', header=0) 
# #xg result
# print("loading lgcReg")
# lgcRegTraining = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/lgcReg_train_kf.csv', header=0) 
# lgcRegTest = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/lgcReg_test_kf.csv', header=0) 
#==============================================================================
#ANN result
print("loading ANN")
ANNTraining = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/ANN_train_extendedMagic2_kf.csv', header=0) 
ANNTest = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/ANN_test_extendedMagic2_kf.csv', header=0) #ANN result




x = pd.DataFrame()
x['gru'] = rnnGRUTraining.is_duplicate
x['xg'] = xgTraining.is_duplicate
x['ANN'] = ANNTraining.is_duplicate



x_test = pd.DataFrame()
x_test['gru'] = rnnGRUTest.is_duplicate
x_test['xg'] = xgTest.is_duplicate
x_test['ANN'] = ANNTest.is_duplicate


gru = x['gru'].values
xg = x['xg'].values
ANN = x['ANN'].values


print(np.corrcoef(ANN,[gru,xg]))

print(x.columns)
print(x.describe())

feature_names = list(x.columns.values)
print("Features: {}".format(feature_names))

ROUNDS = 550
RS = 12357
params = {}
params['scale_pos_weight'] = 0.36 #https://www.kaggle.com/c/quora-question-pairs/discussion/31179   same LB as if = 1 above
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.01
params['max_depth'] = 5
params['silent'] = 1
params['seed'] = RS
params['booster'] = 'dart'


#training
def create_feature_map(features):
	outfile = open('xgb.fmap', 'w')
	i = 0
	for feat in features:
		outfile.write('{0}\t{1}\tq\n'.format(i, feat))
		i = i + 1
	outfile.close()
create_feature_map(feature_names)


X_training, X_val, y_training, y_val = train_test_split(x, y, test_size=0.2, random_state=RS)
xg_train = xgb.DMatrix(X_training, label=y_training)
xg_val = xgb.DMatrix(X_val, label=y_val)
watchlist  = [(xg_train,'train'), (xg_val,'eval')]
clr = xgb.train(params, xg_train, ROUNDS, watchlist, early_stopping_rounds=20)


preds = clr.predict(xgb.DMatrix(x_test))
print("Features importances...")
importance = clr.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))
ft = pd.DataFrame(importance, columns=['feature', 'fscore'])
ft.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
plt.gcf().savefig('features_importance.png')
print("Writing output...")
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = preds #(*0.85 best)
sub.to_csv("xgb_gbtree_train_extendedMagic2_gru_xg_ann_dart.csv".format(RS, ROUNDS), index=False)


print("Done.")

