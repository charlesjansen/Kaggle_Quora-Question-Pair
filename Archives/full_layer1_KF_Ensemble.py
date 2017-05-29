# -*- coding: utf-8 -*-
'''
 Ensemble learning creation: http://www.kdnuggets.com/2016/02/ensemble-methods-techniques-produce-improved-machine-learning.html  //http://www.kdnuggets.com/2015/06/ensembles-kaggle-data-science-competition-p3.html
 base_algorithms = [logistic_regression, decision_tree_classification, ...] #for classification
 
 stacking_train_dataset = matrix(row_length=len(target), column_length=len(algorithms))
 stacking_test_dataset = matrix(row_length=len(test), column_length=len(algorithms))
 
 for i,base_algorithm in enumerate(base_algorithms):
     for trainix, testix in split(train, k=10): #you may use sklearn.cross_validation.KFold of sklearn library
         stacking_train_dataset[testix,i] = base_algorithm.fit(train[trainix], target[trainix]).predict(train[testix])
     stacking_test_dataset[,i] = base_algorithm.fit(train).predict(test)
 
 
 final_predictions = combiner_algorithm.fit(stacking_train_dataset, target).predict(stacking_test_dataset)
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
-------------------------------------------------------------------------------------------------------------------------
 Menu:
    # XGBoost
    # preprocessing Imputer
    # Linear regression
    # preprocessing StandardScaler
    # logistic regression
    # KNNs
    # SVM
    # naive bayes
    # random forest
    # ANN
'''
laptop = 0
if laptop == 1:
    drive = "C"
else:
    drive = "F"

preprocessing = "_MyMagic3_spacyLemma_trigram"


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
#x = x.drop(["rarestWordID1", "rarestWordID2"], axis=1)

print(x.columns)
print(x.describe())

feature_names = list(x.columns.values)
print("Features: {}".format(feature_names))

x_train = x[:df_train.shape[0]].astype("float64")
x_test_real  = x[df_train.shape[0]:].astype("float64")
labels = df_train['is_duplicate'].values
x_train_real =  x_train
labels_real =  labels
del x, df_train



####################################################################
# LightGBM
####################################################################
import lightgbm as lgb

params2 = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'binary_logloss'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1
}


###KFold
stacking_train = np.empty(shape=(len(labels_real)))
stacking_test = np.empty(shape=(len(x_test_real),K))
for kth, (train_index, test_index) in enumerate(kf.split(x_train_real)):
    print("\n********************** FOLD ", kth+1, " ******************\n")
    max_idx_test  = np.amax(test_index)+1
    min_idx_test  = np.amin(test_index)
    max_idx_train = np.amax(train_index)+1
    min_idx_train = np.amin(train_index)
    X_training, X_val, labels_training, labels_val = train_test_split(x_train[min_idx_train:max_idx_train], labels[min_idx_train:max_idx_train], test_size=0.2)#, random_state=RS)
    lgb_train = lgb.Dataset(X_training, labels_training)
    lgb_eval = lgb.Dataset(X_val, labels_val, reference=lgb_train)
    gbm = lgb.train(params2,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=20)
    print('Save model...')
    # save model to file
    #gbm.save_model('gbm_model.txt')
    #predict training ids of the testing fold
    print('Start predicting...')
    # predict
    stacking_train[min_idx_test:max_idx_test] = gbm.predict(x_train[min_idx_test:max_idx_test], num_iteration=gbm.best_iteration)
    #pred test
    stacking_test[:,kth] = gbm.predict(x_test_real, num_iteration=gbm.best_iteration)
del max_idx_test, min_idx_test, max_idx_train, min_idx_train, X_training, X_val, labels_training, labels_val
    

#pred average
preds = (stacking_test[:,0] + stacking_test[:,1] + stacking_test[:,2] + stacking_test[:,3] + stacking_test[:,4])/5
#Saving 
print("Writing output...")
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = preds *.75
sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/gbm_test" + preprocessing  + ".csv", index=False)

#--- pred training for ensemble
print("Writing training pred output...")
sub = pd.DataFrame()
sub['is_duplicate'] = stacking_train *.75
sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/gbm_train" + preprocessing  + ".csv", index=False)

del stacking_test, stacking_train, gbm







####################################################################
# XGBoost GBTREE
####################################################################
import xgboost as xgb
#==============================================================================
# if 0: # Now we oversample the negative class - on your own risk of overfitting!
# 	pos_train = x_train[labels == 1]
# 	neg_train = x_train[labels == 0]
# 
# 	print("Oversampling started for proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))
# 	p = 0.165
# 	scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
# 	while scale > 1:
# 		neg_train = pd.concat([neg_train, neg_train])
# 		scale -=1
# 	neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
# 	print("Oversampling done, new proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))
# 
# 	x_train = pd.concat([pos_train, neg_train])
# 	labels = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
# 	del pos_train, neg_train
#==============================================================================
print("Starting XGB GBTREE")
ratio = float(np.sum(labels == 0)) / np.sum(labels==1)
params = {}
params['scale_pos_weight'] = 0.36 #https://www.kaggle.com/c/quora-question-pairs/discussion/31179   same LB as if = 1 above
params['objective'] = 'binary:logistic'
#params['seed'] = RS
params['eval_metric'] = 'logloss'
params['eta'] = 0.11
params['max_depth'] = 5
params['silent'] = 1 
#params['updater'] = 'grow_gpu'
print("Training data: X_train: {}, labels: {}, X_test: {}".format(x_train.shape, len(labels), x_test_real.shape))

ROUNDS = 1000

#training
def create_feature_map(features):
	outfile = open('xgb.fmap', 'w')
	i = 0
	for feat in features:
		outfile.write('{0}\t{1}\tq\n'.format(i, feat))
		i = i + 1
	outfile.close()
create_feature_map(feature_names)

print("Will train XGB for {} rounds, RandomSeed: {}".format(ROUNDS, RS))

###KFold
stacking_train = np.empty(shape=(len(labels_real)))
stacking_test = np.empty(shape=(len(x_test_real),K))
for kth, (train_index, test_index) in enumerate(kf.split(x_train_real)):
    print("\n********************** FOLD ", kth+1, " ******************\n")
    max_idx_test  = np.amax(test_index)+1
    min_idx_test  = np.amin(test_index)
    max_idx_train = np.amax(train_index)+1
    min_idx_train = np.amin(train_index)
    X_training, X_val, labels_training, labels_val = train_test_split(x_train[min_idx_train:max_idx_train], labels[min_idx_train:max_idx_train], test_size=0.2)#, random_state=RS)
    xg_train = xgb.DMatrix(X_training, label=labels_training)
    xg_val = xgb.DMatrix(X_val, label=labels_val)
    watchlist  = [(xg_train,'train'), (xg_val,'eval')]
    clr = xgb.train(params, xg_train, ROUNDS, watchlist, early_stopping_rounds=20, verbose_eval=10)
    #predict training ids of the testing fold
    stacking_train[min_idx_test:max_idx_test] = clr.predict(xgb.DMatrix(x_train[min_idx_test:max_idx_test]))
    #pred test
    stacking_test[:,kth] = clr.predict(xgb.DMatrix(x_test_real))
del max_idx_test, min_idx_test, max_idx_train, min_idx_train, ratio, X_training, X_val, labels_training, labels_val
    

#Saving 
#pred average
preds = (stacking_test[:,0] + stacking_test[:,1] + stacking_test[:,2] + stacking_test[:,3] + stacking_test[:,4])/5

print("Features importances...")
importance = clr.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))
ft = pd.DataFrame(importance, columns=['feature', 'fscore'])
ft.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
plt.gcf().savefig(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/xgb_gbtree_test" + preprocessing  + ".png")
print("Writing output...")
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = preds *.75
sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/xgb_gbtree_test" + preprocessing  + ".csv", index=False)

#--- pred training for ensemble
print("Writing training pred output...")
sub = pd.DataFrame()
sub['is_duplicate'] = stacking_train *.75
sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/xgb_gbtree_train" + preprocessing  + ".csv", index=False)


#http://stackoverflow.com/questions/33749735/how-to-install-xgboost-package-in-python-windows-platform/35119904#35119904








#==============================================================================
# 
# 
# ####################################################################
# # XGBoost gblinear
# ####################################################################
# print("Starting XGB gblinear")
# 
# import xgboost as xgb
# 
# print("Starting XGB gblinear")
# 
# params['booster'] = 'gblinear'
# 
# print("Training data: X_train: {}, labels: {}, X_test: {}".format(x_train.shape, len(labels), x_test_real.shape))
# 
# ROUNDS = 1000
# 
# #training
# def create_feature_map(features):
# 	outfile = open('xgb.fmap', 'w')
# 	i = 0
# 	for feat in features:
# 		outfile.write('{0}\t{1}\tq\n'.format(i, feat))
# 		i = i + 1
# 	outfile.close()
# create_feature_map(feature_names)
# 
# print("Will train XGB for {} rounds, RandomSeed: {}".format(ROUNDS, RS))
# 
# ###KFold
# stacking_train_xgb = np.empty(shape=(len(labels_real)))
# for kth, (train_index, test_index) in enumerate(kf.split(x_train_real)):
#     print("\n********************** FOLD ", kth+1, " ******************\n")
#     max_idx_test  = np.amax(test_index)+1
#     min_idx_test  = np.amin(test_index)
#     max_idx_train = np.amax(train_index)+1
#     min_idx_train = np.amin(train_index)
#     X_training, X_val, labels_training, labels_val = train_test_split(x_train[min_idx_train:max_idx_train], labels[min_idx_train:max_idx_train], test_size=0.2)#, random_state=RS)
#     xg_train = xgb.DMatrix(X_training, label=labels_training)
#     xg_val = xgb.DMatrix(X_val, label=labels_val)
#     watchlist  = [(xg_train,'train'), (xg_val,'eval')]
#     
#     clr = xgb.train(params, xg_train, ROUNDS, watchlist, early_stopping_rounds=30, verbose_eval=10)
#     
#     #predict training ids of the testing fold
#     stacking_train_xgb[min_idx_test:max_idx_test] = clr.predict(xgb.DMatrix(x_train[min_idx_test:max_idx_test]))
# del max_idx_test, min_idx_test, max_idx_train, min_idx_train, ratio, X_training, X_val, labels_training, labels_val
#     
# 
# #Saving 
# #pred test
# preds = clr.predict(xgb.DMatrix(x_test_real))
# print("Features importances...")
# importance = clr.get_fscore(fmap='xgb.fmap')
# importance = sorted(importance.items(), key=operator.itemgetter(1))
# ft = pd.DataFrame(importance, columns=['feature', 'fscore'])
# ft.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
# plt.gcf().savefig('features_importance.png')
# print("Writing output...")
# sub = pd.DataFrame()
# sub['test_id'] = df_test['test_id']
# sub['is_duplicate'] = preds *.75
# sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/xgb_gblinear_test_kf.csv", index=False)
# 
# #--- pred training for ensemble
# print("Writing training pred output...")
# sub = pd.DataFrame()
# sub['is_duplicate'] = stacking_train_xgb *.75
# sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/xgb_gblinear_train_kf.csv", index=False)
# 
# 
# #http://stackoverflow.com/questions/33749735/how-to-install-xgboost-package-in-python-windows-platform/35119904#35119904
# 
# 
# 
# 
#==============================================================================








#==============================================================================
# 
# ####################################################################
# # XGBoost dart
# ####################################################################
# print("Starting XGB dart")
# import xgboost as xgb
# 
# print("Starting XGB dart")
# 
# params['booster'] = 'dart'
# 
# print("Training data: X_train: {}, labels: {}, X_test: {}".format(x_train.shape, len(labels), x_test_real.shape))
# 
# ROUNDS = 10000
# 
# #training
# def create_feature_map(features):
# 	outfile = open('xgb.fmap', 'w')
# 	i = 0
# 	for feat in features:
# 		outfile.write('{0}\t{1}\tq\n'.format(i, feat))
# 		i = i + 1
# 	outfile.close()
# create_feature_map(feature_names)
# 
# print("Will train XGB for {} rounds, RandomSeed: {}".format(ROUNDS, RS))
# 
# ###KFold
# stacking_train_xgb = np.empty(shape=(len(labels_real)))
# for kth, (train_index, test_index) in enumerate(kf.split(x_train_real)):
#     print("\n********************** FOLD ", kth+1, " ******************\n")
#     max_idx_test  = np.amax(test_index)+1
#     min_idx_test  = np.amin(test_index)
#     max_idx_train = np.amax(train_index)+1
#     min_idx_train = np.amin(train_index)
#     X_training, X_val, labels_training, labels_val = train_test_split(x_train[min_idx_train:max_idx_train], labels[min_idx_train:max_idx_train], test_size=0.2)#, random_state=RS)
#     xg_train = xgb.DMatrix(X_training, label=labels_training)
#     xg_val = xgb.DMatrix(X_val, label=labels_val)
#     watchlist  = [(xg_train,'train'), (xg_val,'eval')]
#     
#     clr = xgb.train(params, xg_train, ROUNDS, watchlist, early_stopping_rounds=30, verbose_eval=10)
#     
#     #predict training ids of the testing fold
#     stacking_train_xgb[min_idx_test:max_idx_test] = clr.predict(xgb.DMatrix(x_train[min_idx_test:max_idx_test]))
# del max_idx_test, min_idx_test, max_idx_train, min_idx_train, ratio, X_training, X_val, labels_training, labels_val
#     
# 
# #Saving 
# #pred test
# preds = clr.predict(xgb.DMatrix(x_test_real))
# print("Features importances...")
# importance = clr.get_fscore(fmap='xgb.fmap')
# importance = sorted(importance.items(), key=operator.itemgetter(1))
# ft = pd.DataFrame(importance, columns=['feature', 'fscore'])
# ft.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
# plt.gcf().savefig('features_importance.png')
# print("Writing output...")
# sub = pd.DataFrame()
# sub['test_id'] = df_test['test_id']
# sub['is_duplicate'] = preds *.75
# sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/xgb_dart_test_kf.csv", index=False)
# 
# #--- pred training for ensemble
# print("Writing training pred output...")
# sub = pd.DataFrame()
# sub['is_duplicate'] = stacking_train_xgb *.75
# sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/xgb_dart_train_kf.csv", index=False)
# 
# 
# #http://stackoverflow.com/questions/33749735/how-to-install-xgboost-package-in-python-windows-platform/35119904#35119904
# 
#==============================================================================















####################################################################
# preprocessing Imputer
####################################################################


from sklearn.preprocessing import Imputer
x_train_real = Imputer().fit_transform(x_train_real)
x_test_real = Imputer().fit_transform(x_test_real)





#==============================================================================
# 
# 
# ####################################################################
# # Linear regression
# ####################################################################
# print("Linear regression")
# 
# from sklearn.linear_model import LinearRegression
# 
# stacking_train_linearReg = np.empty(shape=(len(labels_real)))
# for kth, (train_index, test_index) in enumerate(kf.split(x_train_real)):    
#     print("\n********************** FOLD ", kth+1, " ******************\n")
#     max_idx_test  = np.amax(test_index)+1
#     min_idx_test  = np.amin(test_index)
#     max_idx_train = np.amax(train_index)+1
#     min_idx_train = np.amin(train_index)
#         
#     classifier = LinearRegression(n_jobs=-1)
#     classifier.fit(x_train_real[min_idx_train:max_idx_train], labels[min_idx_train:max_idx_train])
#     
#     #predict training ids of the testing fold
#     stacking_train_linearReg[min_idx_test:max_idx_test] = classifier.predict(x_train_real[min_idx_test:max_idx_test])
# del max_idx_test, min_idx_test, max_idx_train, min_idx_train
#     
# 
# #Saving 
# #pred test
# preds = classifier.predict(x_test_real)
# print("Writing output...")
# sub = pd.DataFrame()
# sub['test_id'] = df_test['test_id']
# sub['is_duplicate'] = preds *.75
# sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/linReg_test_kf" + preprocessing  + ".csv", index=False)
# 
# #--- pred training for ensemble
# print("Writing training pred output...")
# sub = pd.DataFrame()
# sub['is_duplicate'] = stacking_train_linearReg *.75
# sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/linReg_train_kf" + preprocessing  + ".csv", index=False)
# 
# 
#==============================================================================







####################################################################
# preprocessing StandardScaler
####################################################################

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_real = sc.fit_transform(x_train_real)
x_test_real = sc.transform(x_test_real)






#==============================================================================
# 
# 
# ####################################################################
# # logistic regression
# ####################################################################
# 
# #print(x_train.isnull().sum())
# #x_train = x_train.fillna(x_train.mean())
# #x_train_real = x_train_real.fillna(x_train_real.mean())
# #print(np.isfinite(x_train_real).sum())
# 
# from sklearn.linear_model import LogisticRegression
# 
# stacking_train_logisticReg = np.empty(shape=(len(labels_real)))
# for kth, (train_index, test_index) in enumerate(kf.split(x_train_real)):    
#     print("\n********************** FOLD ", kth+1, " ******************\n")
#     max_idx_test  = np.amax(test_index)+1
#     min_idx_test  = np.amin(test_index)
#     max_idx_train = np.amax(train_index)+1
#     min_idx_train = np.amin(train_index)
#     #print(np.any(np.isnan(x_train_real[min_idx_train:max_idx_train])))
#     #print(np.argwhere(np.isnan(x_train_real[min_idx_train:max_idx_train])))
#     #print(list(map(tuple, np.where(np.any(np.isnan(x_train_real[min_idx_train:max_idx_train]))))))
#     #print(min_idx_test, " ",  max_idx_test)
#     ##print(np.isnan(labels[min_idx_train:max_idx_train]).sum())
#     
#     classifier = LogisticRegression(n_jobs = -1)
#     classifier.fit(x_train_real[min_idx_train:max_idx_train], labels[min_idx_train:max_idx_train])
#     
#     #predict training ids of the testing fold
#     stacking_train_logisticReg[min_idx_test:max_idx_test] = classifier.predict(x_train_real[min_idx_test:max_idx_test])
# del max_idx_test, min_idx_test, max_idx_train, min_idx_train
#     
# 
# #Saving 
# #pred test
# preds = classifier.predict(x_test_real)
# print("Writing output...")
# sub = pd.DataFrame()
# sub['test_id'] = df_test['test_id']
# sub['is_duplicate'] = preds *.75
# sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/lgcReg_test_kf" + preprocessing  + ".csv", index=False)
# 
# #--- pred training for ensemble
# print("Writing training pred output...")
# sub = pd.DataFrame()
# sub['is_duplicate'] = stacking_train_logisticReg *.75
# sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/lgcReg_train_kf" + preprocessing  + ".csv", index=False)
# 
# 
# 
#==============================================================================






#==============================================================================
# 
# 
# ####################################################################
# # KNNs
# ####################################################################
# 
# from sklearn.neighbors import KNeighborsClassifier
# KNNs = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
# 
# for KNN in KNNs:
#     stacking_train_KNN = np.empty(shape=(len(labels_real)))
#     for kth, (train_index, test_index) in enumerate(kf.split(x_train_real)):    
#         print("\n********************** FOLD ", kth+1, " KNN ",KNN," ******************\n")
#         max_idx_test  = np.amax(test_index)+1
#         min_idx_test  = np.amin(test_index)
#         max_idx_train = np.amax(train_index)+1
#         min_idx_train = np.amin(train_index)
#         
#         classifier = KNeighborsClassifier(n_neighbors = KNN, metric = 'minkowski', p = 2, n_jobs = -1)
#         classifier.fit(x_train_real[min_idx_train:max_idx_train], labels[min_idx_train:max_idx_train])
#     
#         #predict training ids of the testing fold
#         stacking_train_KNN[min_idx_test:max_idx_test] = classifier.predict(x_train_real[min_idx_test:max_idx_test])
#     del max_idx_test, min_idx_test, max_idx_train, min_idx_train
#     
#     
#     #Saving 
#     #pred test
#     preds = classifier.predict(x_test_real)
#     print("Writing output...")
#     sub = pd.DataFrame()
#     sub['test_id'] = df_test['test_id']
#     sub['is_duplicate'] = preds
#     sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/KNN_" + KNN + "" + preprocessing  + ".csv", index=False)
#     
#     #--- pred training for ensemble
#     print("Writing training pred output...")
#     sub = pd.DataFrame()
#     sub['is_duplicate'] = stacking_train_KNN
#     sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/KNN_" + KNN + "" + preprocessing  + ".csv", index=False)
# 
# 
# 
#==============================================================================




#==============================================================================
# 
# 
# 
# 
# ####################################################################
# # SVM
# ####################################################################
# from sklearn.svm import SVC
# 
# stacking_train_SVC = np.empty(shape=(len(labels_real)))
# for kth, (train_index, test_index) in enumerate(kf.split(x_train_real)):    
#     print("\n********************** FOLD ", kth+1, " ******************\n")
#     max_idx_test  = np.amax(test_index)+1
#     min_idx_test  = np.amin(test_index)
#     max_idx_train = np.amax(train_index)+1
#     min_idx_train = np.amin(train_index)
#     
#     classifier = SVC(kernel = 'rbf')
#     classifier.fit(x_train_real[min_idx_train:max_idx_train], labels[min_idx_train:max_idx_train])
# 
#     #predict training ids of the testing fold
#     stacking_train_SVC[min_idx_test:max_idx_test] = classifier.predict(x_train_real[min_idx_test:max_idx_test])
# del max_idx_test, min_idx_test, max_idx_train, min_idx_train
# 
# 
# #Saving 
# #pred test
# preds = classifier.predict(x_test_real)
# print("Writing output...")
# sub = pd.DataFrame()
# sub['test_id'] = df_test['test_id']
# sub['is_duplicate'] = preds
# sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/SVC_test_kf" + preprocessing  + ".csv", index=False)
# 
# #--- pred training for ensemble
# print("Writing training pred output...")
# sub = pd.DataFrame()
# sub['is_duplicate'] = stacking_train_SVC
# sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/SVC_train_kf" + preprocessing  + ".csv", index=False)
# 
# 
# 
# 
# 
#==============================================================================



#==============================================================================
# 
# 
# ####################################################################
# # naive bayes
# ####################################################################
# from sklearn.naive_bayes import GaussianNB
# 
# stacking_train_naive_bayes = np.empty(shape=(len(labels_real)))
# for kth, (train_index, test_index) in enumerate(kf.split(x_train_real)):    
#     print("\n********************** FOLD ", kth+1, " ******************\n")
#     max_idx_test  = np.amax(test_index)+1
#     min_idx_test  = np.amin(test_index)
#     max_idx_train = np.amax(train_index)+1
#     min_idx_train = np.amin(train_index)
#     
#     classifier = GaussianNB()
#     classifier.fit(x_train_real[min_idx_train:max_idx_train], labels[min_idx_train:max_idx_train])
# 
#     #predict training ids of the testing fold
#     stacking_train_naive_bayes[min_idx_test:max_idx_test] = classifier.predict(x_train_real[min_idx_test:max_idx_test])
# del max_idx_test, min_idx_test, max_idx_train, min_idx_train
# 
# 
# #Saving 
# #pred test
# preds = classifier.predict(x_test_real)
# print("Writing output...")
# sub = pd.DataFrame()
# sub['test_id'] = df_test['test_id']
# sub['is_duplicate'] = preds
# sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/naive_bayes_test_kf" + preprocessing  + ".csv", index=False)
# 
# #--- pred training for ensemble
# print("Writing training pred output...")
# sub = pd.DataFrame()
# sub['is_duplicate'] = stacking_train_naive_bayes
# sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/naive_bayes_train_kf" + preprocessing  + ".csv", index=False)
# 
# 
# 
#==============================================================================













#==============================================================================
# 
# ####################################################################
# # random forest
# ####################################################################
# print("Random Forest")
# from sklearn.ensemble import RandomForestClassifier
# 
# 
# stacking_train = np.empty(shape=(len(labels_real)))
# stacking_test = np.empty(shape=(len(x_test_real),K))
# for kth, (train_index, test_index) in enumerate(kf.split(x_train_real)):    
#     print("\n********************** FOLD ", kth+1, " ******************\n")
#     max_idx_test  = np.amax(test_index)+1
#     min_idx_test  = np.amin(test_index)
#     max_idx_train = np.amax(train_index)+1
#     min_idx_train = np.amin(train_index)
# 
#     classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', n_jobs = -1)
#     classifier.fit(x_train_real[min_idx_train:max_idx_train], labels[min_idx_train:max_idx_train])
#     #predict training ids of the testing fold
#     stacking_train[min_idx_test:max_idx_test] = classifier.predict(x_train_real[min_idx_test:max_idx_test])
# 
#     stacking_test[:,kth] = classifier.predict(x_test_real)
# #Saving 
# #pred test
# preds = (stacking_test[:,0] + stacking_test[:,1] + stacking_test[:,2] + stacking_test[:,3] + stacking_test[:,4])/5
# print("Writing output...")
# sub = pd.DataFrame()
# sub['test_id'] = df_test['test_id']
# sub['is_duplicate'] = preds
# sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/RF_test" + preprocessing  + ".csv", index=False)
# 
# #--- pred training for ensemble
# print("Writing training pred output...")
# sub = pd.DataFrame()
# sub['is_duplicate'] = stacking_train
# sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/RF_train" + preprocessing  + ".csv", index=False)
# 
# 
# 
# 
# 
# 
#==============================================================================












####################################################################
# ANN
####################################################################
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

stacking_train = np.empty(shape=(len(labels_real)))
stacking_test = np.empty(shape=(len(x_test_real),K))
for kth, (train_index, test_index) in enumerate(kf.split(x_train_real)):    
    print("\n********************** FOLD ", kth+1, " ******************\n")
    max_idx_test  = np.amax(test_index)+1
    min_idx_test  = np.amin(test_index)
    max_idx_train = np.amax(train_index)+1
    min_idx_train = np.amin(train_index)

    # Initialising the ANN
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 300, kernel_initializer = 'truncated_normal', activation='relu', input_dim = x_train_real.shape[1]))
    classifier.add(Dropout(rate = 0.3))
    classifier.add(BatchNormalization())
    
    classifier.add(Dense(200, kernel_initializer = 'truncated_normal', activation='relu'))
    classifier.add(Dropout(rate = 0.3))
    classifier.add(BatchNormalization())
    
    classifier.add(Dense(200, kernel_initializer = 'truncated_normal', activation='relu'))
    classifier.add(Dropout(rate = 0.3))
    classifier.add(BatchNormalization())
    
    classifier.add(Dense(units = 1, kernel_initializer = 'truncated_normal', activation = 'sigmoid'))
    
    early_stopping =EarlyStopping(monitor='val_loss', patience = 20)
    model_checkpoint = ModelCheckpoint('weightsANN_X_final_Features_MyMagic3.h5', save_best_only=True, save_weights_only=True, verbose=1)
    
    classifier.compile(optimizer = 'nadam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifier.fit(x_train_real[min_idx_train:max_idx_train], labels[min_idx_train:max_idx_train], batch_size=4096, epochs=300, verbose=1, validation_split=0.1, shuffle=True, callbacks=[early_stopping, model_checkpoint])
    
    classifier.load_weights('weightsANN_X_final_Features_MyMagic3.h5')
    #predict training ids of the testing fold
    stacking_train[min_idx_test:max_idx_test] = classifier.predict(x_train_real[min_idx_test:max_idx_test], batch_size=4096, verbose=1)[:,0]
    #pred test
    stacking_test[:,kth] = classifier.predict(x_test_real, batch_size=4096, verbose=1)[:,0]

#Saving 
#pred test
preds = (stacking_test[:,0] + stacking_test[:,1] + stacking_test[:,2] + stacking_test[:,3] + stacking_test[:,4])/5
print("Writing output...")
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = preds
sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/ANN_test" + preprocessing  + ".csv", index=False)

#--- pred training for ensemble
print("Writing training pred output...")
sub = pd.DataFrame()
sub['is_duplicate'] = stacking_train
sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/ANN_train" + preprocessing  + ".csv", index=False)




















####################################################################
# GRU
####################################################################
import re
import csv
import codecs
import numpy as np
import pandas as pd

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Bidirectional, GRU
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

########################################
## set directories and parameters
BASE_DIR = 'F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/'
EMBEDDING_FILE = 'F:/DS-main/BigFiles/GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILE = BASE_DIR + 'train_spacyLemma_trigram.csv'
TEST_DATA_FILE = BASE_DIR + 'test_spacyLemma_trigram.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = 300
num_dense = 200
rate_drop_lstm = 0.3
rate_drop_dense = 0.3

act = 'relu'
re_weight = True # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
        rate_drop_dense)

########################################
## index word vectors
print('Indexing word vectors')

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
        binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))

########################################
## process texts in datasets
print('Processing text dataset')

# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text):
    # Clean the text, with the option to remove stopwords and to stem words.
    if text == "":
        text = "-empty-"
    # Return a list of words
    return(text)

texts_1 = [] 
texts_2 = []
labels = []
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts_1.append(text_to_wordlist(values[3]))
        texts_2.append(text_to_wordlist(values[4]))
        labels.append(int(values[5]))
        if (len(texts_1)%100000 == 0 ):
            print ('proc train',len(texts_1))
print('Found %s texts in train.csv' % len(texts_1))

test_texts_1 = []
test_texts_2 = []
test_ids = []
with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        test_texts_1.append(text_to_wordlist(values[1]))
        test_texts_2.append(text_to_wordlist(values[2]))
        test_ids.append(values[0])
        if (len(test_texts_1)%100000 == 0 ):
            print ('test',len(test_texts_1))
print('Found %s texts in test.csv' % len(test_texts_1))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)

sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_ids = np.array(test_ids)

del sequences_1, sequences_2, test_sequences_1, test_sequences_2, texts_1, texts_2, test_texts_1, test_texts_2

########################################
## prepare embeddings
print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))



########################################
## sample train/validation data

data_1_dbl = np.vstack((data_1, data_2))
data_2_dbl = np.vstack((data_2, data_1))
labels_dbl = np.concatenate((labels, labels))

K = 3
kf = KFold(n_splits = K)


stacking_train = np.empty(shape=(len(data_1_dbl)))
stacking_test = np.empty(shape=(len(test_data_1),K))
for kth, (train_index, test_index) in enumerate(kf.split(data_1_dbl)):    
    print("\n********************** FOLD ", kth+1, " ******************\n")
    max_idx_test  = np.amax(test_index)+1
    min_idx_test  = np.amin(test_index)
    max_idx_train = np.amax(train_index)+1
    min_idx_train = np.amin(train_index)
    
    
    weight_val = np.ones(len(labels_dbl[min_idx_test:max_idx_test]))
    if re_weight:
        weight_val *= 0.472001959
        weight_val[labels[min_idx_test:max_idx_test]==0] = 1.309028344
        
    ########################################
    ## define the model structure
    model = Sequential()
    
    embedding_layer = Embedding(nb_words,
            EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_SEQUENCE_LENGTH,
            trainable=False)
    #lstm_layer0 = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=True)
    gru_layer = GRU(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)
    #lstm_layer1 = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)
    #lstm_layer = Bidirectional(GRU(num_lstm, return_sequences=True))
    
    
    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = gru_layer(embedded_sequences_1)
    #x2 = lstm_layer1(x1)
    
    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = gru_layer(embedded_sequences_2)
    #y2 = lstm_layer1(y1)
    
    merged = concatenate([x1, y1])
    #Imerged = concatenate([x2, y2])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    
    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    
    preds = Dense(1, activation='sigmoid')(merged)
    
    ########################################
    ## add class weight
    if re_weight:
        class_weight = {0: 1.309028344, 1: 0.472001959}
    else:
        class_weight = None
    
    ########################################
    ## train the model
    model = Model(inputs=[sequence_1_input, sequence_2_input], \
            outputs=preds)
    model.compile(loss='binary_crossentropy',
            optimizer='nadam',
            metrics=['acc'])
    #model.summary()
    print(STAMP)
    batch_size=256
    early_stopping =EarlyStopping(monitor='val_loss', patience=3)
    bst_model_path = STAMP + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True, verbose=1)
    
    hist = model.fit([data_1_dbl[min_idx_train:max_idx_train], data_2_dbl[min_idx_train:max_idx_train]], labels_dbl[min_idx_train:max_idx_train], \
        validation_data=([data_1_dbl[min_idx_test:max_idx_test], data_2_dbl[min_idx_test:max_idx_test]], labels_dbl[min_idx_test:max_idx_test], weight_val), \
        epochs=2000, batch_size=batch_size, shuffle=True, \
        class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

    model.load_weights(bst_model_path)
    bst_val_score = min(hist.history['val_loss'])
       #predict training ids of the testing fold
    stacking_train[min_idx_test:max_idx_test] = model.predict([data_1_dbl[min_idx_test:max_idx_test], data_2_dbl[min_idx_test:max_idx_test]], batch_size=batch_size, verbose=1)[:,0]
    stacking_train[min_idx_test:max_idx_test] += model.predict([data_2_dbl[min_idx_test:max_idx_test], data_1_dbl[min_idx_test:max_idx_test]], batch_size=batch_size, verbose=1)[:,0]
    stacking_train[min_idx_test:max_idx_test] /= 2
    #pred test
    stacking_test[:,kth] = model.predict([test_data_1, test_data_2], batch_size=batch_size, verbose=1)[:,0]
    stacking_test[:,kth] += model.predict([test_data_2, test_data_1], batch_size=batch_size, verbose=1)[:,0]
    stacking_test[:,kth] /= 2


#Saving 
#pred test
preds = (stacking_test[:,0] + stacking_test[:,1] + stacking_test[:,2] + stacking_test[:,3] + stacking_test[:,4])/5
print("Writing output...")
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = preds
sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/GRU_test_kf" + preprocessing  + "_epochs3.csv", index=False)

#--- pred training for ensemble
print("Writing training pred output...")
sub = pd.DataFrame()
sub['is_duplicate'] = stacking_train
sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/GRU_train_kf" + preprocessing  + "_epochs3.csv", index=False)

























####################################################################
# LSTM
####################################################################
stacking_train = np.empty(shape=(len(data_1_dbl)))
stacking_test = np.empty(shape=(len(test_data_1),K))
for kth, (train_index, test_index) in enumerate(kf.split(data_1_dbl)):    
    print("\n********************** FOLD ", kth+1, " ******************\n")
    max_idx_test  = np.amax(test_index)+1
    min_idx_test  = np.amin(test_index)
    max_idx_train = np.amax(train_index)+1
    min_idx_train = np.amin(train_index)
    
    
    weight_val = np.ones(len(labels_dbl[min_idx_test:max_idx_test]))
    if re_weight:
        weight_val *= 0.472001959
        weight_val[labels[min_idx_test:max_idx_test]==0] = 1.309028344
        
    ########################################
    ## define the model structure
    model = Sequential()
    
    embedding_layer = Embedding(nb_words,
            EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_SEQUENCE_LENGTH,
            trainable=False)
    lstm_layer0 = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=True)
    #gru_layer = GRU(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)
    lstm_layer1 = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)
    #lstm_layer = Bidirectional(GRU(num_lstm, return_sequences=True))
    
    
    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = gru_layer(embedded_sequences_1)
    x1 = lstm_layer1(x1)
    
    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = gru_layer(embedded_sequences_2)
    y1 = lstm_layer1(y1)
    
    #merged = concatenate([x1, y1])
    merged = concatenate([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    
    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    
    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    
    preds = Dense(1, activation='sigmoid')(merged)
    
    ########################################
    ## add class weight
    if re_weight:
        class_weight = {0: 1.309028344, 1: 0.472001959}
    else:
        class_weight = None
    
    ########################################
    ## train the model
    model = Model(inputs=[sequence_1_input, sequence_2_input], \
            outputs=preds)
    model.compile(loss='binary_crossentropy',
            optimizer='nadam',
            metrics=['acc'])
    #model.summary()
    print(STAMP)
    
    early_stopping =EarlyStopping(monitor='val_loss', patience=5)
    bst_model_path = STAMP + 'dblLSTM.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True, verbose=1)
    
    hist = model.fit([data_1_dbl[min_idx_train:max_idx_train], data_2_dbl[min_idx_train:max_idx_train]], labels_dbl[min_idx_train:max_idx_train], \
        validation_data=([data_1_dbl[min_idx_test:max_idx_test], data_2_dbl[min_idx_test:max_idx_test]], labels_dbl[min_idx_test:max_idx_test], weight_val), \
        epochs=200, batch_size=512, shuffle=True, \
        class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

    model.load_weights(bst_model_path)
    bst_val_score = min(hist.history['val_loss'])
       #predict training ids of the testing fold
    stacking_train[min_idx_test:max_idx_test] = model.predict([data_1_dbl[min_idx_test:max_idx_test], data_2_dbl[min_idx_test:max_idx_test]], batch_size=512, verbose=1)[:,0]
    stacking_train[min_idx_test:max_idx_test] += model.predict([data_2_dbl[min_idx_test:max_idx_test], data_1_dbl[min_idx_test:max_idx_test]], batch_size=512, verbose=1)[:,0]
    stacking_train[min_idx_test:max_idx_test] /= 2
    #pred test
    stacking_test[:,kth] = model.predict([test_data_1, test_data_2], batch_size=128, verbose=1)[:,0]
    stacking_test[:,kth] += model.predict([test_data_2, test_data_1], batch_size=128, verbose=1)[:,0]
    stacking_test[:,kth] /= 2


#Saving 
#pred test
preds = (stacking_test[:,0] + stacking_test[:,1] + stacking_test[:,2] + stacking_test[:,3] + stacking_test[:,4])/5
print("Writing output...")
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = preds
sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/LSTM_dbl_test_kf" + preprocessing  + ".csv", index=False)

#--- pred training for ensemble
print("Writing training pred output...")
sub = pd.DataFrame()
sub['is_duplicate'] = stacking_train
sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/LSTM_dbl_train_kf" + preprocessing  + ".csv", index=False)





























#==============================================================================
# New Lystdo LSTM    https://www.kaggle.com/lystdo/lb-0-18-lstm-with-glove-and-magic-features
#==============================================================================
########################################
## import packages
########################################
import csv
import codecs
import numpy as np
import pandas as pd

from collections import defaultdict

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import StandardScaler


########################################
## set directories and parameters
########################################
BASE_DIR = 'F:/DS-main/'
EMBEDDING_FILE = BASE_DIR + 'BigFiles/glove.42B.300d/glove.42B.300d.txt'
TRAIN_DATA_FILE = BASE_DIR + 'Kaggle-main/Quora Question Pairs - inputs/train_spacyLemma_trigram.csv'
TEST_DATA_FILE = BASE_DIR + 'Kaggle-main/Quora Question Pairs - inputs/test_spacyLemma_trigram.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = 300
num_dense = 300
rate_drop_lstm = 0.3
rate_drop_dense = 0.3

act = 'relu'
re_weight = True # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
        rate_drop_dense)

########################################
## index word vectors
########################################
print('Indexing word vectors')

embeddings_index = {}
f = open(EMBEDDING_FILE, encoding="utf-8")
count = 0
for line in f:
    if count == 0:
        count = 1
        continue
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %d word vectors of glove.' % len(embeddings_index))

########################################
## process texts in datasets
########################################
print('Processing text dataset')

# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    if text == "":
        text = "-empty-"
    return(text)

texts_1 = [] 
texts_2 = []
labels = []
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts_1.append(text_to_wordlist(values[3]))
        texts_2.append(text_to_wordlist(values[4]))
        labels.append(int(values[5]))
print('Found %s texts in train.csv' % len(texts_1))

test_texts_1 = []
test_texts_2 = []
test_ids = []
with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        test_texts_1.append(text_to_wordlist(values[1]))
        test_texts_2.append(text_to_wordlist(values[2]))
        test_ids.append(values[0])
print('Found %s texts in test.csv' % len(test_texts_1))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)

sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_ids = np.array(test_ids)

########################################
## generate leaky features
########################################

train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)

ques = pd.concat([train_df[['question1', 'question2']], \
        test_df[['question1', 'question2']]], axis=0).reset_index(drop='index')
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

train_df['q1_q2_intersect'] = train_df.apply(q1_q2_intersect, axis=1, raw=True)
train_df['q1_freq'] = train_df.apply(q1_freq, axis=1, raw=True)
train_df['q2_freq'] = train_df.apply(q2_freq, axis=1, raw=True)

test_df['q1_q2_intersect'] = test_df.apply(q1_q2_intersect, axis=1, raw=True)
test_df['q1_freq'] = test_df.apply(q1_freq, axis=1, raw=True)
test_df['q2_freq'] = test_df.apply(q2_freq, axis=1, raw=True)

leaks = train_df[['q1_q2_intersect', 'q1_freq', 'q2_freq']]
test_leaks = test_df[['q1_q2_intersect', 'q1_freq', 'q2_freq']]

ss = StandardScaler()
ss.fit(np.vstack((leaks, test_leaks)))
leaks = ss.transform(leaks)
test_leaks = ss.transform(test_leaks)

########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

########################################
## sample train/validation data
########################################
#np.random.seed(1234)
perm = np.random.permutation(len(data_1))
idx_train = perm[:int(len(data_1)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(data_1)*(1-VALIDATION_SPLIT)):]

data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
leaks_train = np.vstack((leaks[idx_train], leaks[idx_train]))
labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
leaks_val = np.vstack((leaks[idx_val], leaks[idx_val]))
labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

weight_val = np.ones(len(labels_val))
if re_weight:
    weight_val *= 0.472001959
    weight_val[labels_val==0] = 1.309028344

########################################
## define the model structure
########################################
embedding_layer = Embedding(nb_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)
lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)

leaks_input = Input(shape=(leaks.shape[1],))
leaks_dense = Dense(num_dense/2, activation=act)(leaks_input)

merged = concatenate([x1, y1, leaks_dense])
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

preds = Dense(1, activation='sigmoid')(merged)

########################################
## add class weight
########################################
if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None

########################################
## train the model
########################################
model = Model(inputs=[sequence_1_input, sequence_2_input, leaks_input], \
        outputs=preds)
model.compile(loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['acc'])
#model.summary()
print(STAMP)

early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([data_1_train, data_2_train, leaks_train], labels_train, \
        validation_data=([data_1_val, data_2_val, leaks_val], labels_val, weight_val), \
        epochs=200, batch_size=2048, shuffle=True, \
        class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])

########################################
## make the submission
########################################
print('Start making the submission before fine-tuning')

preds = model.predict([test_data_1, test_data_2, test_leaks], batch_size=8192, verbose=1)
preds += model.predict([test_data_2, test_data_1, test_leaks], batch_size=8192, verbose=1)
preds /= 2

submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds.ravel()})
submission.to_csv('%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)



