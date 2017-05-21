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
x = pd.read_csv(drive + ':/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/x_final_features_extendedMagic2.csv', header=0) 


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
params['eta'] = 0.07 #0.11
params['max_depth'] = 6 #5
params['silent'] = 1 
#params['updater'] = 'grow_gpu'
print("Training data: X_train: {}, labels: {}, X_test: {}".format(x_train.shape, len(labels), x_test_real.shape))

ROUNDS = 10000

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
stacking_train_xgb = np.empty(shape=(len(labels_real)))
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
    
    clr = xgb.train(params, xg_train, ROUNDS, watchlist, early_stopping_rounds=20, verbose_eval=1)
    
    #predict training ids of the testing fold
    stacking_train_xgb[min_idx_test:max_idx_test] = clr.predict(xgb.DMatrix(x_train[min_idx_test:max_idx_test]))
del max_idx_test, min_idx_test, max_idx_train, min_idx_train, ratio, X_training, X_val, labels_training, labels_val
    

#Saving 
#pred test
preds = clr.predict(xgb.DMatrix(x_test_real))
print("Features importances...")
importance = clr.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))
ft = pd.DataFrame(importance, columns=['feature', 'fscore'])
ft.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
plt.gcf().savefig('features_importance.png')
print("Writing output...")
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = preds *.75
sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/xgb_gbtree_test_extendedMagic2_kf.csv", index=False)

#--- pred training for ensemble
print("Writing training pred output...")
sub = pd.DataFrame()
sub['is_duplicate'] = stacking_train_xgb *.75
sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/xgb_gbtree_train_extendedMagic2_kf.csv", index=False)


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







####################################################################
# Linear regression
####################################################################
print("Linear regression")

from sklearn.linear_model import LinearRegression

stacking_train_linearReg = np.empty(shape=(len(labels_real)))
for kth, (train_index, test_index) in enumerate(kf.split(x_train_real)):    
    print("\n********************** FOLD ", kth+1, " ******************\n")
    max_idx_test  = np.amax(test_index)+1
    min_idx_test  = np.amin(test_index)
    max_idx_train = np.amax(train_index)+1
    min_idx_train = np.amin(train_index)
        
    classifier = LinearRegression(n_jobs=-1)
    classifier.fit(x_train_real[min_idx_train:max_idx_train], labels[min_idx_train:max_idx_train])
    
    #predict training ids of the testing fold
    stacking_train_linearReg[min_idx_test:max_idx_test] = classifier.predict(x_train_real[min_idx_test:max_idx_test])
del max_idx_test, min_idx_test, max_idx_train, min_idx_train
    

#Saving 
#pred test
preds = classifier.predict(x_test_real)
print("Writing output...")
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = preds *.75
sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/linReg_test_kf.csv", index=False)

#--- pred training for ensemble
print("Writing training pred output...")
sub = pd.DataFrame()
sub['is_duplicate'] = stacking_train_linearReg *.75
sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/linReg_train_kf.csv", index=False)









####################################################################
# preprocessing StandardScaler
####################################################################

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_real = sc.fit_transform(x_train_real)
x_test_real = sc.transform(x_test_real)








####################################################################
# logistic regression
####################################################################

#print(x_train.isnull().sum())
#x_train = x_train.fillna(x_train.mean())
#x_train_real = x_train_real.fillna(x_train_real.mean())
#print(np.isfinite(x_train_real).sum())

from sklearn.linear_model import LogisticRegression

stacking_train_logisticReg = np.empty(shape=(len(labels_real)))
for kth, (train_index, test_index) in enumerate(kf.split(x_train_real)):    
    print("\n********************** FOLD ", kth+1, " ******************\n")
    max_idx_test  = np.amax(test_index)+1
    min_idx_test  = np.amin(test_index)
    max_idx_train = np.amax(train_index)+1
    min_idx_train = np.amin(train_index)
    #print(np.any(np.isnan(x_train_real[min_idx_train:max_idx_train])))
    #print(np.argwhere(np.isnan(x_train_real[min_idx_train:max_idx_train])))
    #print(list(map(tuple, np.where(np.any(np.isnan(x_train_real[min_idx_train:max_idx_train]))))))
    #print(min_idx_test, " ",  max_idx_test)
    ##print(np.isnan(labels[min_idx_train:max_idx_train]).sum())
    
    classifier = LogisticRegression(n_jobs = -1)
    classifier.fit(x_train_real[min_idx_train:max_idx_train], labels[min_idx_train:max_idx_train])
    
    #predict training ids of the testing fold
    stacking_train_logisticReg[min_idx_test:max_idx_test] = classifier.predict(x_train_real[min_idx_test:max_idx_test])
del max_idx_test, min_idx_test, max_idx_train, min_idx_train
    

#Saving 
#pred test
preds = classifier.predict(x_test_real)
print("Writing output...")
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = preds *.75
sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/lgcReg_test_kf.csv", index=False)

#--- pred training for ensemble
print("Writing training pred output...")
sub = pd.DataFrame()
sub['is_duplicate'] = stacking_train_logisticReg *.75
sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/lgcReg_train_kf.csv", index=False)











####################################################################
# KNNs
####################################################################

from sklearn.neighbors import KNeighborsClassifier
KNNs = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

for KNN in KNNs:
    stacking_train_KNN = np.empty(shape=(len(labels_real)))
    for kth, (train_index, test_index) in enumerate(kf.split(x_train_real)):    
        print("\n********************** FOLD ", KNN, " ******************\n")
        max_idx_test  = np.amax(test_index)+1
        min_idx_test  = np.amin(test_index)
        max_idx_train = np.amax(train_index)+1
        min_idx_train = np.amin(train_index)
        
        classifier = KNeighborsClassifier(n_neighbors = KNN, metric = 'minkowski', p = 2, n_jobs = -1)
        classifier.fit(x_train_real[min_idx_train:max_idx_train], labels[min_idx_train:max_idx_train])
    
        #predict training ids of the testing fold
        stacking_train_KNN[min_idx_test:max_idx_test] = classifier.predict(x_train_real[min_idx_test:max_idx_test])
    del max_idx_test, min_idx_test, max_idx_train, min_idx_train
    
    
    #Saving 
    #pred test
    preds = classifier.predict(x_test_real)
    print("Writing output...")
    sub = pd.DataFrame()
    sub['test_id'] = df_test['test_id']
    sub['is_duplicate'] = preds
    sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/KNN_" + KNN + "_extendedMagic2_test_kf.csv", index=False)
    
    #--- pred training for ensemble
    print("Writing training pred output...")
    sub = pd.DataFrame()
    sub['is_duplicate'] = stacking_train_KNN
    sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/KNN_" + KNN + "_extendedMagic2_train_kf.csv", index=False)











####################################################################
# SVM
####################################################################
from sklearn.svm import SVC

stacking_train_SVC = np.empty(shape=(len(labels_real)))
for kth, (train_index, test_index) in enumerate(kf.split(x_train_real)):    
    print("\n********************** FOLD ", kth+1, " ******************\n")
    max_idx_test  = np.amax(test_index)+1
    min_idx_test  = np.amin(test_index)
    max_idx_train = np.amax(train_index)+1
    min_idx_train = np.amin(train_index)
    
    classifier = SVC(kernel = 'rbf')
    classifier.fit(x_train_real[min_idx_train:max_idx_train], labels[min_idx_train:max_idx_train])

    #predict training ids of the testing fold
    stacking_train_SVC[min_idx_test:max_idx_test] = classifier.predict(x_train_real[min_idx_test:max_idx_test])
del max_idx_test, min_idx_test, max_idx_train, min_idx_train


#Saving 
#pred test
preds = classifier.predict(x_test_real)
print("Writing output...")
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = preds
sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/SVC_test_kf.csv", index=False)

#--- pred training for ensemble
print("Writing training pred output...")
sub = pd.DataFrame()
sub['is_duplicate'] = stacking_train_SVC
sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/SVC_train_kf.csv", index=False)










####################################################################
# naive bayes
####################################################################
from sklearn.naive_bayes import GaussianNB

stacking_train_naive_bayes = np.empty(shape=(len(labels_real)))
for kth, (train_index, test_index) in enumerate(kf.split(x_train_real)):    
    print("\n********************** FOLD ", kth+1, " ******************\n")
    max_idx_test  = np.amax(test_index)+1
    min_idx_test  = np.amin(test_index)
    max_idx_train = np.amax(train_index)+1
    min_idx_train = np.amin(train_index)
    
    classifier = GaussianNB()
    classifier.fit(x_train_real[min_idx_train:max_idx_train], labels[min_idx_train:max_idx_train])

    #predict training ids of the testing fold
    stacking_train_naive_bayes[min_idx_test:max_idx_test] = classifier.predict(x_train_real[min_idx_test:max_idx_test])
del max_idx_test, min_idx_test, max_idx_train, min_idx_train


#Saving 
#pred test
preds = classifier.predict(x_test_real)
print("Writing output...")
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = preds
sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/naive_bayes_test_kf.csv", index=False)

#--- pred training for ensemble
print("Writing training pred output...")
sub = pd.DataFrame()
sub['is_duplicate'] = stacking_train_naive_bayes
sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/naive_bayes_train_kf.csv", index=False)











####################################################################
# random forest
####################################################################
print("Random Forest")
from sklearn.ensemble import RandomForestClassifier


stacking_train_RF = np.empty(shape=(len(labels_real)))
for kth, (train_index, test_index) in enumerate(kf.split(x_train_real)):    
    print("\n********************** FOLD ", kth+1, " ******************\n")
    max_idx_test  = np.amax(test_index)+1
    min_idx_test  = np.amin(test_index)
    max_idx_train = np.amax(train_index)+1
    min_idx_train = np.amin(train_index)

    classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', n_jobs = -1)
    classifier.fit(x_train_real[min_idx_train:max_idx_train], labels[min_idx_train:max_idx_train])
    #predict training ids of the testing fold
    stacking_train_RF[min_idx_test:max_idx_test] = classifier.predict(x_train_real[min_idx_test:max_idx_test])


#Saving 
#pred test
preds = classifier.predict(x_test_real)
print("Writing output...")
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = preds
sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/RF_test_kf.csv", index=False)

#--- pred training for ensemble
print("Writing training pred output...")
sub = pd.DataFrame()
sub['is_duplicate'] = stacking_train_RF
sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/RF_train_kf.csv", index=False)


















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

stacking_train_ANN = np.empty(shape=(len(labels_real)))
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
    model_checkpoint = ModelCheckpoint('weightsANN_X_final_Features_extendedMagic2.h5', save_best_only=True, save_weights_only=True, verbose=1)
    
    classifier.compile(optimizer = 'nadam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifier.fit(x_train_real[min_idx_train:max_idx_train], labels[min_idx_train:max_idx_train], batch_size=4096, epochs=300, verbose=1, validation_split=0.1, shuffle=True, callbacks=[early_stopping, model_checkpoint])
    
    classifier.load_weights('weightsANN_X_final_Features_extendedMagic2.h5')
    #predict training ids of the testing fold
    stacking_train_ANN[min_idx_test:max_idx_test] = classifier.predict(x_train_real[min_idx_test:max_idx_test], batch_size=4096, verbose=1)[:,0]


#Saving 
#pred test
preds = classifier.predict(x_test_real, batch_size=4096, verbose=1)[:,0]
print("Writing output...")
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = preds
sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/ANN_test_extendedMagic2_kf.csv", index=False)

#--- pred training for ensemble
print("Writing training pred output...")
sub = pd.DataFrame()
sub['is_duplicate'] = stacking_train_ANN
sub.to_csv(drive + ":/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/layer1Features/ANN_train_extendedMagic2_kf.csv", index=False)



















