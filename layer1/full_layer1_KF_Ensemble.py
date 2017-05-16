# -*- coding: utf-8 -*-
'''
# Ensemble learning creation:
# base_algorithms = [logistic_regression, decision_tree_classification, ...] #for classification
# 
# stacking_train_dataset = matrix(row_length=len(target), column_length=len(algorithms))
# stacking_test_dataset = matrix(row_length=len(test), column_length=len(algorithms))
# 
# for i,base_algorithm in enumerate(base_algorithms):
#     for trainix, testix in split(train, k=10): #you may use sklearn.cross_validation.KFold of sklearn library
#         stacking_train_dataset[testix,i] = base_algorithm.fit(train[trainix], target[trainix]).predict(train[testix])
#     stacking_test_dataset[,i] = base_algorithm.fit(train).predict(test)
# 
# 
# final_predictions = combiner_algorithm.fit(stacking_train_dataset, target).predict(stacking_test_dataset)
'''

import numpy as np
import pandas as pd
import xgboost as xgb
import operator
from sklearn.model_selection import train_test_split
from collections import Counter
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from pylab import plot, show, subplot, specgram, imshow, savefig

print("Started")
RS = 12357
np.random.seed(RS)

input_folder = 'F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/'
df_train = pd.read_csv(input_folder + 'train.csv')
df_test  = pd.read_csv(input_folder + 'test.csv')

#adding final x features
x = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/x_final_features.csv', header=0) 

print(x.columns)
print(x.describe())

feature_names = list(x.columns.values)
print("Features: {}".format(feature_names))

x_train = x[:df_train.shape[0]]
x_test_real  = x[df_train.shape[0]:]
labels = df_train['is_duplicate'].values
x_train_real =  x_train
labels_real =  labels
del x, df_train


####################################################################
# XGBoost
####################################################################
if 0: # Now we oversample the negative class - on your own risk of overfitting!
	pos_train = x_train[labels == 1]
	neg_train = x_train[labels == 0]

	print("Oversampling started for proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))
	p = 0.165
	scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
	while scale > 1:
		neg_train = pd.concat([neg_train, neg_train])
		scale -=1
	neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
	print("Oversampling done, new proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))

	x_train = pd.concat([pos_train, neg_train])
	labels = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
	del pos_train, neg_train

ratio = float(np.sum(labels == 0)) / np.sum(labels==1)
params = {}
params['scale_pos_weight'] = 0.36#https://www.kaggle.com/c/quora-question-pairs/discussion/31179
params['objective'] = 'binary:logistic'
#params['seed'] = RS
params['eval_metric'] = 'logloss'
params['eta'] = 0.11
params['max_depth'] = 5
params['silent'] = 1 

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



X_training, X_val, labelsing, y_val = train_test_split(x_train, labels, test_size=0.2)#, random_state=RS)
xg_train = xgb.DMatrix(X_training, label=labelsing)
xg_val = xgb.DMatrix(X_val, label=y_val)
watchlist  = [(xg_train,'train'), (xg_val,'eval')]

clr = xgb.train(params, xg_train, ROUNDS, watchlist, early_stopping_rounds=50)

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
sub.to_csv("xgb", index=False)


#--- pred training for ensemble
print("pred training for ensemble")
preds = clr.predict(xgb.DMatrix(x_train_real))
print("Writing output...")
sub = pd.DataFrame()
sub['is_duplicate'] = preds *.75
sub.to_csv("xgb_seed{}_n{}training.csv".format(RS, ROUNDS), index=False)





####################################################################
# ANN
####################################################################
# Importing the Keras libraries and packages
#==============================================================================
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, PReLU
# from keras.layers import Dropout
# from keras.layers.normalization import BatchNormalization
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# 
# 
# # Initialising the ANN
# classifier = Sequential()
# # Adding the input layer and the first hidden layer
# classifier.add(Dense(units = 200, kernel_initializer = 'truncated_normal', activation='relu', input_dim = 71))
# classifier.add(Dropout(rate = 0.1))
# 
# classifier.add(Dense(200, kernel_initializer = 'truncated_normal', activation='relu'))
# classifier.add(Dropout(rate = 0.1))
# 
# classifier.add(Dense(200, kernel_initializer = 'truncated_normal', activation='relu'))
# classifier.add(Dropout(rate = 0.1))
# 
# classifier.add(Dense(units = 1, kernel_initializer = 'truncated_normal', activation = 'sigmoid'))
# 
# early_stopping =EarlyStopping(monitor='val_loss', patience = 10)
# model_checkpoint = ModelCheckpoint('weightsANN_X_final_Features.h5', save_best_only=True, save_weights_only=True, verbose=1)
# 
# classifier.compile(optimizer = 'nadam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# classifier.fit(x_train_real.values, np.array(labels_real), batch_size=2048, epochs=300, verbose=1, validation_split=0.1, shuffle=True, callbacks=[early_stopping, model_checkpoint])
# 
# classifier.load_weights('weightsANN_X_final_Features.h5')
# 
# y_pred = classifier.predict(x_test_real.values, batch_size=2048, verbose=1)
# submissionTest = pd.DataFrame({'is_duplicate':y_pred.ravel()})
# submissionTest.to_csv('denseANN_Test.csv', index=False)
# 
# y_pred_training = classifier.predict(x_train_real.values, batch_size=2048, verbose=1)
# submissionTraining = pd.DataFrame({'is_duplicate':y_pred_training.ravel()})
# submissionTraining.to_csv('denseANN_Training.csv', index=False)
#==============================================================================


####################################################################
# K-nn
####################################################################















