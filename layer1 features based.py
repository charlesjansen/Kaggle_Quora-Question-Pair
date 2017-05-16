# -*- coding: utf-8 -*-
#https://www.kaggle.com/dasolmar/xgb-with-whq-jaccard/code/code

import numpy as np
import pandas as pd
import xgboost as xgb
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
df_train = pd.read_csv(input_folder + 'train.csv')
df_test  = pd.read_csv(input_folder + 'test.csv')

#==============================================================================
# #rnn 1 lstm
# train_rnn1Lstm = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/nn 1 lstm best results/0.2647_lstm_300_200_0.30_0.30Train.csv', header=0) 
# test_rnn1Lstm = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/nn 1 lstm best results/0.2647_lstm_300_200_0.30_0.30.csv', header=0) 
# df_rnn1Lstm = pd.concat([train_rnn1Lstm, test_rnn1Lstm]) 
# 
# #rnn 1 GRU
# train_rnn1GRU = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/nn 1 gru/0.3143_lstm_300_200_0.50_0.50_gru_Train.csv', header=0) 
# test_rnn1GRU = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/nn 1 gru/0.3143_lstm_300_200_0.50_0.50_gru.csv', header=0) 
# df_rnn1GRU = pd.concat([train_rnn1GRU, test_rnn1GRU]) 
#==============================================================================




#adding xgb whq javvard features (final x feature)
x = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/x_final_features.csv', header=0) 


#==============================================================================
# ##rnn 1 lstm
# print("rnn features")
# x['rnn1Lstm'] = df_rnn1Lstm['is_duplicate'] 
# 
# 
# ##rnn 1 gru
# print("rnn gru features")
# x['rnn1GRU'] = df_rnn1GRU['is_duplicate'] 
# 
#==============================================================================




def create_feature_map(features):
	outfile = open('xgb.fmap', 'w')
	i = 0
	for feat in features:
		outfile.write('{0}\t{1}\tq\n'.format(i, feat))
		i = i + 1
	outfile.close()

print(x.columns)
print(x.describe())

feature_names = list(x.columns.values)
create_feature_map(feature_names)
print("Features: {}".format(feature_names))

x_train = x[:df_train.shape[0]]
x_test  = x[df_train.shape[0]:]
y_train = df_train['is_duplicate'].values
x_train_for_ensemble =  x_train
y_train_for_ensemble =  y_train
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

####################################################################
# XGBoost
####################################################################

print("Training data: X_train: {}, Y_train: {}, X_test: {}".format(x_train.shape, len(y_train), x_test.shape))

#training
print("Will train XGB for {} rounds, RandomSeed: {}".format(ROUNDS, RS))
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.11
params['max_depth'] = 5
params['silent'] = 1
params['seed'] = RS
X_training, X_val, y_training, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=RS)
xg_train = xgb.DMatrix(X_training, label=y_training)
xg_val = xgb.DMatrix(X_val, label=y_val)
watchlist  = [(xg_train,'train'), (xg_val,'eval')]
clr = xgb.train(params, xg_train, ROUNDS, watchlist)

#predict
preds = clr.predict(xgb.DMatrix(x_test))

print("Writing output...")
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = preds *.75
sub.to_csv("xgb1_seed{}_n{}.csv".format(RS, ROUNDS), index=False)


print("Features importances...")
importance = clr.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))
ft = pd.DataFrame(importance, columns=['feature', 'fscore'])
ft.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
plt.gcf().savefig('features_importance.png')

print("Done.")


#--- pred training for ensemble

print("pred training for ensemble")
preds = clr.predict(xgb.DMatrix(x_train_for_ensemble))

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
# classifier.fit(x_train_for_ensemble.values, np.array(y_train_for_ensemble), batch_size=2048, epochs=300, verbose=1, validation_split=0.1, shuffle=True, callbacks=[early_stopping, model_checkpoint])
# 
# classifier.load_weights('weightsANN_X_final_Features.h5')
# 
# y_pred = classifier.predict(x_test.values, batch_size=2048, verbose=1)
# submissionTest = pd.DataFrame({'is_duplicate':y_pred.ravel()})
# submissionTest.to_csv('denseANN_Test.csv', index=False)
# 
# y_pred_training = classifier.predict(x_train_for_ensemble.values, batch_size=2048, verbose=1)
# submissionTraining = pd.DataFrame({'is_duplicate':y_pred_training.ravel()})
# submissionTraining.to_csv('denseANN_Training.csv', index=False)
#==============================================================================


####################################################################
# K-nn
####################################################################















