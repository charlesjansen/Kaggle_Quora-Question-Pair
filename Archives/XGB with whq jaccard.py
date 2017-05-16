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

#adding xgb whq javvard features (final x feature)
x = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/x_final_features.csv', header=0) 

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

params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.11
params['max_depth'] = 5
params['silent'] = 1
params['seed'] = RS

print(x.columns)
print(x.describe())

feature_names = list(x.columns.values)
create_feature_map(feature_names)
print("Features: {}".format(feature_names))

x_train = x[:df_train.shape[0]]
x_test  = x[df_train.shape[0]:]
y_train = df_train['is_duplicate'].values
x_train_for_ensemble =  x_train
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


#--- pred training for ensemble

print("pred training for ensemble")
preds = predict_xgb(clr, x_train_for_ensemble)

print("Writing output...")
sub = pd.DataFrame()
sub['is_duplicate'] = preds *.75
sub.to_csv("xgb_seed{}_n{}training.csv".format(RS, ROUNDS), index=False)




