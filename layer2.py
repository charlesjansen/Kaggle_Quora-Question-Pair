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
rnnGRUTraining = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/nn 1 gru/0.3143_lstm_300_200_0.50_0.50_gru_Train.csv', header=0) 
rnnGRUTest = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/nn 1 gru/0.3143_lstm_300_200_0.50_0.50_gru.csv', header=0) 

#xg result
xgTraining = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/xgboost magic abhis/xgb_seed12357_n315training.csv', header=0) 
xgTest = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/xgboost magic abhis/xgb_seed12357_n315.csv', header=0) 

#adding xgb whq javvard features (final x feature)
#==============================================================================
# x = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/x_final_features.csv', header=0) 
# X = x[:df_train.shape[0]]
# X_test  = x[df_train.shape[0]:]
#==============================================================================

X = pd.DataFrame()
X['rnn'] = rnnGRUTraining.is_duplicate
X['xg'] = xgTraining.is_duplicate

X_test = pd.DataFrame()
X_test['rnn'] = rnnGRUTest.is_duplicate
X_test['xg'] = xgTest.is_duplicate


from scipy.stats.stats import pearsonr 
rnn = X['rnn'].values
xg = X['xg'].values
print(np.corrcoef(rnn,xg))

if 1: # Now we oversample the negative class - on your own risk of overfitting!
	pos_train = X[y == 1]
	neg_train = X[y == 0]

	print("Oversampling started for proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))
	p = 0.165
	scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
	while scale > 1:
		neg_train = pd.concat([neg_train, neg_train])
		scale -=1
	neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
	print("Oversampling done, new proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))

	X = pd.concat([pos_train, neg_train])
	y = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
	del pos_train, neg_train

ROUNDS = 800
RS = 12357
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.01
params['max_depth'] = 5
params['silent'] = 1
params['seed'] = RS

X_training, X_val, y_training, y_val = train_test_split(X, y, test_size=0.2, random_state=RS)

xg_train = xgb.DMatrix(X_training, label=y_training)
xg_val = xgb.DMatrix(X_val, label=y_val)

watchlist  = [(xg_train,'train'), (xg_val,'eval')]
clr = xgb.train(params, xg_train, ROUNDS, watchlist)



preds = clr.predict(xgb.DMatrix(X_test))

print("Writing output...")
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = preds *.75
sub.to_csv("xgb_seed{}_n{}.csv".format(RS, ROUNDS), index=False)


print("Done.")

