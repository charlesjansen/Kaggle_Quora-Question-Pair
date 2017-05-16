
import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords

#rnn 1 lstm
test_rnn1Lstm = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/nn 1 lstm best results/0.2647_lstm_300_200_0.30_0.30.csv', header=0) 
#rnn 1 GRU
test_rnn1GRU = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/nn 1 gru/0.3143_lstm_300_200_0.50_0.50_gru.csv', header=0)
#xg result
xgTest = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/xgboost magic abhis/xgb_seed12357_n315.csv', header=0)

x=pd.DataFrame()

x = (test_rnn1Lstm + test_rnn1GRU + xgTest)/3



x["test_id"] = pd.read_csv('F:/DS-main/Kaggle-main/Quora Question Pairs - inputs/data/xgboost magic abhis/xgb_seed12357_n315.csv', header=0).test_id.astype(int)
x.to_csv("average.csv", index=False)#0.25386

x.dtypes