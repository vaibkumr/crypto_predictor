from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_regression

# regresion_data = pd.read_csv('bitstampUSD.csv')
# time=regresion_data['1315922016']
# price=regresion_data['5.800000000000']

import requests
url = 'https://rest.coinapi.io/v1/ohlcv/BITSTAMP_SPOT_BTC_USD/history?period_id=1DAY&time_start=2011-01-01T00:00:00&limit=100000'
headers = {'X-CoinAPI-Key' : '73034021-0EBC-493D-8A00-E0F138111F41'}
response = requests.get(url, headers=headers)

# plt.figure()
# plt.title("Bitcoin Price")
# plt.plot([1,2,3,43,45,5,5],'g.')
# plt.ylabel('values')
# plt.xlabel('indices')
# plt.show()


# scaler=MinMaxScaler()
# X,y=digs.data,digs.target
# X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
# X_train=scaler.fit_transform(X_train)
# X_test=scaler.fit_transform(X_test)
# scores={}
# for x in [1,10,100]:
#     a=MLPClassifier(hidden_layer_sizes=x,solver='lbfgs',alpha=5,random_state=0).fit(X_train,y_train)
#     #alpha above is for L2 regularization
#     score=a.score(X_test,y_test)
#     scores[x]=score
# for index in scores.keys():
#     print("Score["+str(index)+"]: "+str(scores[index]))
