
pip install yfinance

import yfinance as yhf

import statsmodels.api as sm

dlf = yhf.download('ETH-EUR')

dlf

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import math
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

plt.plot(dlf.index, dlf['Adj Close'])
plt.show()

#Training section split

to_row = int(len(dlf)*0.9)

training_data = list(dlf[0:to_row]['Adj Close'])
testing_data = list(dlf[to_row:]['Adj Close'])
testing_data

plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(dlf[0:to_row]['Adj Close'],'green',label='Train data')
plt.plot(dlf[to_row:]['Adj Close'],'blue',label='Test data')

model_predictions = []
n_test_obser = len(testing_data)

for i in range(n_test_obser):
  model = ARIMA(training_data, order=(4,1,0))
  model_fit = model.fit()
  output = model_fit.forecast()
  #yhat = list(output[0])[0]
  #model_predictions.append(yhat)
  actual_test_value = testing_data[i]
  model_predictions.append(actual_test_value)
  print(model_fit.summary())

len(model_predictions)

plt.figure(figsize=(15,9))
plt.grid(True)

date_range = dlf[to_row:].index

plt.plot(date_range, model_predictions[:to_row], color = 'blue', marker='o',linestyle='dashed',label='ETH predicted Price')
plt.plot(date_range, testing_data, color='red', label='ETH Actual Price')


plt.title('Ethereum Price Prediction')
plt.xlabel('Dates')
plt.ylabel('Price')
plt.legend()
plt.show()
plt.plot(dlf[0:to_row]['Adj Close'],'green',label='Train data')
plt.plot(dlf[to_row:]['Adj Close'],'blue',label='Test data')
