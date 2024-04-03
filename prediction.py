!pip install prophet

!pip install fbprophet

!pip install yfinance
!pip install yahoofinancials

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import StandardScaler
import yfinance as yf

df = yf.download('ETH-USD',
   start='2020-10-10',
   end='2024-04-2',
   progress=False)
df.head()

df['Close'].plot(kind='line', figsize=(8, 4), title='Close')
plt.gca().spines[['top', 'right']].set_visible(False)

series = df['Close'].values.reshape(-1, 1)

scaler = StandardScaler()
scaler.fit(series[:len(series) // 2])
series = scaler.transform(series).flatten()

T = 10
D = 1
X = []
Y = []

for t in range(len(series)-T):
  x = series[t:t+T]
  X = np.append(X, x)
  y = series[t+T]
  Y = np.append(Y, y)
  X = np.array(X).reshape(-1, T)
  Y = np.array(Y)
  N = len(X)
  print("X.shape", X.shape, "Y.shape", Y.shape)

class BaselineModel:
  def predict(self, X):
    return X[:,-1] # return the last value for each input sequence

Xtrain, Ytrain = X[:-N//2], Y[:-N//2]
Xtest, Ytest = X[-N//2:], Y[-N//2:]


if len(Ytrain) > 0:
    Ytrain2 = scaler.inverse_transform(Ytrain.reshape(-1, 1)).flatten()
else:
    Ytrain2 = np.array([])

if len(Ytest) > 0:
    Ytest2 = scaler.inverse_transform(Ytest.reshape(-1, 1)).flatten()
else:
    Ytest2 = np.array([])

print("Ytrain2:", Ytrain2)
print("Ytest2:", Ytest2)

from prophet import Prophet

df_prophet = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})

model_prophet = Prophet(daily_seasonality=False, yearly_seasonality=True)
model_prophet.fit(df_prophet)

future_dates = pd.date_range(start=df_prophet['ds'].max(), periods=31, freq='D')[1:]  # Start from the next day

future_df = pd.DataFrame({'ds': future_dates})

forecast_prophet = model_prophet.predict(future_df)

predicted_price_2024 = forecast_prophet.loc[forecast_prophet['ds'] == '2024-04-10', 'yhat'].values

average_daily_return = df_prophet['y'].pct_change().mean()

last_observed_price = df_prophet['y'].iloc[-1]

days_until_march_31 = (pd.Timestamp('2024-04-10') - df_prophet['ds'].max()).days
predicted_price_2024 = last_observed_price * (1 + average_daily_return)**days_until_march_31

print(f"The predicted closing price for the 31st of April 10th is: {predicted_price_2024:.2f}")


import pandas as pd

average_daily_return = df_prophet['y'].pct_change().mean()

last_observed_price = df_prophet['y'].iloc[-1]

start_date = pd.Timestamp('2024-04-05')
end_date = pd.Timestamp('2024-04-10')

predicted_prices = {}
for single_date in pd.date_range(start=start_date, end=end_date):
    days_until_date = (single_date - df_prophet['ds'].max()).days
    predicted_price = last_observed_price * (1 + average_daily_return)**days_until_date
    predicted_prices[single_date.strftime('%m/%d')] = predicted_price

for date, price in predicted_prices.items():
    print(f"The predicted closing price for {date} is: {price:.2f}")

