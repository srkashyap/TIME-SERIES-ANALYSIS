
import numpy as np
import pandas as pd
import yfinance as yf
import datetime 
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing,Holt, ExponentialSmoothing


# DOWNLOADING DATA FROM API
data = yf.download(tickers='TSLA', period='10mo', interval='1d')
#TEST RUN FOR DATA
# data.head(3)
# #PLOT DOWNLOADED DATA
today = datetime.date.today()
plt.figure(figsize=(10,6))
plt.title(' MICROSOFT STOCK FOR LAST 10 MONTH AS ON  '+ str(today))
plt.plot(data['Close'])
plt.show()

#DEFINING INPUTS
data = data['Close'].tolist()
#CHANGE DATA FOR 30 DAYS FROM THE DATE WHEN YOU ARE RUNNING THIS CODE
start_date = '2020-04-16'
end_date = '2021-02-03'
index= pd.date_range(start=start_date, end=end_date, freq='B')
stock_data = pd.Series(data, index)
forecast_timestep = 2


get_ipython().magic('matplotlib inline')
fit1 = ExponentialSmoothing(stock_data, seasonal_periods=4, trend='add', seasonal='add', use_boxcox=True, initialization_method="estimated").fit()
fit2 = ExponentialSmoothing(stock_data, seasonal_periods=4, trend='add', seasonal='mul', use_boxcox=True, initialization_method="estimated").fit()
fit3 = ExponentialSmoothing(stock_data, seasonal_periods=4, trend='add', seasonal='add', damped_trend=True, use_boxcox=True, initialization_method="estimated").fit()
fit4 = ExponentialSmoothing(stock_data, seasonal_periods=4, trend='add', seasonal='mul', damped_trend=True, use_boxcox=True, initialization_method="estimated").fit()

ax = stock_data.plot(figsize=(16,10), color='black', title="Forecasts Without Damping factor" )
ax.set_ylabel("Prices $")
ax.set_xlabel("Date")
fit1.fittedvalues.plot(ax=ax, style='--', color='red')
fit2.fittedvalues.plot(ax=ax, style='--', color='green')
fit1.forecast(2).rename('Holt-Winters (add-seasonal)').plot(ax=ax, style='--', color='red', legend=True)
fit2.forecast(2).rename('Holt-Winters (mul-seasonal)').plot(ax=ax, style='--', color='green', legend=True)

ax = stock_data.plot(figsize=(16,10), color='black', title="Forecasts with Damping Factor" )
ax.set_ylabel("Prices $ ")
ax.set_xlabel("Year")
fit3.fittedvalues.plot(ax=ax, style='--', color='red')
fit4.fittedvalues.plot(ax=ax, style='--', color='green')
fit3.forecast(2).rename('Holt-Winters (add-seasonal)').plot(ax=ax, style='--', color='red', legend=True)
fit4.forecast(2).rename('Holt-Winters (mul-seasonal)').plot(ax=ax, style='--', color='green', legend=True)
plt.show()
