#IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import yfinance as yf
import datetime 
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing

# DOWNLOADING DATA FROM API
data = yf.download(tickers='MSFT', period='1mo', interval='1d')
#TEST RUN FOR DATA
data.head(3)

#PLOT DOWNLOADED DATA
today = datetime.date.today()
plt.figure(figsize=(10,6))
plt.title(' MICROSOFT STOCK FOR LAST 1 MONTH AS ON  '+ str(today))
plt.plot(data['Close'])
plt.show()

#DEFINING INPUTS
data = data['Close'].tolist()
#CHANGE DATA FOR 30 DAYS FROM THE DATE WHEN YOU ARE RUNNING THIS CODE
start_date = '2021-01-05'
end_date = '2021-02-01'
index= pd.date_range(start=start_date, end=end_date, freq='B')
stock_data = pd.Series(data, index)
forecast_timestep = 2

#TEST 1:    ALPHA = 0.1
plt.figure(figsize=(10,6))
fit_1 = SimpleExpSmoothing(stock_data, initialization_method="heuristic").fit(smoothing_level=0.1,optimized=False)
forecast1 = fit_1.forecast(forecast_timestep).rename(r'$\alpha=0.1$')

plt.plot(stock_data,  color='black')
plt.plot(fit_1.fittedvalues,  color='cyan')
line1, = plt.plot(forecast1,  color='cyan')
plt.show()

#TEST 2:    ALPHA = 0.0.4
plt.figure(figsize=(10,6))
fit_2 = SimpleExpSmoothing(stock_data, initialization_method="heuristic").fit(smoothing_level=0.4,optimized=False)
forecast2 = fit_2.forecast(forecast_timestep).rename(r'$\alpha=0.4$')

plt.plot(stock_data,  color='black')
plt.plot(fit_2.fittedvalues,  color='red')
line2, = plt.plot(forecast2, color='red')
plt.show()

    #     FINAL RUN.................
fit_1 = SimpleExpSmoothing(stock_data, initialization_method="heuristic").fit(smoothing_level=0.1,optimized=False)
forecast1 = fit_1.forecast(forecast_timestep).rename(r'$\alpha=0.1$')

fit_2 = SimpleExpSmoothing(stock_data, initialization_method="heuristic").fit(smoothing_level=0.4,optimized=False)
forecast2 = fit_2.forecast(forecast_timestep).rename(r'$\alpha=0.4$')

fit_3 = SimpleExpSmoothing(stock_data, initialization_method="heuristic").fit(smoothing_level=0.6,optimized=False)
forecast3 = fit_3.forecast(forecast_timestep).rename(r'$\alpha=0.8$')

fit_4 = SimpleExpSmoothing(stock_data, initialization_method="estimated").fit()
forecast4 = fit_4.forecast(forecast_timestep).rename(r'$\alpha=%s$'%fit_4.model.params['smoothing_level'])

plt.figure(figsize=(16,10))
plt.plot(stock_data,  color='black')
plt.plot(fit_1.fittedvalues,  color='cyan')
line1, = plt.plot(forecast1,  color='cyan')
plt.plot(fit_2.fittedvalues,  color='red')
line2, = plt.plot(forecast2, color='red')
plt.plot(fit_3.fittedvalues,  color='green')
line3, = plt.plot(forecast3, color='green')
plt.plot(fit_4.fittedvalues,  color='blue')
line4, = plt.plot(forecast4, color='blue')
plt.legend([line1, line2, line3,line4], [forecast1.name, forecast2.name, forecast3.name,forecast4.name])
plt.show()
