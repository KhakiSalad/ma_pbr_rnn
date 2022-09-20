import numpy as np
import statsmodels.api as sm
import statsmodels.tools.eval_measures as measures
import itertools
import sys

from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm.notebook import tqdm

import config
import util as u

cols = ["pH", "Temperature_C", "LightIntensity", "TankVolume", "Growth_Rate"]
df = u.load_data(config.dataset_path)
X = df[cols].values
y = df['RelativeDensity'].values
model = sm.tsa.arima.ARIMA(order=(10,1,10), endog=y, exog=X)
model_fit = model.fit()
opt_params = model_fit.params
#%%
df = u.load_data(config.dataset_path_test)
# (T, T_win, X_N)
X = np.zeros((len(df), config["window_size"], len(cols)))
for i, name in enumerate(cols):
    for j in range(config["window_size"]):
        X[:, j, i] = df[name].shift(config["window_size"] - j - 1).fillna(method='bfill')

# (T, T_win, X_N)
y = np.zeros((len(df), config["window_size"], 1))
for j in range(config["window_size"]):
    X[:, j, 0] = df['RelativeDensity'].shift(config["window_size"] - j - 1).fillna(method='bfill')

model = sm.tsa.arima.ARIMA(order=(10,1,10), endog=df['RelativeDensity'][:20], exog=df[cols][:20])
model_fit = model.fit(start_params=opt_params)
opt_params = model_fit.params

y_pred = np.zeros(len(y))
for i in range(20, len(y_pred)):
    #model_fit = model_fit.append(endog=y[i-1], exog=X[i-1])
    model = sm.tsa.arima.ARIMA(order=(10,1,10), endog=df['RelativeDensity'][:i], exog=df[cols][:i])
    model_fit = model.fit(start_params=opt_params)
    y_pred[i] = model_fit.forecast(exog=X[i,0])
    opt_params = model_fit.params

mae = mean_absolute_error(y_pred[20:], df['RelativeDensity'][20:])
mse = mean_squared_error(y_pred[20:], df['RelativeDensity'][20:])
print(f'mae: {mae}')
print(f'mse: {mse}')
#%%
mean_absolute_error(np.repeat(df['RelativeDensity'].mean(), len(df)), df['RelativeDensity'])
plt.plot(y_pred)
plt.plot(df['RelativeDensity'].values)
plt.show()
