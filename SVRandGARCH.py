# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import arch
import numpy as np
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt


data = pd.read_csv('DEXCHUS.csv')
data = data[data['DEXCHUS'] != '.']
data.DEXCHUS = data.DEXCHUS.astype(float)
data['returns'] = 100 * data.DEXCHUS.pct_change().dropna()
data['log_ret'] = np.log(data.DEXCHUS) - np.log(data.DEXCHUS.shift(1))
data = data[data.log_ret!=0]
data['vol_squared'] = data['log_ret']**2
data['log_vol_squared'] = np.log(data['log_ret']**2)
data = data[1:]
data = data.reset_index(drop=True)


data.DEXCHUS.plot()
data.returns.plot()
data.vol_squared.plot()
'''
returns = data.returns.dropna().as_matrix()
from arch.univariate import arch_model
am = arch_model(returns)
res = am.fit()
res.summary()
fig = res.plot()
'''

# set n as moving window period
n = 50
X = np.zeros((data.shape[0]-n,n))
for i in range(n,data.shape[0]):
    for j in range(n):
        X[i-n][j] = data.log_ret[i-j-1]
y = np.zeros(data.shape[0]-n)
for i in range(data.shape[0]-n):
    y[i] = data.log_vol_squared[i+n]

svr_rbf = SVR(kernel='rbf',C=1e3,gamma=250)
y_rbf = svr_rbf.fit(X,y).predict(X)
predicted_vol_rbf = np.exp(y_rbf)

svr_linear = SVR(kernel='linear',C=1e3,gamma=250)
y_linear = svr_linear.fit(X,y).predict(X)
predicted_vol_linear = np.exp(y_linear)

svr_poly = SVR(kernel = 'poly',C=1e3,gamma=250)
y_poly = svr_poly.fit(X,y).predict(X)
predicted_vol_poly = np.exp(y_poly)

real_vol = data.vol_squared[n:].as_matrix()
xaxis = range(150-n)

plt.style.use('ggplot')
plt.plot(xaxis,real_vol[data.shape[0]-150:],color='r',label='real_vol')
plt.plot(xaxis,predicted_vol_rbf[data.shape[0]-150:],color='b',label='svr_rbf')
plt.plot(xaxis,predicted_vol_linear[data.shape[0]-150:],color='g',label='svr_linear')
plt.plot(xaxis,predicted_vol_poly[data.shape[0]-150:],color='orange',label='svr_poly')
plt.title('SVR Volatility Forecasting')
plt.legend()
plt.show()

def SVR_with_moving_windows(n):
    X = np.zeros((data.shape[0]-n,n))
    for i in range(n,data.shape[0]):
        for j in range(n):
            X[i-n][j] = data.log_ret[i-j-1]
    y = np.zeros(data.shape[0]-n)
    for i in range(data.shape[0]-n):
        y[i] = data.log_vol_squared[i+n]

    svr_rbf = SVR(kernel='rbf',C=1e3,gamma=250)
    y_rbf = svr_rbf.fit(X,y).predict(X)
    predicted_vol_rbf = np.exp(y_rbf)

    real_vol = data.vol_squared[n:].as_matrix()
    xaxis = range(100)
    
    plt.style.use('ggplot')
    plt.plot(xaxis,real_vol[data.shape[0]-100-n:],color='r',label='real_vol')
    plt.plot(xaxis,predicted_vol_rbf[data.shape[0]-100-n:],color='b',label='svr_rbf')
    plt.title('SVR Volatility Forecasting window length: %d days'%(n))
    plt.legend()
    plt.show()

for i in range(10,60,10):
    SVR_with_moving_windows(i)