import numpy as np 
import pandas as pd 
from sklearn .linear_model import LinearRegression
from sklearn.svm import  SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('SBIN.NS.csv')
df = df[[ 'Adj Close' ]]
df = df.apply (pd.to_numeric, errors='coerce')
df = df.dropna()
print(df.head(5))
forecast_out = 50
df['prediction'] = df[['Adj Close' ]].shift(-forecast_out)
#---------------------------------------------------------------------------------------------
x = np.array(df.drop(['prediction'], axis = 1))
x = x[:-forecast_out]
y = np.array(df['prediction'])
y = y[:-forecast_out]
x_train, x_test,y_train, y_test = train_test_split(x, y, test_size = 0.15)
#---------------------------------------------------------------------------------------------
svr_rbf = SVR(kernel='rbf', C =1e3 , gamma=0.1)
svr_rbf.fit(x_train, y_train)
svm_confidence = svr_rbf.score(x_test, y_test)
print('svm_confidence:', svm_confidence) 
#--------------------------------------------------------------------------------------------------
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_confidence = lr.score(x_test, y_test)
print('lr_confidence:', lr_confidence)
x_forecast = np.array(df.drop(['prediction'],1))[-forecast_out:]
lr_prediction = lr.predict(x_forecast) 
print(lr_prediction)
svm_prediction = svr_rbf.predict(x_forecast)
print(svm_prediction)

x = lr_prediction
y = range(forecast_out)
plt.plot(x,y)
plt.xlabel('lr_prediction')
plt.ylabel('no of days')
plt.title('SBIN')
plt.show()
