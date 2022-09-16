import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import date
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

today = date.today()
url = 'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=EUR&to_symbol=USD&apikey=267A9RJE3K2QXJE9'
r = requests.get(url)
data = r.json()
data = pd.DataFrame(data=data)

data_new = data[6:]
data_new = data_new.drop(columns='Meta Data')

data_new1 = data_new['Time Series FX (Daily)'].apply(pd.Series)
data_new1['Date'] = pd.bdate_range(end=today, periods=len(data_new1), freq='B')[::-1]

result = data_new1.drop(data_new1.columns[[0, 1, 2]], axis=1)
result = result.iloc[::-1]

price = result.iloc[:,0].astype(float)
dates = result.iloc[:,1]


fig, ax=plt.subplots()
price_tick_spacing = 0.1
date_tick_spacing = 10
ax.plot(dates, price)
ax.xaxis.set_major_locator(ticker.MultipleLocator(date_tick_spacing))
plt.ylabel("EURUSD")
plt.xlabel("Date")
plt.show()

result['MA3'] = result.iloc[:,0].shift(1).rolling(window=3).mean()
result['MA5'] = result.iloc[:,0].shift(1).rolling(window=5).mean()
result['MA10'] = result.iloc[:,0].shift(1).rolling(window=10).mean()
result = result.dropna()
x = result[['MA3','MA5', 'MA10']]
x.head()

y = result.iloc[:,0].astype(float)
y.head()

t = 0.7
t = int(t*len(result))

#Training Data
x_train = x[:t]
y_train = y[:t]

#Testing Data
x_test = x[t:]
y_test = y[t:]

linear = LinearRegression().fit(x_train,y_train)

predicted_price = linear.predict(x_test)
predicted_price = pd.DataFrame(predicted_price, index=y_test.index, columns = ['price'])
r_sq = linear.score(x[t:],y[t:])*100
print("Model Accuracy of Linear Regression: %.3f" % r_sq)
predicted_price.plot(figsize=(10,5))
y_test.plot()
plt.legend(['predicted_price', 'actual_price'])
plt.ylabel('EURUSD')
plt.show()

x_train_gbm, x_test_gbm, y_train_gbm, y_test_gbm = train_test_split(x, y, random_state=42, test_size=0.3)
sc = StandardScaler()
x_train_std = sc.fit_transform(x_train_gbm)
x_test_std = sc.transform(x_test_gbm)
gbr_params = {'n_estimators': 1000, 'max_depth': 3,'min_samples_split': 5,'learning_rate': 0.01,'loss': 'ls'}
gbr = GradientBoostingRegressor(**gbr_params)
gbr.fit(x_train_std, y_train_gbm)
print("Model Accuracy of Gradient Boosting Regression: %.3f" % gbr.score(x_test_std, y_test_gbm))
mse = mean_squared_error(y_test_gbm, gbr.predict(x_test_std))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

test_score = np.zeros((gbr_params['n_estimators'],), dtype=np.float64)
for i, y_pred in enumerate(gbr.staged_predict(x_test_std)):
    test_score[i] = gbr.loss_(y_test_gbm, y_pred)

fig = plt.figure(figsize=(8, 8))
plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(gbr_params['n_estimators']) + 1, gbr.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(gbr_params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
fig.tight_layout()
plt.show()

params = {'n_estimators': 3,'max_depth': 3,'learning_rate': 1,'criterion': 'mse'}
gbm = GradientBoostingRegressor(**params)
gbm.fit(x_train,y_train)
plt.figure(figsize=(12,6))
# plt.scatter(x_train, y_train)
plt.plot(x_test, gbm.predict(x_test), color='black')
plt.show()


