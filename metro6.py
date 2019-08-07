import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

print(os.getcwd())
df = pd.read_csv('../dat/metro.csv')

# data type
df.holiday = df.holiday.astype('category')
df.weather_main = df.weather_main.astype('category')
df.weather_description = df.weather_description.astype('category')
cat_columns = df.select_dtypes(['category']).columns
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
df.date_time = df.date_time.astype('datetime64[ns]')
df = df.astype({'temp':'float16','rain_1h':'float32','snow_1h':'float16','clouds_all':'int8','traffic_volume':'uint16'})

print(df.head())
print(df.shape)

X = df[df.columns[0:7]]
y = df.traffic_volume

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# regress
regressor = AdaBoostRegressor(random_state=0, n_estimators=100)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

print('X_test.shape = ',X_test.shape)
print('y_test.shape = ',y_test.shape)

# result
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
print('mse = ',mse,'; rmse = ',rmse)


