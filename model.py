import numpy as np
import math
import pandas_datareader as web 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from pandas_datareader import data as pdr
import tensorflow as tf

plt.style.use('fivethirtyeight')

#Get Stock Data
yf.pdr_override()

data = pdr.get_data_yahoo('NVDA', start='2012-01-01', end='2022-01-01')
data = data.filter(['Close'])
dataset = data.values

training_data_len = math.ceil(len(dataset) * 0.8)

#Scale Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

#Create Training Data
train_data =scaled_data[:training_data_len, :]

X_train = [] #Features
y_train = [] #Targets

for i in range(60, len(train_data)):
    X_train.append(train_data[i - 60: i, 0])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", 
              loss="mean_squared_error")

model.fit(X_train, y_train, batch_size=1, epochs=2)

#Create Testing Data
test_data = scaled_data[training_data_len - 60: , :]

X_test = []
y_test = dataset[training_data_len: , :]

for i in range(60, len(test_data)):
    X_test.append(test_data[i - 60: i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
print(rmse)

#Plot model predictions and metrics

train = data[:training_data_len]
valid = data[training_data_len: ]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price USD($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc="lower right")
plt.show()




