import numpy as np
import utils.data as dt
import params
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import utils.technical_analysis as ta

np.random.seed(123)

''' Input parameters '''
symbol = 'C'
rsi_period = 2
slow_sma_period = 20
fast_sma_period = 18

look_back = np.max((rsi_period, slow_sma_period))
look_ahead = 1
train_size = 0.95
input_fields = [0, 1, 2] # open, high, low, close, volume
input_ref_field = 1 #close
output_fields = [3] # returns

''' Hyper parameters '''
epochs = 50
validation_split =0.05 # part of the training set

''' Loading data '''
df = dt.load_data(params.global_params['db_path'], symbol, index_col='date')
df = df[['open', 'high', 'low', 'close', 'volume']]

s_w = df['close'].rolling(center=False, window=slow_sma_period)
s_mean = s_w.mean()
s_std = s_w.std()

f_w = df['close'].rolling(center=False, window=fast_sma_period)
f_mean = f_w.mean()
f_std = f_w.std()

df = df.join(pd.Series((f_mean-s_mean)/s_std , name='MA'))

df = ta.myRSI(df, rsi_period)
df = ta.myRSI(df, slow_sma_period) 
df['RSI_{}'.format(rsi_period)] = dt.mapping_value(df['RSI_{}'.format(rsi_period)])
df['RSI_{}'.format(slow_sma_period)] = dt.mapping_value(df['RSI_{}'.format(slow_sma_period)])
df['MA'] = dt.mapping_value(df['MA'], old_range=(-0.5, 0.5))
#df = ta.MA(df, slow_sma_period)

''' Preparing data - Inline data as input parameters '''
df = df.join(pd.Series(df['close'].pct_change(1).shift(-1), name='returns'))
df['returns'] = df['returns'].fillna(0)
#threshold = 0.005
df['returns'] = np.where( df['returns'].values >= 0, 0, 1) # 0 upward# 1 downward # 2 flat

df = df[look_back: ].iloc[:, [5, 6, 7, 8]]
print(df.tail(100))
#exit()
#print(df.tail(10))
data = df.values
#dates = df.index.values[:]
x_data = []
y_data = []

for index in range(look_back, data.shape[0]-1):
    x_data.append(np.reshape(data[index-look_back:index, input_fields], (look_back*len(input_fields), 1)) )
    y_data.append(np.reshape(data[index, output_fields], (len(output_fields), 1)))

x_data = np.array(x_data)
y_data = np.array(y_data)

train_rows = int(round(x_data.shape[0] * train_size))

x_close_train = x_data[:train_rows]
y_train = y_data[:train_rows]

shuffled = list(np.random.permutation(x_close_train.shape[0]))
x_close_train = x_close_train[shuffled]
y_train = y_train[shuffled]
# y_train_dates = dates[:train_rows]

x_close_test = x_data[train_rows:]
y_test = y_data[train_rows:]
# y_test_dates = dates[train_rows:]
y_train = y_train.astype(int)
y_train = np.reshape(y_train, (y_train.shape[0]))
y_train = dt.onehottify(y_train, dtype=float)

y_test = y_test.astype(int)
y_test = np.reshape(y_test, (y_test.shape[0]))
y_test = dt.onehottify(y_test, dtype=float)

x_close_train = np.reshape(x_close_train, (x_close_train.shape[0], x_close_train.shape[1]))
x_close_test = np.reshape(x_close_test, (x_close_test.shape[0], x_close_test.shape[1]))

print('x_close_train.shape', x_close_train.shape)
print('y_train.shape', y_train.shape)
print('x_close_test.shape', x_close_test.shape)
print('y_test.shape', y_test.shape)

''' Build model '''
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(x_close_train.shape[1], )))
model.add(Dropout(0.3))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

''' Train model '''
history = model.fit(x_close_train, y_train, epochs=epochs, 
                    #callbacks=[utils.plot_learning.plot_learning], 
                    validation_split=validation_split
                    #validation_data=(x_close_test, y_test)
                    )

''' Predictions on test set (different from validation set) '''
predictions = model.predict(x_close_test)

tmp = predictions * y_test
tmp = np.sum(tmp, axis=1)
tmp = np.where(tmp >= 0.5, 1, 0)
accuracy = np.sum(tmp)/len(tmp)

for p, a in zip(predictions, y_test):
    print(p, a)

print(accuracy)

''' Print model output '''
print(history.history.keys())
plt.figure(1)
plt.subplot(211)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
# summarize history for loss
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
