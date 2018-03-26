import numpy as np
import utils.data as dt
import utils.dates as dts
import params
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU, Activation
import matplotlib.pyplot as plt
import os.path

# TODO implement look_ahead, implememt volume as feature MinMaxScaler

np.random.seed(123)
''' Model management'''
_save_model = False

''' Input parameters '''
symbol = 'GOOG'
look_back = 15 #15
look_ahead = 1
train_size = 0.8
randomize_data = False

''' Hyper parameters '''
epochs = 5
validation_split=0.1 # part of the training set
batch_size = 64
alpha = 5.0

''' Loading data '''
df = dt.load_data(params.global_params['db_path'], symbol, index_col='date')
df = df[['open', 'high', 'low', 'close', 'volume']]

''' Preparing data - Inline data as input parameters '''
data = df.values
train_rows = int(data.shape[0]*train_size)

train_data = data[:train_rows]
test_data = data[train_rows:]
test_dates = df.index.values[train_rows+look_back:]

x_close_train, y_train = dt.normalize(train_data[:, 3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)
x_close_test, y_test = dt.normalize(test_data[:, 3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)

x_open_train, _ = dt.normalize(train_data[:, 0], ref_data=train_data[:, 3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)
x_open_test, _ = dt.normalize(test_data[:, 0], ref_data=test_data[:, 3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)

x_high_train, _ = dt.normalize(train_data[:, 1], ref_data=train_data[:, 3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)
x_high_test, _ = dt.normalize(test_data[:, 1], ref_data=test_data[:, 3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)

x_low_train, _ = dt.normalize(train_data[:, 2], ref_data=train_data[:, 3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)
x_low_test, _ = dt.normalize(test_data[:, 2], ref_data=test_data[:, 3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)

# Start Volume normalization
max_vol = np.max(train_data[:, 4])
min_vol = np.min(train_data[:, 4])
 
train_data[:, 4] = 2*(train_data[:, 4]-min_vol)/(max_vol-min_vol)-1
test_data[:, 4] = 2*(test_data[:, 4]-min_vol)/(max_vol-min_vol)-1
 
x_volume_train, _ = dt.normalize(train_data[:, 4], ref_data=train_data[:, 4], look_back=look_back, look_ahead=look_ahead, alpha=alpha)
x_volume_test, _ = dt.normalize(test_data[:, 4], ref_data=test_data[:, 4], look_back=look_back, look_ahead=look_ahead, alpha=alpha)
 
x_volume_train[np.isnan(x_volume_train)]= 0.5
x_volume_test[np.isnan(x_volume_test)]= 0.5
# End Volume normalization

''' Randomize train data '''
if(randomize_data):
    shuffled = list(np.random.permutation(x_close_train.shape[0]))
    x_close_train = x_close_train[shuffled]
    x_open_train = x_open_train[shuffled]
    x_high_train = x_high_train[shuffled]
    x_volume_train = x_volume_train[shuffled]

    y_train = y_train[shuffled]

''' Reshape data for model '''
x_close_train = np.reshape(x_close_train, (x_close_train.shape[0], x_close_train.shape[1]))
x_close_test = np.reshape(x_close_test, (x_close_test.shape[0], x_close_test.shape[1]))

x_open_train = np.reshape(x_open_train, (x_open_train.shape[0], x_open_train.shape[1]))
x_open_test = np.reshape(x_open_test, (x_open_test.shape[0], x_open_test.shape[1]))

x_high_train = np.reshape(x_high_train, (x_high_train.shape[0], x_high_train.shape[1]))
x_high_test = np.reshape(x_high_test, (x_high_test.shape[0], x_high_test.shape[1]))

x_low_train = np.reshape(x_low_train, (x_low_train.shape[0], x_low_train.shape[1]))
x_low_test = np.reshape(x_low_test, (x_low_test.shape[0], x_low_test.shape[1]))

x_volume_train = np.reshape(x_volume_train, (x_volume_train.shape[0], x_volume_train.shape[1]))
x_volume_test = np.reshape(x_volume_test, (x_volume_test.shape[0], x_volume_test.shape[1]))

x_train = np.hstack((x_open_train, x_high_train, x_low_train))
x_test = np.hstack((x_open_test, x_high_test, x_low_test))

dimensions = int(x_train.shape[1]/x_open_train.shape[1])
print('dimensions', dimensions)
# x_train = np.hstack((x_close_train))
# x_test = np.hstack((x_close_test))
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

y_train = np.reshape(y_train, (y_train.shape[0], 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1))

''' Build model '''
model = Sequential()
model.add(GRU(input_shape=(look_back*dimensions, 1), output_dim=x_train.shape[1], return_sequences=True,  activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Dropout(0.3))
model.add(GRU(x_train.shape[1], activation='sigmoid', return_sequences=True))
model.add(Dropout(0.3))
model.add(GRU(x_train.shape[1], activation='sigmoid', return_sequences=True))
model.add(Dropout(0.3))
model.add(GRU(x_train.shape[1], activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Activation('linear'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 1°
# model = Sequential()
# model.add(GRU(input_shape=(look_back*4, 1), output_dim=50, return_sequences=True,  activation='sigmoid', inner_activation='hard_sigmoid'))
# model.add(Dropout(0.3))
# model.add(GRU(50, activation='sigmoid', return_sequences=True))
# model.add(Dropout(0.3))
# model.add(GRU(50, activation='sigmoid', return_sequences=True))
# model.add(Dropout(0.3))
# model.add(GRU(50, activation='sigmoid'))
# model.add(Dropout(0.3))
# model.add(Activation('linear'))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')

# 2°
# model = Sequential()
# model.add(GRU(input_shape=(look_back*4, 1), output_dim=128, return_sequences=True,  activation='sigmoid', inner_activation='hard_sigmoid'))
# model.add(Dropout(0.5))
# model.add(GRU(128, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
# model.add(Dropout(0.5))
# model.add(GRU(128, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
# model.add(Dropout(0.5))
# model.add(GRU(128, activation='sigmoid'))
# model.add(Dropout(0.5))
# model.add(Activation('linear'))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
''' Train model '''
# file_name = './models/model_{0}_weights.h5'.format(symbol)
file_name = './models/model_X1_GRU_weights.h5'
history= None

if(_save_model and os.path.isfile(file_name)):
    model.load_weights(file_name)
else:
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test) )
    if(_save_model):
        model.save_weights(file_name)
    
''' Predictions on test set (different from validation set) '''
predictions = model.predict(x_train)

p_diff = predictions[1:]-predictions[:-1]
a_diff = y_train[1:]-y_train[:-1]

res = p_diff * a_diff
res = np.where(res > 0., 1., 0.)
cum = np.sum(res)
my_accuracy = cum/res.shape[0]

print('train accuracy', my_accuracy)

predictions = model.predict(x_test)

p_diff = predictions[1:]-predictions[:-1]
a_diff = y_test[1:]-y_test[:-1]

res = p_diff * a_diff
res = np.where(res > 0., 1., 0.)
cum = np.sum(res)

my_accuracy = cum/res.shape[0]

res = np.where(res == 0.0, -1.0, 0.85)
res = np.cumsum(res)

plt.plot(res)
plt.show()

print('test accuracy', my_accuracy)

y_test = dt.denormalize(y_test, test_data[:, 3], look_back, look_ahead, alpha)
predictions = dt.denormalize(predictions, test_data[:, 3], look_back, look_ahead, alpha)

test_dates = dts.int2dates(test_dates)

#for p, a, d in zip(predictions, y_test, test_dates):
#    print(d, p, a)
limit = min((50, len(y_test)))
plt.plot(test_dates[-limit:], y_test[-limit:], label='actual')
plt.plot(test_dates[-limit:], predictions[-limit:], label='predictions')
plt.legend()
plt.show()

''' Print model output '''
if(history is not None):
    print(history.history.keys())
#     plt.figure(1)
#     plt.subplot(211)
    
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    # summarize history for loss
#     plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
