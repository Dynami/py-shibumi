import numpy as np
import utils.data as dt
import utils.dates as dts
import params
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, GRU, Input, Reshape
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import os.path
from sklearn.externals import joblib

np.random.seed(123)
''' Model management'''
save_model = True
rnn_model_file = './models/model_mix76.h5'
rnn_weights_file = './models/model_mix76_weights.h5'
hmm_model_file = './models/hmm_model.pkl'

''' Input parameters '''
symbol = '^GSPC'
look_back = 15 #15
look_ahead = 1
train_size = 0.8
randomize_data = False
hmm_components = 3
activation_method = 'tanh'
l_size = 60# x_train.shape[1]

''' Hyper parameters '''
epochs = 5
validation_split=0.1 # part of the training set
batch_size = 64
alpha = 3.0
dropout_rate = 0.5

''' Mode parameters '''

''' Loading data '''
df = dt.load_data(params.global_params['db_path'], symbol, index_col='date')
df = df[['open', 'high', 'low', 'close', 'volume']]

''' Preparing data - Inline data as input parameters '''
_log_returns = df['close'].values
_log_returns = np.diff(np.log(_log_returns), n=look_ahead)
_log_returns = np.insert(_log_returns, 0, np.zeros((look_ahead)), axis=0)

# rolling = df['close'].rolling(look_back)
# _mean = rolling.mean()
# _mean = dt.min_max_normalize(_mean, method='tanh')
# _mean[0:look_back-look_ahead] = 0.0
# 
# _std = rolling.std()
# _std = dt.min_max_normalize(_std, method='tanh')
# _std[0:look_back-look_ahead] = 0.0

data = df.values
train_rows = int(data.shape[0]*train_size)
test_dates = df.index.values[train_rows+look_back:]

data = np.insert(data, data.shape[1], _log_returns, axis=1) #5
# data = np.insert(data, data.shape[1], _mean, axis=1) #6
# data = np.insert(data, data.shape[1], _std, axis=1) #7

train_data = data[:train_rows]
test_data = data[train_rows:]

#normalize volumes
train_data[:, 4] = dt.min_max_normalize(train_data[:, 4], method='tanh') #2*(train_data[:, 4]-min_vol)/(max_vol-min_vol)-1
test_data[:, 4] = dt.min_max_normalize(test_data[:, 4], method='tanh') #2*(test_data[:, 4]-min_vol)/(max_vol-min_vol)-1

hmm_input_train = np.column_stack([train_data[:, 5]])
hmm_input_test = np.column_stack([test_data[:, 5]])

if(save_model and os.path.isfile(hmm_model_file)):
    hmm_model = joblib.load(hmm_model_file)
else:
    hmm_model = GaussianHMM(n_components=hmm_components, covariance_type="diag", n_iter=1000).fit(hmm_input_train)
    joblib.dump(hmm_model, hmm_model_file)
    
hmm_train = hmm_model.predict_proba(hmm_input_train)
hmm_test = hmm_model.predict_proba(hmm_input_test)


if(False):
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df['close'].values[train_rows:], label='Close')
    #plt.set_autoscaley_on(True)
    
    ax2 = plt.subplot(2, 1, 2)
    # ax2.plot(test_data[:, 7])
    ax2.plot(hmm_test[:, 0], label='Hidden 0')
    ax2.plot(hmm_test[:, 1], label='Hidden 1')
    ax2.plot(hmm_test[:, 2], label='Hidden 1')
    ax2.set_ylim([0,1])
    
    plt.show()
    exit()
    
# normalize and stack OHLC data for NN
x_close_train, y_train = dt.normalize(train_data[:, 3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)
x_close_test, y_test = dt.normalize(test_data[:, 3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)

x_open_train, _ = dt.normalize(train_data[:, 0], ref_data=train_data[:, 3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)
x_open_test, _ = dt.normalize(test_data[:, 0], ref_data=test_data[:, 3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)

x_high_train, _ = dt.normalize(train_data[:, 1], ref_data=train_data[:, 3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)
x_high_test, _ = dt.normalize(test_data[:, 1], ref_data=test_data[:, 3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)

x_low_train, _ = dt.normalize(train_data[:, 2], ref_data=train_data[:, 3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)
x_low_test, _ = dt.normalize(test_data[:, 2], ref_data=test_data[:, 3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)

x_volume_train, _ = dt.normalize(train_data[:, 4], ref_data=train_data[:, 4], look_back=look_back, look_ahead=look_ahead, alpha=alpha)
x_volume_test, _ = dt.normalize(test_data[:, 4], ref_data=test_data[:, 4], look_back=look_back, look_ahead=look_ahead, alpha=alpha)
 
x_volume_train[np.isnan(x_volume_train)]= 0.0
x_volume_test[np.isnan(x_volume_test)]= 0.0

x_returns_train = dt.stack(train_data[:, 5], look_back=look_back, look_ahead=look_ahead)
x_returns_test = dt.stack(test_data[:, 5], look_back=look_back, look_ahead=look_ahead)

hmm_hidden0_train = dt.stack(hmm_train[:, 0], look_back=look_back, look_ahead=look_ahead)
hmm_hidden1_train = dt.stack(hmm_train[:, 1], look_back=look_back, look_ahead=look_ahead)
hmm_hidden2_train = dt.stack(hmm_train[:, 2], look_back=look_back, look_ahead=look_ahead)

hmm_hidden0_test = dt.stack(hmm_test[:, 0], look_back=look_back, look_ahead=look_ahead)
hmm_hidden1_test = dt.stack(hmm_test[:, 1], look_back=look_back, look_ahead=look_ahead)
hmm_hidden2_test = dt.stack(hmm_test[:, 2], look_back=look_back, look_ahead=look_ahead)

''' Randomize train data '''
if(randomize_data):
    shuffled = list(np.random.permutation(x_close_train.shape[0]))
    x_close_train = x_close_train[shuffled]
    x_open_train = x_open_train[shuffled]
    x_high_train = x_high_train[shuffled]
    x_volume_train = x_volume_train[shuffled]
    x_returns_train = x_returns_train[shuffled]
    hmm_hidden0_train = hmm_hidden0_train[shuffled]
    hmm_hidden1_train = hmm_hidden1_train[shuffled]
#     hmm_hidden2_train = hmm_hidden2_train[shuffled]

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

x_returns_train = np.reshape(x_returns_train, (x_returns_train.shape[0], x_returns_train.shape[1]))
x_returns_test = np.reshape(x_returns_test, (x_returns_test.shape[0], x_returns_test.shape[1]))

hmm_hidden0_train = np.reshape(hmm_hidden0_train, (hmm_hidden0_train.shape[0], hmm_hidden0_train.shape[1]))*2-1
hmm_hidden1_train = np.reshape(hmm_hidden1_train, (hmm_hidden1_train.shape[0], hmm_hidden1_train.shape[1]))*2-1
hmm_hidden2_train = np.reshape(hmm_hidden2_train, (hmm_hidden2_train.shape[0], hmm_hidden2_train.shape[1]))*2-1

hmm_hidden0_test = np.reshape(hmm_hidden0_test, (hmm_hidden0_test.shape[0], hmm_hidden0_test.shape[1]))*2-1
hmm_hidden1_test = np.reshape(hmm_hidden1_test, (hmm_hidden1_test.shape[0], hmm_hidden1_test.shape[1]))*2-1
hmm_hidden2_test = np.reshape(hmm_hidden2_test, (hmm_hidden2_test.shape[0], hmm_hidden2_test.shape[1]))*2-1

x_train = np.hstack((x_open_train, x_high_train, x_low_train, x_returns_train, hmm_hidden0_train, hmm_hidden1_train, hmm_hidden2_train))
x_test = np.hstack((x_open_test, x_high_test, x_low_test, x_returns_test, hmm_hidden0_test, hmm_hidden1_test, hmm_hidden2_test))

dimensions = int(x_train.shape[1]/x_open_train.shape[1])
print('dimensions', dimensions)
# x_train = np.hstack((x_close_train))
# x_test = np.hstack((x_close_test))


''' Build model '''
# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# y_train = np.reshape(y_train, (y_train.shape[0], 1))
# y_test = np.reshape(y_test, (y_test.shape[0], 1))
#   
# model = Sequential()
# model.add(GRU(input_shape=(look_back*dimensions, 1), output_dim=x_train.shape[1], return_sequences=True,  activation='sigmoid', inner_activation='hard_sigmoid'))
# model.add(Dropout(dropout_rate))
# # model.add(GRU(x_train.shape[1], activation='sigmoid', return_sequences=True))
# # model.add(Dropout(dropout_rate))
# # model.add(GRU(x_train.shape[1], activation='sigmoid', return_sequences=True))
# # model.add(Dropout(dropout_rate))
# model.add(GRU(x_train.shape[1], activation='sigmoid'))
# model.add(Dropout(dropout_rate))
# model.add(Activation('linear'))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')

print('x_train.shape', x_train.shape)
# 1°
if(save_model and os.path.isfile(rnn_model_file)):
    model = load_model(rnn_model_file)
else:
    X = Input(shape=(x_train.shape[1], ))
    Y = Dense(l_size, activation=activation_method)(X)
    Y = Dropout(dropout_rate)(Y)
    # Y = Dense(l_size, activation=activation_method)(Y)
    # Y = Dropout(dropout_rate)(Y)
    Y = Reshape((l_size, 1))(Y)
    Y = GRU(l_size, activation=activation_method, return_sequences=True)(Y)
    Y = Dropout(dropout_rate)(Y)
    Y = GRU(l_size, activation=activation_method)(Y)
    Y = Dropout(dropout_rate)(Y)
    Y = Dense(1, activation='linear')(Y)
    model = Model(X, Y)
    model.compile(optimizer='adam', loss='mse')
    
    model.save(rnn_model_file, overwrite=False, include_optimizer=True)
    
print(model.summary())

# model = Sequential()
# model.add(Dense(128, input_shape=(x_train.shape[1], ), activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Activation('linear'))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

''' Train model '''
# rnn_weights_file = './models/model_{0}_weights.h5'.format(symbol)
history= None

if(save_model and os.path.isfile(rnn_weights_file)):
    model.load_weights(rnn_weights_file)
else:
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test) )
    if(save_model):
        model.save_weights(rnn_weights_file)
    
''' Predictions on test set (different from validation set) '''
predictions = model.predict(x_train)
print('predictions.shape', predictions.shape)
p_diff = np.diff(np.reshape(predictions, predictions.shape[0])) # predictions[1:]-predictions[:-1]
a_diff = np.diff(np.reshape(y_train, y_train.shape[0])) # y_train[1:]-y_train[:-1]

res = p_diff * a_diff
res = np.where(res > 0., 1., 0.)
cum = np.sum(res)
my_accuracy = cum/res.shape[0]

print('train accuracy', my_accuracy)

predictions = model.predict(x_test)

p_diff = np.diff(np.reshape(predictions, predictions.shape[0])) # predictions[1:]-predictions[:-1]
a_diff = np.diff(np.reshape(y_test, y_test.shape[0])) # y_train[1:]-y_train[:-1]

res = p_diff * a_diff
res = np.where(res > 0., 1., 0.)
cum = np.sum(res)

my_accuracy = cum/res.shape[0]

res = np.where(res == 0.0, -1.0, 0.85)
res = np.cumsum(res)

plt.plot(res)
plt.show()

print('test accuracy', my_accuracy)

# y_test = dt.denormalize(y_test, test_data[:, 3], look_back, look_ahead, alpha)
# predictions = dt.denormalize(predictions, test_data[:, 3], look_back, look_ahead, alpha)

test_dates = dts.int2dates(test_dates)

limit = min((200, len(y_test)))
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
