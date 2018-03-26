import numpy as np
import utils.data as dt
import params

np.random.seed(123)
''' Model management'''
_save_model = True

''' Input parameters '''
symbol = 'AAPL'
look_back = 15 #15
look_ahead = 1
train_size = 0.8
randomize_data = False

''' Hyper parameters '''
epochs = 20
validation_split=0.1 # part of the training set
batch_size = 64
alpha = 3.0

''' Loading data '''
df = dt.load_data(params.global_params['db_path'], symbol, index_col='date')
df = df[['open', 'high', 'low', 'close']]

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

''' Randomize train data '''
if(randomize_data):
    shuffled = list(np.random.permutation(x_close_train.shape[0]))
    x_close_train = x_close_train[shuffled]
    x_open_train = x_open_train[shuffled]
    x_high_train = x_high_train[shuffled]
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

x_train = np.hstack((x_open_train, x_high_train, x_low_train))
x_test = np.hstack((x_open_test, x_high_test, x_low_test))

# x_train = np.hstack((x_close_train))
# x_test = np.hstack((x_close_test))
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

y_train = np.reshape(y_train, (y_train.shape[0], 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1))
print('x_train.shape', x_train.shape)
print('x_test.shape', x_test.shape)
