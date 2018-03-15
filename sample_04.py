import numpy as np
import utils.data as dt
import utils.dates as dts
import params
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import os.path

np.random.seed(123)
''' Model management'''
_save_model = False

''' Input parameters '''
symbol = 'MSFT'
look_back = 30
look_ahead = 1
train_size = 0.20

''' Hyper parameters '''
epochs = 20
validation_split=0.1 # part of the training set
batch_size = 32
alpha = 3.0

''' Loading data '''
df = dt.load_data(params.global_params['db_path'], symbol, index_col='date')
df = df['close']

''' Preparing data - Inline data as input parameters '''

data = df.values
train_rows = int(data.shape[0]*train_size)

train_data = data[:train_rows]
test_data = data[train_rows:]
test_dates = df.index.values[train_rows+look_back:]

x_close_train, y_train = dt.normalize(train_data, look_back=look_back, look_ahead=look_ahead, alpha=alpha)
x_close_test, y_test = dt.normalize(test_data, look_back=look_back, look_ahead=look_ahead, alpha=alpha)

''' Randomize train data '''
shuffled = list(np.random.permutation(x_close_train.shape[0]))
x_close_train = x_close_train[shuffled]
y_train = y_train[shuffled]

''' Reshape data for model '''
x_close_train = np.reshape(x_close_train, (x_close_train.shape[0], x_close_train.shape[1]))
x_close_test = np.reshape(x_close_test, (x_close_test.shape[0], x_close_test.shape[1]))
y_train = np.reshape(y_train, (y_train.shape[0], 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1))

''' Build model '''
model = Sequential()
model.add(Dense(5000, input_shape=(x_close_train.shape[1], ), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='tanh'))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

''' Train model '''
# file_name = './models/model_{0}_weights.h5'.format(symbol)
file_name = './models/model_MSFT_weights.h5'
history= None

if(_save_model and os.path.isfile(file_name)):
    model.load_weights(file_name)
else:
    history = model.fit(x_close_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(x_close_test, y_test)
                    )
    if(_save_model):
        model.save_weights(file_name)
    
''' Predictions on test set (different from validation set) '''
predictions = model.predict(x_close_train)

p_diff = predictions[1:]-predictions[:-1]
a_diff = y_train[1:]-y_train[:-1]

res = p_diff * a_diff
res = np.where(res > 0., 1., 0.)
cum = np.sum(res)
my_accuracy = cum/res.shape[0]

print('train accuracy', my_accuracy)

predictions = model.predict(x_close_test)

p_diff = predictions[1:]-predictions[:-1]
a_diff = y_test[1:]-y_test[:-1]

res = p_diff * a_diff
res = np.where(res > 0., 1., 0.)
cum = np.sum(res)

my_accuracy = cum/res.shape[0]
print('test accuracy', my_accuracy)


y_test = dt.denormalize(y_test, test_data, look_back, look_ahead, alpha)
predictions = dt.denormalize(predictions, test_data, look_back, look_ahead, alpha)

test_dates = dts.int2dates(test_dates)

#for p, a, d in zip(predictions, y_test, test_dates):
#    print(d, p, a)

plt.plot(test_dates[-100:], y_test[-100:], label='actual')
plt.plot(test_dates[-100:], predictions[-100:], label='predictions')
plt.legend()
plt.show()

''' Print model output '''
if(history is not None):
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
