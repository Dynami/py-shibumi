import numpy as np
import utils.data as dt
import utils.dates as dts
import params
from keras.models import Input, Model
from keras.layers import ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Add, Flatten, AveragePooling2D, Dense
from keras.layers.convolutional import Conv2D
from keras.initializers import glorot_uniform
from keras.utils.vis_utils import plot_model

np.random.seed(123)

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', padding='valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###
    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X

def ShibumiConvNet(input_shape = (20, 4, 1), classes = 2):
    X_input = Input(input_shape)
    #X = ZeroPadding2D((2, 2))(X_input)
    
    X = Conv2D(20, (1, 3), strides = (1, 1), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)

    # Stage 2
#     X = convolutional_block(X, f=2, filters=[1, 1, 20], stage=2, block='a', s=1)
#     X = identity_block(X, f=1, filters=[1, 1, 128], stage=2, block='b')
#     X = identity_block(X, f=3, filters=[3, 3, 128], stage=2, block='c')
#     X = identity_block(X, f=3, filters=[3, 3, 128], stage=2, block='d')
    
    
#     X = convolutional_block(X, f=3, filters=[2, 2, 256], stage = 3, block='a', s = 1)
#     X = identity_block(X, f=3, filters=[2, 2, 256], stage=3, block='b')
#     X = identity_block(X, f=3, filters=[2, 2, 256], stage=3, block='c')
#     X = identity_block(X, f=3, filters=[2, 2, 256], stage=3, block='d')
    
#     X = convolutional_block(X, f=1, filters=[4, 4, 256], stage = 3, block='a', s = 1)
#     X = identity_block(X, 1, [4, 4, 256], stage=3, block='b')
#     X = identity_block(X, 1, [4, 4, 256], stage=3, block='c')
    
    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    #X = AveragePooling2D(input_shape=(2,2), name='avg_pool')(X)

    # output layer
    X = Flatten()(X)
#     X = Dense(1000, activation='relu')(X)
#     X = Dense(100, activation='relu')(X)    
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ShibumiConvNet')

    return model

def normalize(data, ref_data=None, look_back=20, look_ahead=1, alpha=3.0):
    x_data = []
    y_data = []
    if ref_data is None:
        ref_data = data
    
    for index in range(look_back, data.shape[0]):
        x_data.append((data[index-look_back:index]/ref_data[index-look_back])-1)
        y_data.append((data[index-1+look_ahead]/ref_data[index-look_back])-1)
    
    x_data = np.array(x_data)*alpha
    y_data = np.reshape(np.array(y_data)*alpha, (len(y_data), 1))
    
    return x_data, y_data

def denormalize(y_data, data, alpha=3.0):
    out = []
    for index in range(0, y_data.shape[0]):
        t = data[index] * (y_data[index]/alpha) + data[index]
        out.append(t)
    return np.reshape(np.array(out), (-1))


symbol = 'JPM'
look_back = 20
look_ahead = 1
train_size = 0.3

''' Hyper parameters '''
epochs = 5
validation_split=0.1 # part of the training set
batch_size = 32
alpha = 3.0

''' Loading data '''
df = dt.load_data(params.global_params['db_path'], symbol, index_col='date')
data = df[['open', 'high', 'low', 'close']].values
train_rows = int(data.shape[0]*train_size)

''' preparing data '''
train_data = data[:train_rows]
test_data = data[train_rows:]
test_dates = df.index.values[train_rows+look_back:]

train_data = np.reshape(train_data, (train_data.shape[1], train_data.shape[0]))
test_data = np.reshape(test_data, (test_data.shape[1], test_data.shape[0]))

print('train_data.shape', train_data.shape)
x_close_train, y_train = normalize(train_data[3], ref_data=train_data[3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)
x_open_train, _ = normalize(train_data[0], ref_data=train_data[3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)
x_high_train, _ = normalize(train_data[1], ref_data=train_data[3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)
x_low_train, _ = normalize(train_data[2], ref_data=train_data[3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)

y_train = y_train[1:]-y_train[:-1]
y_train = np.where(y_train > 0, 1, 0)
y_train = np.concatenate((np.array([[0]]), y_train), axis=0)

y_train = dt.onehottify(y_train, n=2)
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[2]))

x_train_data = np.stack((x_open_train, x_high_train, x_low_train, x_close_train), axis=2)
x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], x_train_data.shape[2], 1))

x_close_test, y_test = normalize(test_data[3], ref_data=test_data[3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)
x_open_test, _ = normalize(test_data[0], ref_data=test_data[3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)
x_high_test, _ = normalize(test_data[1], ref_data=test_data[3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)
x_low_test, _ = normalize(test_data[2], ref_data=test_data[3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)

x_test_data = np.stack((x_open_test, x_high_test, x_low_test, x_close_test), axis=2)
x_test_data = np.reshape(x_test_data, (x_test_data.shape[0], x_test_data.shape[1], x_test_data.shape[2], 1))

y_test = y_test[1:]-y_test[:-1]
y_test = np.where(y_test > 0, 1, 0)
y_test = np.concatenate((np.array([[0]]), y_test), axis=0)
y_test = dt.onehottify(y_test, n=2)
y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[2]))
print('x_train.shape', x_train_data.shape)
print('y_train.shape', y_train.shape)
print('x_test.shape', x_test_data.shape)
print('y_test.shape', y_test.shape)

model = ShibumiConvNet(input_shape = (look_back, 4, 1), classes = 2)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train_data, y_train, epochs = epochs, batch_size = batch_size)

preds = model.evaluate(x_test_data, y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

print(model.summary())
#plot_model(model, to_file='model.png')
#SVG(model_to_dot(model).create(prog='dot', format='svg'))
exit()
