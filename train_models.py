import numpy as np
import utils.data as dt
import params
import pandas as pd
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import os
    
def train_model(symbol='C', look_back=5, look_ahead=1, train_size=0.95, plot=True):
    np.random.seed(123)
    
    ''' Input parameters '''
    input_fields=[0, 1, 2, 3, 4] # open, high, low, close, volume
    output_fields=[5] # returns
    
    ''' Internal parameters'''
    saved_models_dir = params.global_params['models_dir']
    save_models = bool(params.global_params['save_models'])
    
    ''' Hyper parameters '''
    epochs = 100
    validation_split =0.05 # part of the training set
    
    ''' Loading data '''
    df = dt.load_data(params.global_params['db_path'], symbol, index_col='date')
    
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df = df.join(pd.Series(pd.Series(df['close'].diff(1), name='returns')))
    
    ''' Preparing data '''
    c_w = df['close'].rolling(center=False, window=look_back)
    c_mean = c_w.mean()
    c_std = c_w.std()
    
    df['close'] = (df['close'] - c_mean) / (2*c_std)
    df['open'] = (df['open'] - c_mean) / (2*c_std)
    df['high'] = (df['high'] - c_mean) / (2*c_std)
    df['low'] = (df['low'] - c_mean) / (2*c_std)
    
    df['returns'] = df['returns'].fillna(0)
    df['returns'] = np.where(df['returns'].values > 0 , 0, 1) # 0 upward # 1 downward
    df['returns'] = df['returns'].shift(-look_ahead)
    v_w = df['volume'].rolling(center=False, window=look_back) 
    v_mean = v_w.mean()
    v_std = v_w.std()
    df['volume'] = (df['volume'] - v_mean) / (v_std)
    
    df = df[look_back:]
    
    ''' Inline data as input parameters '''
    data = df.values
    
    x_data = []
    y_data = []
    for index in range(data.shape[0] - look_back):
        x_data.append(np.reshape(data[index:index+look_back, input_fields], (look_back*len(input_fields), 1)))
        y_data.append(np.reshape(data[index, output_fields], (len(output_fields), 1)))
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    train_rows = int(round(x_data.shape[0] * train_size))
    
    x_close_train = x_data[:train_rows]
    y_train = y_data[:train_rows]
    
    x_close_test = x_data[train_rows:]
    y_test = y_data[train_rows:]
    
    y_train = y_train.astype(int)
    y_train = np.reshape(y_train, (y_train.shape[0]))
    y_train = dt.onehottify(y_train, dtype=float)
    
    y_test = y_test.astype(int)
    y_test = np.reshape(y_test, (y_test.shape[0]))
    y_test = dt.onehottify(y_test, dtype=float)
    
    x_close_train = np.reshape(x_close_train, (x_close_train.shape[0], x_close_train.shape[1]))
    x_close_test = np.reshape(x_close_test, (x_close_test.shape[0], x_close_test.shape[1]))
        
    ''' Build model '''
    model_file = saved_models_dir + 'model.' + symbol + '.json'
    
    if(save_models and os.path.exists(model_file)):
        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
    else:
        model = Sequential()
        model.add(Dense(100, activation='relu', input_shape=(x_close_train.shape[1], )))
        model.add(Dropout(0.3))
        model.add(Dense(2, activation='softmax'))
        
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        
        model_json = model.to_json()
        with open(model_file, "w") as json_file:
            json_file.write(model_json)
            json_file.close()
    
    ''' Train model '''
    model_weights = saved_models_dir + 'model.' + symbol + '.h5'
    
    history = None
    if (save_models and os.path.exists(model_weights)):
        model.load_weights(model_weights)
    else:
        history = model.fit(x_close_train, y_train, epochs=epochs, 
                            #callbacks=[utils.plot_learning.plot_learning], 
                            validation_split=validation_split
                            #validation_data=(x_close_test, y_test)
                            )
    
        if (save_models and not os.path.exists(model_weights)):
            model.save_weights(model_weights)
            print("Saved model to disk")
            
    ''' Predictions on test set (different from validation set) '''
    predictions = model.predict(x_close_test)
    tmp = predictions * y_test
    tmp = np.sum(tmp, axis=1)
    tmp = np.where(tmp > 0.5, 1, 0)
    accuracy = np.sum(tmp)/len(tmp)
    print(symbol, accuracy)
    
    ''' Print model output '''
    
    if(plot and not history is None):
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

if __name__=='__main__':
    for s in params.symbols:
        train_model(s['symbol'])
