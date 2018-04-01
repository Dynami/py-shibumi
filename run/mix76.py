import utils.data as dt
import numpy as np
import params
from keras.models import load_model


def data_for_prediction(symbol, date, next_date, look_back, look_ahead, alpha, 
                        hmm_model,
                        db_path=params.global_params['db_path']
                        ):
    limit = look_back*2
    
    x_df = dt.load_data(db_path, symbol, to_date=date, limit=limit, index_col='date')
    x_df = x_df[['open', 'high', 'low', 'close', 'volume']]
    
    y_df = dt.load_data(db_path, symbol, to_date=next_date, limit=limit, index_col='date')
    y_data = y_df[['close']].values
    
    x_test_dates = x_df.index.values
    ''' Preparing x_data - Inline x_data as input parameters '''
    _log_returns = x_df['close'].values
    _log_returns = np.diff(np.log(_log_returns), n=look_ahead)
    _log_returns = np.insert(_log_returns, 0, np.zeros((look_ahead)), axis=0)
    
    x_data = x_df.values
    
    x_data = np.insert(x_data, x_data.shape[1], _log_returns, axis=1) #5
    x_data[:, 4] = dt.min_max_normalize(x_data[:, 4], method='tanh')
    
    hmm_input_test = np.column_stack([x_data[:, 5]])
    
    hmm_test = hmm_model.predict_proba(hmm_input_test)
    
#     print('hmm_test.shape', hmm_test.shape)
#     print('x_data.shape', x_data.shape)
    
    # normalize and stack OHLC x_data for NN
    x_close_test = dt.normalize_for_future(x_data[:, 3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)
    _, y_test = dt.normalize(y_data, look_back=look_back, look_ahead=look_ahead, alpha=alpha)
#     print('x_close_test.shape', x_close_test.shape)
#     print('y_test.shape', y_test.shape)
    x_open_test = dt.normalize_for_future(x_data[:, 0], ref_data=x_data[:, 3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)
    
    x_high_test = dt.normalize_for_future(x_data[:, 1], ref_data=x_data[:, 3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)
    
    x_low_test = dt.normalize_for_future(x_data[:, 2], ref_data=x_data[:, 3], look_back=look_back, look_ahead=look_ahead, alpha=alpha)
    
    x_volume_test = dt.normalize_for_future(x_data[:, 4], ref_data=x_data[:, 4], look_back=look_back, look_ahead=look_ahead, alpha=alpha)
     
    x_volume_test[np.isnan(x_volume_test)]= 0.0
    
    x_returns_test = dt.stack(x_data[:, 5], look_back=look_back, look_ahead=look_ahead)
    
    hmm_hidden0_test = dt.stack(hmm_test[:, 0], look_back=look_back, look_ahead=look_ahead)
    hmm_hidden1_test = dt.stack(hmm_test[:, 1], look_back=look_back, look_ahead=look_ahead)
    hmm_hidden2_test = dt.stack(hmm_test[:, 2], look_back=look_back, look_ahead=look_ahead)
    
    ''' Reshape x_data for model '''
    x_close_test = np.reshape(x_close_test, (x_close_test.shape[0], x_close_test.shape[1]))
    
    x_open_test = np.reshape(x_open_test, (x_open_test.shape[0], x_open_test.shape[1]))
    
    x_high_test = np.reshape(x_high_test, (x_high_test.shape[0], x_high_test.shape[1]))
    
    x_low_test = np.reshape(x_low_test, (x_low_test.shape[0], x_low_test.shape[1]))
    
    x_volume_test = np.reshape(x_volume_test, (x_volume_test.shape[0], x_volume_test.shape[1]))
    
    x_returns_test = np.reshape(x_returns_test, (x_returns_test.shape[0], x_returns_test.shape[1]))
    
    hmm_hidden0_test = np.reshape(hmm_hidden0_test, (hmm_hidden0_test.shape[0], hmm_hidden0_test.shape[1]))*2-1
    hmm_hidden1_test = np.reshape(hmm_hidden1_test, (hmm_hidden1_test.shape[0], hmm_hidden1_test.shape[1]))*2-1
    hmm_hidden2_test = np.reshape(hmm_hidden2_test, (hmm_hidden2_test.shape[0], hmm_hidden2_test.shape[1]))*2-1
#     print('hmm_hidden0_test.shape', hmm_hidden0_test.shape)
    
    x_test = np.hstack((x_open_test, x_high_test, x_low_test, x_returns_test, hmm_hidden0_test, hmm_hidden1_test, hmm_hidden2_test))

    dimensions = int(x_test.shape[1]/x_open_test.shape[1])
    
    return x_test, x_test_dates[-look_back:], y_test, dimensions


def rnn_load_model(rnn_model_file, rnn_weights_file):
    model = load_model(rnn_model_file)
    model.load_weights(rnn_weights_file)
    
    return model
