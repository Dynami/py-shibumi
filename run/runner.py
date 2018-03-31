import run.mix76 as m
import utils.dates as dts
import numpy as np
import matplotlib.pyplot as plt
import params

import warnings
warnings.filterwarnings("ignore")

config = {
    'hmm_model_path':'../models/run/hmm_model.pkl',
    'rnn_model_path':'../models/run/model_mix76.h5',
    'rnn_weights_path':'../models/run/model_mix76_weights.h5',
    'look_back': 15,
    'look_ahead': 1,
    'alpha': 3.0
    }

def run(symbol, date, model):
    x_test, x_test_dates, y_test, dims = m.data_for_prediction(symbol, 
                                         date, 
                                         look_back=config['look_back'], 
                                         look_ahead=config['look_ahead'], 
                                         alpha=config['alpha'],
                                         hmm_model_file=config['hmm_model_path'])
    
#     print(x_test.shape, x_test_dates.shape, y_test.shape, dims)
    
    predictions = model.predict(x_test)

    p_diff = np.diff(np.reshape(predictions, predictions.shape[0])) 
    a_diff = np.diff(np.reshape(y_test, y_test.shape[0]))
    
    res = p_diff * a_diff
    res = np.where(res > 0., 1., 0.)
    cum = np.sum(res)

    my_accuracy = cum/res.shape[0]
#     res = np.where(res == 0.0, -1.0, 0.85)
#     res = np.cumsum(res)
#     test_dates = dts.int2dates(x_test_dates[1:])
#     plt.plot(test_dates, res)
#     plt.show()
    
    #print(symbol, 'Acc:', my_accuracy)
    return {'symbol': symbol, 
            'accuracy':my_accuracy, 
            'date':date+1, 
            'pred': (1 if p_diff[-1:] > 0 else -1), 
            'actual': (1 if a_diff[-1:] > 0 else -1)
            }

if __name__ == '__main__':
    #symbol = 'MSFT'
    date = 20180206
    model = m.rnn_load_model(config['rnn_model_path'], config['rnn_weights_path'])
    for s in params.symbols:
        out = run(s['symbol'], date, model)
        print(out['symbol'], round(out['accuracy']*100, 2), out['date'], out['pred'], (out['pred'] == out['actual']))
