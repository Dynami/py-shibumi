import run.mix76 as m
import utils.dates as dts
import numpy as np
import matplotlib.pyplot as plt
import params
from sklearn.externals import joblib

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

def run(symbol, date, next_date, rnn_model, hmm_model):
    x_test, x_test_dates, y_test, dims = m.data_for_prediction(symbol, 
                                         date,
                                         next_date, 
                                         look_back=config['look_back'], 
                                         look_ahead=config['look_ahead'], 
                                         alpha=config['alpha'],
                                         hmm_model=hmm_model)
    
#     print(x_test.shape, x_test_dates.shape, y_test.shape, dims)
    
    predictions = rnn_model.predict(x_test)

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
    #  from  
    win = 0.85
    loss = -1
    
    dates = [20180206, 20180207, 20180208, 20180209, 20180212, 20180213, 20180214, 
             20180215, 20180216, 20180219, 20180220, 20180221, 20180222, 20180223,
             20180226, 20180227, 20180228, 20180301, 20180302, 20180305, 20180306,
             20180307, 20180308, 20180309, 20180312, 20180313, 20180314, 20180315, 20180316  ]
    current = dates[:-1]
    next = dates[1:] 
    rnn_model = m.rnn_load_model(config['rnn_model_path'], config['rnn_weights_path'])
    hmm_model = joblib.load(config['hmm_model_path'])
    symbol_num = len(params.symbols)
    daily_result = []
    for date, next_date in zip(current, next):
        count = 0
        for s in params.symbols:
            out = run(s['symbol'], date, next_date, rnn_model, hmm_model)
            print(out['symbol'], round(out['accuracy']*100, 2), out['date'], out['pred'], (out['pred'] == out['actual']))
            res = win if( out['pred'] == out['actual']) else loss
            count += res
            #print(round(count, 2))
        daily_result.append(round(count, 2)/symbol_num)
        print(date)
    print('final result', np.sum(daily_result))
    current = dts.int2dates(current)
    plt.plot(current, np.cumsum(daily_result))
    plt.show()