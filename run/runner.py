import run.mix76 as m
import utils.dates as dts
import utils.data as dt
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
    'alpha': 10.0
    }

def run(symbol, date, next_date, rnn_model, hmm_model):
    x_test, _, y_test, _ = m.data_for_prediction(symbol, 
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
    amount = 500.0
    limit = 30000.0
    rate = 0.3
    
    df = dt.load_data(params.global_params['db_path'], 'AAPL', to_date=20180323, limit=81, index_col='date')
    dates = df.index.values.tolist()
#     dates = [20180206, 20180207, 20180208, 20180209, 20180212, 20180213, 20180214, 20180215,
#              20180216, 20180220, 20180221, 20180222, 20180223]

    print(type(dates))
    current = dates[:-1]
    next = dates[1:] 
    rnn_model = m.rnn_load_model(config['rnn_model_path'], config['rnn_weights_path'])
    hmm_model = joblib.load(config['hmm_model_path'])
    daily_result = []
    
    symbol_num = len(params.symbols)
    roi = []
    for date, next_date in zip(current, next):
        count = 0
        daily_roi = 0
#         single = min((limit, (amount*rate)/symbol_num))
        single = (min((limit,amount))*rate)/symbol_num
        
        for s in params.symbols:
            out = run(s['symbol'], date, next_date, rnn_model, hmm_model)
#             print(out['symbol'], round(out['accuracy']*100, 2), out['date'], out['pred'], (out['pred'] == out['actual']))
            res = win if( out['pred'] == out['actual']) else loss
            count += res
            daily_roi += single*res
            #print(round(count, 2))
        daily_result.append(round(count, 2)/symbol_num)
        roi.append(daily_roi)
        amount += daily_roi
        print(date, next_date, round(single, 0))
    print('final result', np.sum(daily_result))
    current = dts.int2dates(current)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(current, np.cumsum(daily_result))
    plt.subplot(212)
    plt.plot(current, np.cumsum(roi))
    print('Min', np.min(np.cumsum(roi)))
    print('Max', np.max(np.cumsum(roi)))
    plt.show()