import sqlite3
import pandas as pd
import numpy as np

def load_data(db_path, symbol, from_date=0, to_date=99991231, limit=None, index_col=None):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql('select * from quotes where symbol = ? and date between ? and ? order by date', 
                     con=conn,
                     params=(symbol, from_date, to_date), 
                     index_col=index_col)
    conn.close()
    if(limit is not None):
        return df.iloc[-limit:, :]
    else:
        return df

def stack(data, look_back=20, look_ahead=1):
    x_data = []
    for index in range(look_back, data.shape[0]+1-look_ahead):
        x_data.append(data[index-look_back:index])
    
    x_data = np.array(x_data)
    return x_data

# def norm_for_future(data, ref_data=None, alpha=3.0):
#     if(ref_data is None):
#         ref_data = data
#     print(data)
#     x_data = (data/data[0])-1
#     
#     x_data = np.array(x_data)*alpha
#     return x_data

def normalize_for_future(data, ref_data=None, look_back=20, look_ahead=1, alpha=3.0):
    x_data = []
    y_data = []
    if(ref_data is None):
        ref_data = data
    
    for index in range(look_back+1, data.shape[0]+1):
        x_data.append((data[index-look_back:index]/ref_data[index-look_back])-1)
#         y_data.append((data[index+look_ahead-1]/ref_data[index-look_back])-1)
    
#     for index in range(look_back, data.shape[0]):
# #         x_data.append((data[index-look_back+1:index+1]/ref_data[index-look_back+1])-1)
#         y_data.append((data[index+look_ahead-1]/ref_data[index-look_back])-1)
    
    x_data = np.array(x_data)*alpha
    y_data = np.reshape(np.array(y_data)*alpha, (len(y_data), 1))
    return x_data#, y_data
    
def normalize(data, ref_data=None, look_back=20, look_ahead=1, alpha=3.0):
    x_data = []
    y_data = []
    if(ref_data is None):
        ref_data = data
    
    for index in range(look_back, data.shape[0]+1-look_ahead):
        x_data.append((data[index-look_back:index]/ref_data[index-look_back])-1)
        y_data.append((data[index+look_ahead-1]/ref_data[index-look_back])-1)
    
    x_data = np.array(x_data)*alpha
    y_data = np.reshape(np.array(y_data)*alpha, (len(y_data), 1))
    return x_data, y_data

def denormalize(y_norm, raw_data, look_back=20, look_ahead=1, alpha=3.0):
    y_norm = np.reshape(y_norm, (-1))
    
    out = (y_norm/alpha)*raw_data[:-look_back]+raw_data[:-look_back]
#     out = []
#     for index in range(0, y_norm.shape[0]):
#         t = raw_data[index] * (y_norm[index]/alpha) + raw_data[index]
#         out.append(t)
#     out = np.array(out)
    return np.reshape(out, (-1))

def min_max_normalize(data, method='sigmoid'):
    max = data.max()
    min = data.min()
    out = (data-min)/(max-min)
    if(method == 'sigmoid'):
        return out
    elif(method == 'tanh'):
        return out*2-1
    else:
        return data

def split_data(x_data, y_data, raw_data, train_size, dates=None, random_state=True):
    train_rows = int(round(x_data.shape[0] * train_size))

    x_close_train = x_data[:train_rows]
    y_train = y_data[:train_rows]
    raw_train =  raw_data[:train_rows]
    
    if(dates is not None):
        dates_train = dates[:train_rows]
        
    if(random_state):
        shuffled = list(np.random.permutation(x_close_train.shape[0]))
        x_close_train = x_close_train[shuffled]
        y_train = y_train[shuffled]
        
        if(dates is not None):
            dates_train = dates_train[shuffled]
    
    x_close_test = x_data[train_rows:]
    y_test = y_data[train_rows:]
    raw_test =  raw_data[train_rows:]
    
    if dates is not None:
        dates_test = dates[train_rows:]
    
    return x_close_train, y_train, x_close_test, y_test, raw_train, raw_test, dates_train, dates_test


def onehottify(x, n=None, dtype=float):
    """1-hot encode x with the max value n (computed from data if n is None)."""
    x = np.asarray(x)
    n = np.max(x) + 1 if n is None else n
    return np.eye(n, dtype=dtype)[x]

def mapping_value(old_value, old_range=(0, 1), new_range=(-1, 1)):
    return ( (old_value - old_range[0]) / (old_range[1] - old_range[0]) ) * \
             (new_range[1] - new_range[0]) + new_range[0]

