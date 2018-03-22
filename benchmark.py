import utils.data as dt
import params
import numpy as np
symbol = 'AAPL'
look_back = 20
look_ahead = 1
train_size = 0.8

''' Hyper parameters '''
epochs = 5
validation_split=0.1 # part of the training set
batch_size = 32
alpha = 3.0

''' Loading data '''
df = dt.load_data(params.global_params['db_path'], symbol, index_col='date')
data = df['close'].pct_change()
train_rows = int(data.shape[0]*train_size)

data = data[train_rows:].values

out = data[1:]*data[:-1]
out = np.where(out > 0 , 1, 0)
positives = np.sum(out)
total = out.shape[0]
print((positives/total))