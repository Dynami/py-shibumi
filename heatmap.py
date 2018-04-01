import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
import params
import utils.data as dt

data = pd.DataFrame()
for s in params.symbols:
    symbol = s['symbol']
    df = dt.load_data(params.global_params['db_path'], symbol, to_date=20161231, index_col='date')['close']
    
    data[symbol] = pd.Series(np.cumsum(df.pct_change()), index=df.index)

print(data.head())

sns.heatmap(data.corr(), annot=True)
plt.show()