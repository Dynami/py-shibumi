import numpy as np
from hmmlearn.hmm import GaussianHMM
import utils.data as dt
import params
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator

np.random.seed(123)
''' Input parameters '''
symbol = 'GOOG'
look_back = 15 #15
look_ahead = 1

''' Loading data '''
df = dt.load_data(params.global_params['db_path'], symbol, from_date=20100101, to_date=20110101, index_col='date')
df = df[['open', 'high', 'low', 'close', 'volume']]

dates = df.index.values
close_v = df['close'].values
volume = df['volume'].values

#_log_returns = np._log_returns(close_v)
_log_returns = np.diff(np.log(close_v))

dates = dates[1:]
close_v = close_v[1:]
volume = volume[1:]
# 
# _log_returns = np.reshape(_log_returns, (1, _log_returns.shape[0]))
# dates = np.reshape(dates, (1, dates.shape[0]))
# close_v = np.reshape(close_v, (1, close_v.shape[0]))

print('_log_returns.shape', _log_returns.shape)
print('dates.shape', dates.shape)
print('close_v.shape', close_v.shape)

# Pack _log_returns and volume for training.
X = np.column_stack([_log_returns, volume])

print("fitting to HMM and decoding ...", end="")


# Make an HMM instance and execute fit
model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000).fit(X)

# Predict the optimal sequence of internal hidden state
hidden_states = model.predict_proba(X)
print('hidden_states.shape', hidden_states.shape)
#exit()
print("done")

print("Transition matrix")
print(model.transmat_)
print()

print("Means and vars of each hidden state")
for i in range(model.n_components):
    print("{0}th hidden state".format(i))
    print("mean = ", model.means_[i])
    print("var = ", np.diag(model.covars_[i]))
    print()


ax1 = plt.subplot(2, 1, 1)
ax1.plot(close_v, label='Close')
#plt.set_autoscaley_on(True)

ax2 = plt.subplot(2, 1, 2)
ax2.plot(hidden_states[:, 0], label='Hidden 0')
ax2.plot(hidden_states[:, 1], label='Hidden 1')
ax2.plot(hidden_states[:, 2], label='Hidden 2')
ax2.set_ylim([0,1])

plt.show()
exit()

fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
colours = ('green', 'yellow', 'red', 'blue') #cm.rainbow(np.linspace(0, 1, model.n_components))
for i, (ax, colour) in enumerate(zip(axs, colours)):
    # Use fancy indexing to plot data in each state.
    mask = hidden_states == i
    ax.plot(hidden_states[:, i], ".-", c=colour)
    ax.set_title("{0}th hidden state".format(i))

    # Format the ticks.
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator())

    ax.grid(True)

plt.show()

