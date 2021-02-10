import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
file = "../../../PYTHON/ES.VOLUMEBARS.100.RSI.BOLLINGER.1completo.csv"
data = open(file, "r")
cols = data.readline().strip().split(",")
# print(cols)
counter = 0
dictionary = {}
for column in cols:
    dictionary[column] = []
for line in data:
    values = line.strip().split(",")
    for i in range(len(cols)):
        dictionary[cols[i]].append(values[i])
    counter += 1
# print(" la base de datos tiene %d filas y %d columnas" % (counter, len(cols)))
# print(dictionary['Close'])
dictionary = pd.DataFrame.from_dict(dictionary)
# dictionary = dictionary.set_index('Date')
dictionary.replace('null', np.nan, inplace=True)
dictionary.dropna(inplace=True, how='any')
print(dictionary)
def bollingerbands(dictionary, stock_price, window_size, width):
    print(stock_price.rolling(window=window_size, win_type=None, min_periods=1))
    dictionary['rollin_mean'] = stock_price.rolling(window=window_size, win_type=None, min_periods=1).mean()
    rolling_standev = stock_price.rolling(window=window_size, win_type=None, min_periods=1).std()
    dictionary['upper_band'] = dictionary['rollin_mean'] + (rolling_standev * width)
    dictionary['lower_band'] = dictionary['rollin_mean'] - (rolling_standev * width)
    dictionary.dropna(inplace=True, how='any')
    dictionary[['rollin_mean', 'upper_band', 'lower_band', 'Series.Close']] = dictionary[
        ['rollin_mean', 'upper_band', 'lower_band', 'Series.Close']].astype(float)
    return dictionary
#print(bollingerbands(dictionary=dictionary, stock_price=dictionary['Series.Close'], window_size=80, width=1.9))
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111, xlabel='Date', ylabel='Close')
dictionary['Series.Close'].plot(ax=ax1, color='r', lw=1)
dictionary['upper_band'].plot(ax=ax1, color='b', lw=1)
dictionary['rollin_mean'].plot(ax=ax1, color='g', lw=1)
dictionary['lower_band'].plot(ax=ax1, color='y', lw=1)
plt.show()


class StrategyBollinger(upp=dictionary['upper_band'], low=dictionary['lower_band'], close=dictionary['Series.Close']):

    def __init__(self):
        self.upp = dictionary['upper_band']
        self.low = dictionary['lower_band']
        self.close = dictionary['Close']
        self.go_long = True

        if (self.close > self.upp) or (self.close > self.low):
            self.go_long = True
            



