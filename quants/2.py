# ESTRATEGY BOLLINGER

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

################################   TOMA DE DATOS  ##########################################

file = "../../../PYTHON/ES.VOLUMEBARS.100.RSI.BOLLINGER.1completo.csv"
data = open(file, "r")
cols = data.readline().strip().split(",")
# print(cols)
counter = 0
dataBase = {}
for column in cols:
    dataBase[column] = []
for line in data:
    values = line.strip().split(",")
    for i in range(len(cols)):
        dataBase[cols[i]].append(values[i])
    counter += 1
# print(" la base de datos tiene %d filas y %d columnas" % (counter, len(cols)))
# print(dictionary['Close'])
dataBase = pd.DataFrame.from_dict(dataBase)
# dataBase = dataBase.set_index('Date')
dataBase.replace('null', np.nan, inplace=True)
dataBase.dropna(inplace=True, how='any')
#print(dataBase)

#####################################  SHOW THE BOLINGER STRATEGY  ###############################################

class BollingerBands:

    def __init__(self, dataBase, stock_price, window_size, width, CROSS_BOLLINGER_DOWN,
                 CROSS_BOLLINGER_UP):
        self.dataBase = dataBase
        self.stock_price = stock_price
        self.PeriodBoll = window_size
        self.width = width
        self.stop_Loss = False

        #print(stock_price.rolling(window=window_size, win_type=None, min_periods=1))

        #################### COMPUTE MEAN AND DEVIATION STANDARD ####################################

        dataBase['rollin_mean'] = stock_price.rolling(window=window_size, win_type=None, min_periods=1).mean()
        rolling_standev = stock_price.rolling(window=window_size, win_type=None, min_periods=1).std()

        ##################### COMPUTE  UPPER AND LOW BAND ##########################################

        dataBase['upper_band'] = dataBase['rollin_mean'] + (rolling_standev * width)
        dataBase['lower_band'] = dataBase['rollin_mean'] - (rolling_standev * width)
        dataBase.dropna(inplace=True, how='any')
        dataBase[['rollin_mean', 'upper_band', 'lower_band', 'Close']] = dataBase[
            ['rollin_mean', 'upper_band', 'lower_band', 'Close']].astype(float)
        ####################### SHOW THE BOLLINGER BANDS ############################

        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(111, xlabel='Date', ylabel='Close')
        dataBase['Close'].plot(ax=ax1, color='r', lw=1)
        dataBase['upper_band'].plot(ax=ax1, color='b', lw=1)
        dataBase['rollin_mean'].plot(ax=ax1, color='g', lw=1)
        dataBase['lower_band'].plot(ax=ax1, color='y', lw=1)
        plt.show()

       ############################## BOOLEANS DECITIONS ###########################################

        self.dataBase[CROSS_BOLLINGER_DOWN] = self.dataBase.apply(
            lambda row: self.crossed_down(row['lower_band'], row['Close'], row['upper_band']), axis=1)
        self.dataBase[CROSS_BOLLINGER_UP] = self.dataBase.apply(
            lambda row: self.crossed_up(row['lower_band'], row['Close'], row['upper_band']), axis=1)


    def crossed_down(self, lower_band, Close, upper_band):
        if Close <= lower_band or Close <= upper_band:
            return 1
        else:
            return 0,

    def crossed_up(self, lower_band, Close, upper_band):
        if Close >= lower_band or Close >= upper_band:
            return 1
        else:
            return 0

        open.buy
        open.short
        can.open.shor, buy

        debo hacer,  fn inicializar, fn step(bar)

###################   STRATEGY #####################

    class Strategy:

        def __init__(self):
            self.go_long = False
            self.go_short = False
            self.position = 'none'
            self.buy = 0
            self.sell = 0
            self.profit = []

        def long(self, row):
            if self.position == 'none' or self.go_long == False:
                self.position = 'long'
                self.buy = row['Close']
                self.go_long = True
            else:
                self.buy = 0

        def short(self, row):
            if self.position == 'none':
                self.position = 'short'
                self.sell = row['Close']
                self.go_short = True
            else:
                self.sell = 0

        def close_position(self, row):
            if self.position == 'short':
                profit_add = row['Close'] - self.sell
            elif self.position == 'long':
                profit_add = self.buy - row['Close']






print(BollingerBands(dataBase=dataBase, stock_price=dataBase['Close'], window_size=100, width=1.9))

