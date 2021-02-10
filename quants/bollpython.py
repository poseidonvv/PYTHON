import datetime
from enum import Enum
import matplotlib.pyplot as plt


class BollingerBands:

    def __init__(self, window_size, width):
        self.window_size = window_size
        self.width = width
        self.direction = Direction()
        self.c_position = 0
        self.leftstoploss = False
        self.go_long = False
        self.go_short = False
        self.a: float
        self.b: float

    def bolinger_step(self, bar_ind: int, dB=[]):
        self.c_position = self.direction('FLAT')
        self.index = bar_ind
        dB['dt_mean'][self.index] = dB['Close'].rolling(window=self.window_size, win_type=None,
                                                                     min_periods=1).mean()
        dB['Close'] = Bar(self.index).close
        ######################## mean and dev stand  #############################
        dB['dt_mean'][self.index] = dB['Close'][self.index].rolling(window=self.window_size, win_type=None,
                                                            min_periods=1).mean()
        rolling_standev = dB['Close'][self.index].rolling(window=self.window_size, win_type=None, min_periods=1).std()

        ##################################compute upp and down bands #############################

        dB['upper_band'][self.index] = dB['dt_mean'][self.index] + (rolling_standev * self.width)
        dB['lower_band'][self.index] = dB['dt_mean'][self.index] - (rolling_standev * self.width)
        dB['sigma_prim_upp'][self.index] = dB['upper_band'][self.index].apply(lambda row: self.sigma_prim_upp(row['upper_band'][self.index-1], row['upper_band'][self.index-2]))
        dB['sigma_prim_low'][self.index] = dB['lower_band'][self.index].apply(lambda row: self.sigma_prim_low(row['lower_band'][self.index-1], row['lower_band'][self.index-2]))

        ################# STRATEGY################
        if self.go_long == 'False' and dB['Close'][self.index - 1] >= dB['upper_band'][self.index-1] or dB['Close'][self.index - 1] >= dB['lower_band'][self.index-1]:
            self.c_position = self.direction('BUY')
            self.go_long = True
            a = dB['Close'][self.index - 1]

        if self.go_short == 'False' and dB['Close'][self.index - 1] <= dB['upper_band'][self.index-1] or dB['Close'][self.index - 1] <= dB['lower_band'][self.index-1]:
            self.c_position = self.direction('SELL')
            self.go_short = True
            b = dB['Close'][self.index - 1]

        if self.go_long == 'False' and self.leftstoploss and dB['Close'][self.index - 1] > dB['Close'][self.index - 2] and dB['Close'][self.index - 1] >= self.a:
            self.c_position = self.direction('BUY')
            self.go_long = True
            self.leftstoploss = False

        if self.go_short == 'False' and self.leftstoploss and dB['Close'][self.index - 1] < dB['Close'][self.index - 2] and dB['Close'][self.index - 1] <= self.b:
            self.c_position = self.direction('SELL')
            self.go_short = True
            self.leftstoploss = False

        if self.go_long == 'False' and self.leftstoploss and dB['sigma_prim_upp'][self.index] > 0 and dB['Close'][self.index - 1] > dB['Close'][self.index - 2] and dB['Close'][self.index - 1] > dB['dt_mean'][self.index - 1]:
            self.c_position = self.direction('BUY')
            self.go_long = True
            self.leftstoploss = False

        if self.go_short == 'False' and self.leftstoploss and dB['sigma_prim_low'][self.index] < 0 and dB['Close'][self.index - 1] < dB['Close'][self.index - 2] and dB['Close'][self.index - 1] < dB['dt_mean'][self.index - 1]:
            self.c_position = self.direction('SELL')
            self.go_short = True
            self.leftstoploss = False

######################################### cierro las posiciones ###############################################

        if self.go_long == 'True' and (dB['Close'][self.index - 1] < dB['upper_band'][self.index-1] or dB['Close'][self.index - 1] < dB['lower_band'][self.index-1]):
            self.go_long = False

        if self.go_short == 'True' and (dB['Close'][self.index - 1] > dB['lower_band'][self.index-1] or dB['Close'][self.index - 1] > dB['upper_band'][self.index-1]):
            self.go_short = False

        #NOTEMOS QUE FALTA UNA CONDICIÓN PARA CERRAR POSICIONES POR STOP LOSS
        return self.c_position

    def sigma_prim_upp(self, uppt, uppt_1):
            derivada_upp = uppt - uppt_1
            return derivada_upp

    def sigma_prim_low(self, lowt, lowt_1):
            derivada_low = lowt - lowt_1
            return derivada_low

class Direction(Enum):
    BUY = 1
    SELL = 2
    FLAT = 3


class Order:
    def __init__(self, direction: Direction, quantity: int):
        self.quantity = quantity
        self.direction = direction


class Own_Trade:
    def __init__(self, order: Order, time: datetime, price: float):
        self.order = order
        self.time = time
        self.price = price


class Position:
    def __init__(self, price: float, volume: int):
        self.profits_and_losses = 0.0
        self.average_price = price
        self.volume = volume


class Account:
    def __init__(self):
        self.balance = 20000.0
        self.profits_and_losses = 0.0
        self.commission = 0.0


class Bar:
    def __init__(self):
        self.time = datetime.fromtimestamp(0)
        self.open = 0
        self.close = 0
        self.high = 0
        self.low = 0
        self.volume = 0

    def __init__(self, time: datetime,
                 open: float, close: float,
                 high: float, low: float, volume: int):
        self.time = time
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume


class Trade:
    def __init__(self, time: datetime, price: float, quantity: int, quantit=None):
        self.time = time
        self.price = price
        self.quantity = quantit
