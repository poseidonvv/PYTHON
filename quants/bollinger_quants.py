# CONSTRUCCION DE ESTRATEGIA BOLLINGER

import requests
import pandas as pd
import matplotlib.pyplot as plt

api_key = 'your api key'


def bollingerbands(stock):
   # stockprices = requests.get(
     #   f"https://financialmodelingprep.com/api/v3/historical-price-full/{stock}?serietype=line&apikey={api_key}")
    stockprices = requests.get(
        f"https://query1.finance.yahoo.com/v7/finance/download/{stock}?period1=1450915200&period2=1608768000&interval=1d&events=history&includeAdjustedClose=true")
    stockprices = stockprices.json()

    stockprices = stockprices['historical'][-150:]

    stockprices = pd.DataFrame.from_dict(stockprices)
    stockprices = stockprices.set_index('date')
    stockprices['MA20'] = stockprices['close'].rolling(window=20).mean()
    stockprices['20dSTD'] = stockprices['close'].rolling(window=20).std()

    stockprices['Upper'] = stockprices['MA20'] + (stockprices['20dSTD'] * 2)
    stockprices['Lower'] = stockprices['MA20'] - (stockprices['20dSTD'] * 2)

    stockprices[['close', 'MA20', 'Upper', 'Lower']].plot(figsize=(10, 4))
    plt.grid(True)
    plt.title(stock + ' Bollinger Bands')
    plt.axis('tight')
    plt.ylabel('Price')
    plt.savefig('apple.png', bbox_inches='tight')

bollingerbands('NQ=F')