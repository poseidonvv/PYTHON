import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
file = "../../../PYTHON/NQ=F.csv"
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
class bollingerbands(self, dictionary, stock_price, window_size, width ):
    self.dictionary = dictionary
    self.stock_price = stock_price