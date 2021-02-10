import pandas as pd
import io
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

data = pd.read_csv("../../python-ml-course-master/python-ml-course-master/datasets/customer-churn-model/CustomerChurnModel.txt")
data1 = data
print(data)
#####################  NUBE DE PUNTOS = SCATTER PLOT ##########################
#savefig("g1.jpeg")
#figure, axs = plt.subplots(2,2, sharey=True,sharex=True)
#print(data1.plot(kind="scatter", x="Day Mins", y="Day Charge", ax=axs[0][0]))
#print(data1.plot(kind="scatter", x="Night Mins", y="Night Charge", ax=axs[1][0]))
#print(data1.plot(kind="scatter", x="Day Calls", y="Day Charge", ax=axs[1][1]))
#print(data1.plot(kind="scatter", x="Night Calls", y="Night Charge", ax=axs[0][1]))
################### HISTOGRAMAS DE FRECUENCIAS ###########################
#plt.hist(data["Day Calls"], bins=20)
#plt.hist(data["Day Calls"], bins=50)
#plt.hist(data["Day Calls"], bins=70)
#plt.hist(data["Day Calls"], bins=100)
####################  BOXPLOT = DIAGRAMAS DE CAJAS ####################
plt.boxplot(data["Day Calls"])


plt.show()

