# LEEREMOS LOS DATOS PROCEDENTES DESDE UN CSV

import pandas as pd
import numpy as np


#data = pd.read_excel("../../../DATA/dictionary.xlsx")
#data1 = pd.read_csv("../../../DATA/Nueva carpeta/NQ.VOLUMEBARS4000.BOLLINGER.100.1.9.csv")
#print(data1.head())

#  LEER INFORMACION DE UN ARCHIVO Y CREAR UN ARCHIVO NUEVO, ESCRIBIR SOBRE ÉL
'''
mainpath ="../../../DATA/Nueva carpeta"
filename = "NQ.VOLUMEBARS4000.BOLLINGER.100.1.9.csv"
infile = mainpath + "/" + filename
outfile = mainpath + "/" + "recibo datos.txt"'''

#  USO DE COMANDO WITH OPEN

'''with open(infile, "r") as infile1:
    with open(outfile, "w") as outfile1:
        for line in infile1:
            fields = line.strip().split(",")
            outfile1.write("\t".join(fields))
            outfile1.write("\n")
df = pd.read_csv(outfile, sep="\t")

'''
# ACCEDER A LA INFORMACIÓN OBSERVANDO COLUMNAS Y FILAS UTILIZANDO DICCIONARIOS
'''data2 = open(mainpath + "/" + filename, "r")

cols = data2.readline().strip().split(",")
n_cols = len(cols)
counter = 0
main_dict = {}
for col in cols:
    main_dict[col] = []
for line in data2:
    values = line.strip().split(",")
    for i in range(len(cols)):
        main_dict[cols[i]].append(values[i])
    counter += 1

print("El data set tiene %d filas y %d columnas" %(counter, n_cols))
df3 = pd.DataFrame(main_dict)
#print(df3)
'''

# LEER ARCHIVOS DESDE UNA URL
'''
medals_url = "http://winterolympicsmedals.com/medals.csv"
import urllib3
http = urllib3.PoolManager()
r = http.request('GET', medals_url)
print(r.status)
response = r.data.decode('UTF-8')
#print(response)
lineas = response.split("\n")
#print(lineas[0])
cont = 0
dict = {}
columnas = lineas[0].split(",")
n_col = len(columnas)
for col in columnas:
    dict[col] = []
for lin in lineas:
    values = lin.strip().split(",")
    for i in range(len(columnas)):
        dict[columnas[i]].append(values[i])
    cont += 1
print("El data set tiene %d filas y %d columnas" % (cont, n_col))
df3 = pd.DataFrame(dict)
print(df3)
'''

# OBTENER INFORMACIÓN DESDE UN ARCHIVO EN EXCEL
import xlrd
mainpath = "../../../SIMULACIONES/BOLLINGER"
filepath = "ATAS_statistics_25082020_28082020CL 10SEG.xlsx"
fullpath = mainpath + "/" + filepath
statisics = pd.read_excel(fullpath, sheet_name='Statistics', engine='openpyxl', header=0)
journal = pd.read_excel(fullpath, sheet_name='Journal', engine='openpyxl', header=0)
executions = pd.read_excel(fullpath, sheet_name='Executions', engine='openpyxl', header=0)
df_statistics = pd.DataFrame(statisics)
df_journal = pd.DataFrame(journal)
df_executions = pd.DataFrame(executions)
#print(df_journal)
#convert excel to csv
statisics.to_csv(mainpath + "/" + "statisticsbollifromexcel.csv")
























