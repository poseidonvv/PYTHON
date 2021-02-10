# Resumen de los datos

import pandas as pd
import os

mainpath = "../../python-ml-course-master/python-ml-course-master/datasets"
filepath = "titanic/titanic3.csv"
fullpath = os.path.join(mainpath, filepath) # the method os, join the two address  without we need th "/"

url_data = "https://raw.githubusercontent.com/joanby/python-ml-course/master/datasets/titanic/titanic3.csv"

data1 = pd.read_csv(url_data)
#data = pd.read_csv(fullpath)
########### para ver las dimensiones del csv
#print(data1.columns)
#print(data1.columns.values)
############ VAMOS A HACER UN RESUMEN ESTADÍSTICO BÁSICO DE LAS VARIABLES NUMÉRICAS ###################
#print(data1.describe())   # PRINT THE MEAN, STD, MIN, MAX 25%, 50%, 75% FROM EACH COLUMNS IN OUR  DATASET
#print(data1.dtypes) # PRINT ALL TYPES FOR EACH ONE OF OUR COLUMNS
#######################  missing values ################################
#print(pd.isnull(data1["name"])) # isnull. bool that show all values zero us
#print(pd.notnull(data1["name"])) # not null is the opposite values to  isnull
#print(pd.isnull(data1["name"]).values.ravel().sum()) # SUM THE VALUES NULL
#print(pd.notnull(data1["name"]).values.ravel().sum())# SUM THE VALUES FALSE OR NOT NULL
################  LOS VALORES QUE FALTAN EN UNA BASE DE DATOS QUE HACER ##########################
########## BORRADO DE VALORES QUE FALTAN ##########################
#data1.dropna(axis=1, how= "all ") # elimina todas las columnas que tengan valores NAN cuando axis = 0 borra todas las filas
                      # donde las filas tienen algún NAN, la instrucción HOW ESPECIFICA LAS CONDICIONES DE BORRADO
#data1.dropna(axis=1, how ="any") # lo que indica que elimina una columna si existe a lo sumo un NAN

############## computo de los valores faltantes ########################
data3 = data1
#print(data3.fillna(0)) # CAMBIO TODOS LOS VALORES NaN por ceros
#data_column = data3["name"].fillna(0)
#print(data_column)

#################### VARIABLES DUMMY ###########################
#print(data1.columns.values)
#### se denominan variables basura, en realidad es convertir datos fijos en vectores de ceros y unos
######## el ejemplo más fácil es categorizar  el género si masculino o femenino
data3 = data1["sex"].head()
dummy_sex = pd.get_dummies(data1["sex"], prefix="sex") # obtiene las variables dummy, para juntarlas a la base de datos simplemente
data1 = pd.concat([dummy_sex, data1], axis=1)
print(data1)

########## FUNCION DE DUMMIES ############
def createDummies(df, var_name:str):
    dummy = pd.get_dummies(df[var_name], prefix=var_name)
    df = df.drop(var_name, axis=1)
    df = pd.concat([dummy,df], axis=1)
    return df
######## example ###########
#print(createDummies(data3, "sex")) #imprime el resultado de la función

