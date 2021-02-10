from io import *
'''
#CREAREMOS UN ARCHIVO EXTERNO en el que almacenaremos una frase

archivo_texto = open("archivo.txt", "w") #creación de archivo
frase = "estoy estudiando python " #modificación  de archivo

archivo_texto.write(frase) #escribimos en nuestro archivo 

archivo_texto.close() # cerramos el archivo abierto en memoria desde python 

'''

# Vamos a abrir un archivo en modo lectura

'''
archivo_texto = open("archivo.txt", "r") # Cambiamos la W de write por la R de read

texto = archivo_texto.read() # metodo que lee las líneas del archivo  y los almacena en la variable texto

archivo_texto.close()
print(texto)
'''

#Usaremos Readlines()  leerá los archivos línea a línea y los almacenará en una lista

'''archivo_texto = open("archivo.txt", "r")
lineas_de_texto = archivo_texto.readlines()
archivo_texto.close()
print(lineas_de_texto)
# el resultado es una lista, recordar que las listas se imprimen con paréntesis 

'''

# usaremos el método APPEND()  agregaremos nuevas líneas



archivo_texto = open("archivo.txt", "a")  # Cambiaremos la r de Read, por una a de APPEND
archivo_texto.write("\n siempre es una buena ocasión para estudiar python")
archivo_texto.close()
print(archivo_texto)
