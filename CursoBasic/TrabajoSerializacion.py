# TRABAJAREMOS SERIALIZACIÓN DE ARCHIVOS

import pickle
# FORMA DE SERIALIZAR UN ARCHIVO
'''
lista_nombres = ["Pedro", "Ana", "María", "Isabel"]

fichero_binario = open("lista_nombres", "wb")

pickle.dump(lista_nombres, fichero_binario)
fichero_binario.close()'''

# FORMA DE RESCATAR UN ARCHIVO SERIALIZADO

fichero = open("lista_nombres", "rb")
lista = pickle.load(fichero)
print(lista)




