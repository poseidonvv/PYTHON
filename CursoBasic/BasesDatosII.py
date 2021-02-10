# CAMPOS CLAVES EN UNA BASE DE DATOS

import sqlite3

miconexion = sqlite3.connect("segundaBase.db")
micursor = miconexion.cursor()
#micursor.execute("CREATE TABLE PROD(id integer PRIMARY KEY, NOMBRE_ARTICULO VARCHAR(50), PRECIO INTEGER, SECCION VARCHAR(20))")
miconexion.commit()
#micursor.execute("INSERT INTO PROD VALUES(1, 'EDWIN', 700 'MANAGER')")
variosProductos = [
    (2, 'Juli', 100, 'wife'),
    (3, 'Jero', 50, 'son'),
    (4, 'Arelly', 700, 'manager'),
    (5, 'Leo', 100, 'wife'),
    (6, 'Osvi', 50, 'son'),
    (7, 'Alejo', 50, 'son')
]
micursor.execute("SELECT * FROM PROD")
variosProductos = micursor.fetchall()
micursor.executemany("INSERT INTO PROD VALUES (?,?,?,?)", variosProductos)
miconexion.commit()
print(variosProductos)
miconexion.close()


#nota,, para que el prymary key se autogestiones solo, basta con agregar  la casilla del Id de
#la siguiente manera
#ID INTEGER PRIMARY KEY AUTOINCREMENT,