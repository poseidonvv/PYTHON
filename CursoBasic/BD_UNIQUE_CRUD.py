# CLÁUSULA  UNIQUE
# OPERACIONES CRUD

import sqlite3

miconexion = sqlite3.connect("GestionTienda")
micursor = miconexion.cursor()
micursor.execute('''
    CREATE TABLE PRODUCT(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    NOMBRE_ARTICULO VARCHAR(50) UNIQUE,
    PRECIO INTEGER,
    SECCION VARCHAR(50))
''')
#LA CLÁUSULA UNIQUE SE UTILIZA PARA ESPECIFICAR EN UNA  TABLA DE DATOS CUANDO UNA COLUMNA DEBE TENER VALORES ÚNICOS
# CABE RESALTAR QUE ESTA INSTRUCCIÓN UNIQUE ES DIFERENTE AL PRIMARY KEY

productos = [
    ("pelota", 20, "juguetería"),
    ("destornillador", 25, "ferreteria"),
    ("pantaón", 15, "confección"),
    ("jarrón", 45, "cerámica"),

]
micursor.executemany("INSERT INTO PRODUCT VALUES(NULL,?,?,?)", productos)
miconexion.commit()
miconexion.close()