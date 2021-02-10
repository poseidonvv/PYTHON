import sqlite3
# CREAREMOS UNA BASE DE DATOS  DONDE ESTABLECEMOS LA CONEXIÓN Y ADEMÁS PODEMOS INSERTAR DATOS A LA TABLA CREADA



miConexion = sqlite3.connect("primeraBase.db")
#PARA CREAR NUESTRA TABLA DE DATOS, PRIMERO UTILIZAREOS EL CURSOR
miCursor = miConexion.cursor()
#creamos la table de datos
#miCursor.execute("CREATE TABLE PRODUCTOS(id integer PRIMARY KEY, NOMBRE_ARTICULO VARCHAR(50), PRECIO INTEGER, SECCION VARCHAR(20))")
#el método commit  guarda loa cambios que hacemos
miConexion.commit()
#Insertar datos a la tabla
#miCursor.execute("INSERT INTO PRODUCTOS VALUES(1, 'Edwin', 700, 'manager')")

#INSERTAR EN LA BASE DE DATOS POR TUPLAS
'''variosProductos = [
    (2, 'Juli', 100, 'wife'),
    (3, 'Jero', 50, 'son'),
    (4, 'Arelly', 700, 'manager'),
    (5, 'Leo', 100, 'wife'),
    (6, 'Osvi', 50, 'son'),
    (7, 'Alejo', 50, 'son')
]'''
miCursor.execute("SELECT * FROM PRODUCTOS")
variosProductos = miCursor.fetchall()

print(variosProductos)

#miCursor.executemany("INSERT INTO PRODUCTOS VALUES (?,?,?,?)", variosProductos)
miConexion.commit()
#commit lo que hace es confirmar lo que ya hemos insertado
miConexion.commit()

miConexion.close()# cerramos la conexión

