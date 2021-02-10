# OPERACIONES CRUD

import sqlite3

miconexion = sqlite3.connect("GestionTienda")
micursor = miconexion.cursor()

# MODO READ
'''
micursor.execute("SELECT * FROM PRODUCT WHERE SECCION = 'CONFECCIÓN' ")
productos = micursor.fetchall()
print(productos)
'''

# MODO UPDATE
'''
micursor.execute("UPDATE PRODUCT SET PRECIO=35 WHERE NOMBRE_ARTICULO = 'pelota' ")
'''
# MODO DELETE
micursor.execute("DELETE FROM PRODUCT WHERE ID=2")

miconexion.commit()
miconexion.close()