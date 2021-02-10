print( "Verificación de acceso ")

Edad_Usuario = int(input(" Introducir la edad: "))

if Edad_Usuario <  18:
    print(" No tienes edad para acceder "+  str(Edad_Usuario))
elif Edad_Usuario>100:
    print(" Edad suficientemente alta para acceder "+  str(Edad_Usuario))
else:
    print(" Bienvenido, tienes " +  str(Edad_Usuario))
