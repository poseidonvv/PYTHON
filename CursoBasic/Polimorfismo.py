#ESTUDIANDO POLIMORFISMOS
class auto():

    def desplazamiento(self):
        print("Me desplazo utilizando 4 ruedas")

class moto():

    def desplazamiento(self):
        print("Me desplazo utilizando 2 ruedas")

class camion():

    def desplazamiento(self):
        print("Me desplazo utilizando 6 ruedas ")

def desplazavehiculo(vehiculo):
    vehiculo.desplazamiento()

vehi = camion()
desplazavehiculo(vehi)

vehi1 = moto()
desplazavehiculo(vehi1)

