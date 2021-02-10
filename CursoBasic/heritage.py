class Vehiculos():
    def __init__(self, marca, modelo):
        self.marca = marca
        self.modelo = modelo
        self.enmarcha = False
        self.acelera = False
        self.frena = False
        self.cargar = False



    def arranca(self):
        self.enmarcha = True

    def acelerar(self):
        self.acelera = True

    def frenar(self):
        self.frena = True

    def estado(self):
        print("Marca: ", self.marca, "\n Modelo: ", self.modelo, "\n En marcha: ", self.enmarcha, "\n Acelerando: ", self.acelera, "\n Frenando: ", self.frena)

class Moto(Vehiculos):
    hcaballito = ""
    def caballito(self):
        self.hcaballito = "Voy haciendo caballito"

    def estado(self):
        print("Marca: ", self.marca, "\n Modelo: ", self.modelo, "\n En marcha: ", self.enmarcha, "\n Acelerando: ",
              self.acelera, "\n Frenando: ", self.frena, "\n ", self.hcaballito)


class furgoneta(Vehiculos):

    def carga(self, cargar):
        self.cargado = cargar
        if(self.cargado):
            print("La furgoneta está cargada")
        else:
            print("La furgoneta no está cargada")

    def estado(self):
     print("Marca: ", self.marca, "\n Modelo: ", self.modelo, "\n En marcha: ", self.enmarcha, "\n Acelerando: ",
            self.acelera, "\n Frenando: ", self.frena,  "\n Cargado: ", self.cargado)


class VElectrico(Vehiculos):
    def __init__(self, marca, modelo):
        super().__init__(marca, modelo)

        self.autonomia = 100

    def cargarEnergia(self):
        self.cargadoEnergy = True

class BicicletaElectrica(VElectrico, Vehiculos): #para llamar la clase padre (Vehiculos) usaremos el método super()
    pass

bici1 = BicicletaElectrica("tesla", "osijdfsd")





moto1 = Moto("suzuki", "115")
moto1.caballito()
moto1.estado()
print("------------------nuevo objeto-----------------")
furgon1 = furgoneta("foton", "dog")
furgon1.carga(True)
furgon1.estado()

