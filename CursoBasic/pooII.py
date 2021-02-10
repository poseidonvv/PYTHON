
'''
class coche():
    LargoChasis = 260
    anchoChasis = 180
    ruedas = 4
    enmarcha = False  # método

# La diferencia entre una función y un método es que el método está atado a la clase, mientras que las funciones no

    def arrancar(self):
        self.enmarcha = True

    def estado(self):
        if (self.enmarcha):
            return "El coche está en marcha "
        else:
            return "el coche está parado "

miCoche = coche() #instanciamos una clase, creamos un objeto de la clase coche
print("El largo del coche es: ", miCoche.LargoChasis)
print("El ancho del coche es: ", miCoche.anchoChasis)
miCoche.arrancar()
print(miCoche.estado())

'''


'''
# MEJORANDO EL CÓDIGO ANTERIOR,  el método arrancar lo vamos a suprimir para que el método estado informe si está parado o en movimiento

class coche():
    largoChasis = 260
    anchoChasis = 180
    ruedas = 4
    enmarcha = False

    def arrancar(self,arrancamos):
        self.enmarcha = arrancamos
        if (self.enmarcha):
            return "El coche está en marcha "
        else:
            return "El coche está parado "

    def estado(self): #Nos informará sobre las propiedades del coche
        print("El coche tiene ", self.ruedas, " Ruedas. un ancho de ", self.anchoChasis, " y un largo  de ", self.largoChasis)


miCoche1 = coche()
miCoche2 = coche()
miCoche1.estado()
print(miCoche1.arrancar(True))
miCoche2.estado()
print(miCoche2.arrancar(False))'''


#UTILIZARERMOS EL MISMO CÓDIGO PERO USANDO UN MÉTODO CONSTRUCTOR O __init__
'''
class coche():

    def __init__(self): # método constructor
        # propiedades que tendrá el método constructor en este caso los atributos está encapsulados
        self.__largoChasis = 260
        self.__anchoChasis = 160
        self.__ruedas = 4
        self.__enmarcha = False
        self.chequeo = False




    def arrancar(self, arrancamos):
        self.__enmarcha = arrancamos
        print("Realizando chequeo interno ")
        chequeo = self.__chequeo_interno()
        if (self.__enmarcha and chequeo):
            return "El chequeo fué éxitoso, carro está en marcha"

        elif(self.__enmarcha and chequeo==False):
            return "Algo ha ido mal en el chequeo, no se puede arrancar "

        else:
            return "El carro está parado"

    def estado(self):
        print("El carro tiene un ancho de ", self.__anchoChasis, " y un largo de ", self.__largoChasis, " y tiene ", self.__ruedas, " ruedas")


# Este método tiene sentido si lo llamamos antes del método arrancar
    def __chequeo_interno(self): #encapsulamos el método para que no pueda ser modificado 
#Notemos que para iniciar las variables siguientes, las debo declarar  en los argumentos del método
        self.gasolina = "ok"
        self.aceite = "ok"
        self.puertas = "ok"

        if (self.gasolina == "ok" and self.aceite == "ok" and self.puertas == "ok"):
            return True
        else:
            return False

coche1 = coche()
print(coche1.arrancar(True))
coche1.estado()
coche2 = coche()
print(coche2.arrancar(False))
coche2.estado()

'''

