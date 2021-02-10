# GUARDAREMOS  DE FORMA PERMANENTE EN UN FICHER, CIERTA INFORMACIÓN
import pickle

class persona:
    def __init__(self, nombre, genero, edad):
        self.nombre = nombre
        self.genero = genero
        self.edad = edad
        print(" Se ha creado una persona nueva con el nombre de ", self.nombre)

    def __str__(self):
        return "{} {} {}".format(self.nombre, self.genero, self.edad)

class ListasPersonas:
    personas = []
# CREAREMOS UN CONSTRUCTOR PARA ALMACENAR TODA LA INFORMACIÓN QUE RECIBIMOS
    def __init__(self):
        ListaDePersonas = open("ficheroExterno", "ab+")
        ListaDePersonas.seek(0)

        try:
            self.personas = pickle.load(ListaDePersonas)
            print("Se cargaron {} personas del fichero ".format(len(self.personas)))

        except:
            print("No hay personas almacenadas ")

        finally:
            ListaDePersonas.close()
            del(ListaDePersonas)

    def mostrarInfoFicheroExterno(self):
        print("La información del fichero externo es la siguiente: ")
        for p in self.personas:
            print(p)

    def agregarPersonas(self, p):
        self.personas.append(p)
        self.guardarPersonasEnFicheroExterno()

    def mostrarPersonas(self):
        for p in self.personas:
            print(p)

    def guardarPersonasEnFicheroExterno(self):
        ListaDePersonas = open("ficheroExterno", "wb")
        pickle.dump(self.personas, ListaDePersonas)
        ListaDePersonas.close()
        del (ListaDePersonas)

miLista = ListasPersonas()
p = persona("Edwin", "Diaz", 34)
miLista.agregarPersonas(p)
miLista.mostrarInfoFicheroExterno()

