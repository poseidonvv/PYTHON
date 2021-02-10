
'''limit = int(input(" Digita el número de impares  "))
def genImpar(limit):
    num = 1
    miLista = []
    while num < limit:
        miLista.append(num*2-1)
        num += 1
    return miLista
print ( genImpar ( limit ) )'''

'''USANDO DOS DIMENSIONES'''


def returnCity(*cities):
    for element in cities:
        yield from element
city_return = returnCity("Medellin","Bogotá","Madrid","Barbosa")
print(next(city_return))
print(next(city_return))
