# -*- coding: utf-8 -*-
"""

ejercicio 1
"""
import numpy as np
import matplotlib.pyplot as plt

def lagrange_fund(k,x,z):
    yz = 1. # producto acumulado
    
    for i in range(len(x)):
        if i == k:
            continue
        
        numerador = z - x[i]
        denominador = x[k] - x[i]
        
        yz = yz * (numerador / denominador)    
    
    return yz
#--------------------------
def polinomio_lagrange(x,y,z):
    yz = 0. # suma acumulada
    for i in range(len(x)):
        yz = yz + (y[i] * lagrange_fund(i, x, z))
    return yz # valor del polinomio interpolante en z

#-----------------------------------------
x = np.array([-1., 0, 2, 3, 5])
# Ejemplo 1
k = 2
z = 1.3
yz = lagrange_fund(k,x,z)
print(yz)
#----------------------------------------
# Ejemplo 2
k = 2
z = np.array([1.3, 2.1, 3.2])
yz = lagrange_fund(k,x,z)
print(yz)
#---------------------------------------
# Ejemplo 3 ----------------- Dibujarlos
x = np.array([-1., 0, 2, 3, 5])
xp = np.linspace(min(x),max(x)) # entre el nodo mas pequeño y el grande
Id = np.eye(len(x))

for k in range(len(x)):
    yp = lagrange_fund(k,x,xp)
    y = Id[k]
    plt.figure()
    plt.plot(xp,yp)
    plt.plot(x,y,'or')
    plt.plot(x,0*x,'k')
    plt.show()


# EJERCICIO 2---------------------------------
# Ejemplo 1
x = np.array([-1., 0, 2, 3, 5])
y = np.array([ 1., 3, 4, 3, 1])


xp = np.linspace(min(x),max(x)) # entre el nodo mas pequeño y el grande
yp = polinomio_lagrange(x,y,xp)

plt.figure()
plt.plot(xp,yp,'b-', label = 'polinomio interpolante')
plt.plot(x,y,'ro', label = 'puntos')
plt.legend()
plt.show()

# Ejemplo 2
x1 = np.array([-1., 0, 2, 3, 5, 6, 7])
y1 = np.array([ 1., 3, 4, 3, 2, 2, 1])


xp = np.linspace(min(x1),max(x1)) # entre el nodo mas pequeño y el grande
yp = polinomio_lagrange(x1,y1,xp)

plt.figure()
plt.plot(xp,yp,'b-', label = 'polinomio interpolante')
plt.plot(x1,y1,'ro', label = 'puntos')
plt.legend()
plt.show()