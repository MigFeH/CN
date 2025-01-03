# -*- coding: utf-8 -*-
"""
Búsqueda incremental
"""
import numpy as np


def busquedaIncremental(f,a,b,n):
    dx = (b-a)/n
    intervalos = np.zeros((n,2))
    
    contador = 0
    
    for i in np.arange(a,b,dx):
        if f(i)*f(i+dx) < 0:
            intervalos[contador][0] = i
            intervalos[contador][1] = i+dx
            
            contador = contador + 1
        
    intervalos = intervalos[:contador,:]
    return intervalos
#-----------------------------------------------------
# primer ejemplo
f1 = lambda x : x**5 - 3 * x**2 + 1.6
a1 = -1. ; b1 = 1.5; n1 = 25

intervalos = busquedaIncremental(f1,a1,b1,n1)
print('Intervalos que contienen raíces de f1')
print(intervalos) 

#-----------------------------------------------------
# segundo ejemplo
f2 = lambda x : (x + 2)*np.cos(2*x)
a2 = 0. ; b2 = 10.; n2 = 100

intervalos = busquedaIncremental(f2,a2,b2,n2)
print('Intervalos que contienen raíces de f2')
print(intervalos) 





