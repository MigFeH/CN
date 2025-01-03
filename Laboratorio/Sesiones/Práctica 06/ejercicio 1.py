# -*- coding: utf-8 -*-
"""
ejercicio 1
"""
import numpy as np
import matplotlib.pyplot as plt

def secante(f,x0,x1,tol = 1.e-6,maxiter=100):
    niter = 0
    x2 = 0
    while True:
        x2 = x1 - f(x1) * ( (x1 - x0) / (f(x1) - f(x0)))
        
        niter = niter + 1
        
        if np.abs(x2-x1) < tol or niter == maxiter:
            break
        
        x0 = x1
        x1 = x2
        
    return x2, niter
#---------------------------------------------------
f = lambda x : x**5 - 3 * x**2 + 1.6
r = np.zeros(3)


x0 = -0.7; x1 = -0.6
r[0],i = secante(f,x0,x1)
print(r[0], i)

x0 = 0.8; x1 = 0.9
r[1],i = secante(f,x0,x1)
print(r[1], i)

x0 = 1.2; x1 = 1.3
r[2],i = secante(f,x0,x1)
print(r[2], i)

x = np.linspace(-1,1.5)                # definimos un vector con 50 elementos en (-1,1.5)
ox = 0*x                               # definimos un vector de ceros del tamaño de x
plt.figure()
plt.plot(x,f(x))                   # dibujamos la función 
plt.plot(x,ox,'k-')                # dibujamos el eje X
plt.plot(r,0*r,'ro')    # dibujamos las raíces
plt.show()