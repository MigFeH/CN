# -*- coding: utf-8 -*-
"""
ejercicio 3
"""
import numpy as np
import matplotlib.pyplot as plt

def busquedaIncremental(f,a,b,n):
    # x = np.linspace(a,b,n+1)
    # Another way
    dx = (b-a)/n
    x = np.arange(a,b+dx,dx)
    intervalos = np.zeros((n,2))

    c = 0
    for i in range(n):
        if f(x[i])*f(x[i+1]) < 0:
            intervalos[c,:] = (x[i], x[i+1])
            c += 1

    return intervalos[:c,:]

def puntoFijo(g,x0,tol=1.e-6,maxiter=200):
    niter = 0
    x1 = 0
    
    while True:
        x1 = g(x0)
        niter = niter + 1
        
        if np.abs(x1-x0) < tol or niter == maxiter:
            break
        
        x0 = x1
    
    return x1, niter

#-----------------------------------------
# primer ejemplo
f = lambda x: np.exp(-x) - x
g = lambda x: np.exp(-x)

x = np.linspace(0,1)

intervalo = busquedaIncremental(f,0.,1.,10)
print('Existe una raiz en ', intervalo)

x0 = intervalo[0,0]
pf, niter = puntoFijo(g,x0)
print(pf,niter)

plt.figure()
plt.plot(x,g(x),'r', label = 'g')
plt.plot(pf,g(pf),'bo')
plt.plot(x,x,'b')
plt.legend()
plt.show()

#-----------------------------------------
# segundo ejemplo
f = lambda x: x - np.cos(x)
g1 = lambda x: np.cos(x)
g2 = lambda x: (2*x) - np.cos(x)
g3 = lambda x: x - ((x-np.cos(x))/(1+np.sin(x)))
g4 = lambda x: ( (9*x) + np.cos(x) ) / 10
    
    
x = np.linspace(0,1)

intervalo = busquedaIncremental(f,0.,1.,10)
print('Existe una raiz en ', intervalo)

x0 = intervalo[0,0]

pf1, niter1 = puntoFijo(g1,x0)
pf2, niter2 = puntoFijo(g2,x0)
pf3, niter3 = puntoFijo(g3,x0)
pf4, niter4 = puntoFijo(g4,x0)

print('g1  ',pf1,niter1)
print('g2  ',pf2,niter2)
print('g3  ',pf3,niter3)
print('g4  ',pf4,niter4)

plt.figure()
plt.plot(pf1,g1(pf1),'bo')

plt.plot(x,g1(x),'red',label = 'g1')
plt.plot(x,g2(x),'purple',label = 'g2')
plt.plot(x,g3(x),'green',label = 'g3')
plt.plot(x,g4(x),'black',label = 'g4')

plt.plot(x,x,'b', label = 'y = x')
plt.legend()
plt.show()
 