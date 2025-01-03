# -*- coding: utf-8 -*-
"""
ejercicio 3
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as pol


def chebyshev(f,a,b,n):
    # Con n nodos equiespaciados en (a,b)
    x = np.linspace(a,b,n)
    y = f(x) # y de los nodos
    
    xp = np.linspace(min(x),max(x),500)
    p  = pol.polyfit(x,y,len(x)-1) # coeficientes del polinomio
    yp = pol.polyval(xp,p)

    plt.figure()
    plt.plot(xp,f(xp),'b', label = 'función') # función en azul
    plt.plot(xp,yp,'r-', label = 'polinomio interpolante')
    plt.plot( x, y,'ro', label = 'puntos')
    plt.legend()
    plt.show()
    
    # Con n nodos de Chebyshev
    x = np.linspace(-1,1,n)
    for i in range(n):
        x[i] = np.cos((((2*(i+1))-1)*np.pi) / (2*n))
    y = f(x) # y de los nodos
    
    xp = np.linspace(min(x),max(x),500)
    p  = pol.polyfit(x,y,len(x)-1) # coeficientes del polinomio
    yp = pol.polyval(xp,p)

    plt.figure()
    plt.plot(xp,f(xp),'b', label = 'función') # función en azul
    plt.plot(xp,yp,'r-', label = 'polinomio interpolante')
    plt.plot( x, y,'ro', label = 'puntos')
    plt.legend()
    plt.show()
    

f1 = lambda x: 1 / (1 + 25*(x**2))
n = 11
a = -1 ; b = 1
chebyshev(f1,a,b,n)

f2 = lambda x: np.exp(-20*(x**2))
n = 15
a = -1 ; b = 1
chebyshev(f2,a,b,n)





