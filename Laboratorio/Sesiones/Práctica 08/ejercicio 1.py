# -*- coding: utf-8 -*-
"""
ejercicio 1
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as pol

def aprox1(f,g,a,b,n):
    # 1. Construir n puntos equiespaciados en (a,b)
    x = np.linspace(a,b,n)
    y = f(x)
    
    # 2. Construir el sistema (C y d)
    # Construir V (matriz de valdemorde)
    
    # tantas filas como nodos y tantas columnas como g+1
    V = np.zeros((n,g+1))
    
    #V[:,i] = (columna anterior) * x
    for i in range(len(V)):
        V[i][0] = 1
        for j in range(1,len(V[0])): 
            V[i][j] = V[i][j-1] * x[i]
    
    C = np.dot(V.T,V)
    d = np.dot(V.T,y)
    
    # 3. Resolver el sistema
    p = np.linalg.solve(C,d)
    print(p)
    
    # 4. Dibujar los puntos y el polinomio de ajuste
    xp = np.linspace(a,b)
    yp = pol.polyval(xp,p)
    
    plt.figure()
    plt.plot(xp,yp,'b-', label = 'función aproximada')
    plt.plot( x, y,'ro', label = 'puntos')
    plt.legend()
    plt.show()
    

#----------------------------------
# primer ejemplo
f = lambda x: np.sin(x)
a = 0.; b = 2.; n = 5; g = 2
aprox1(f,g,a,b,n)

#----------------------------------
# segundo ejemplo
f2 = lambda x: np.cos(np.arctan(x)) - np.log(x + 5)
a = -2.; b = 0.; n = 10; g = 4
aprox1(f2,g,a,b,n)






