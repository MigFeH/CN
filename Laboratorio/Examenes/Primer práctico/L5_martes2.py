# -*- coding: utf-8 -*-
"""
Raíces
"""
import numpy as np
import matplotlib.pyplot as plt

def busquedaIncremental(f,a,b,n):
    x = np.linspace(a,b,n+1)
    intervalos = np.zeros((n,2))
    contador = 0
    for i in range(n):
        if f(x[i])*f(x[i+1]) < 0:
            intervalos[contador] = x[i:i+2]
            contador +=1
    intervalos = intervalos[:contador]
    return intervalos
#-------------------------------------------    

def raices(f,a,b,tol=1.e-4):
    # Primera pasada
    n1 = int((b-a) / 0.1)
    intervalos = busquedaIncremental(f,a,b,n1)

    num_raices = len(intervalos)

    r = np.zeros((1,num_raices))
    
    for i in range(len(intervalos)):
        x0 = intervalos[i][0]
        x1 = intervalos[i][1]
        
        numiter = int(-np.log10(tol)-1)
        
        for j in range(numiter):
            
            inter = busquedaIncremental(f,x0,x1,10)
            
            x0 = inter[0][0]
            x1 = inter[0][1]
            
            if x1 - x0 < tol:
                r[0][i] = x0
                break
    
            r[0][i] = x0
    return r 

#-------------------------------------------   
# primer ejemplo
np.set_printoptions(precision = 5) 

p = np.array([1., 0.2, -7., 0.7, 8])
f = lambda x: np.polyval(p,x)
a = -3
b = 2.5
tol = 1.e-5
r = raices(f,a,b,tol)
print(r)

x = np.linspace(a,b)

plt.figure()
plt.plot(x,f(x))
plt.plot(x,0*x, 'k')
plt.show()
   
#-------------------------------------------
# segundo ejemplo
np.set_printoptions(precision = 3) 

p = np.array([1., -0.3, -5.8, 3, 5, -2.6])
f = lambda x: np.polyval(p,x)
a = -2.5; b = 2.5
tol = 1.e-3
r = raices(f,a,b,tol)
print(r)

x = np.linspace(a,b)

plt.figure()
plt.plot(x,f(x))
plt.plot(x,0*x, 'k')
plt.show()