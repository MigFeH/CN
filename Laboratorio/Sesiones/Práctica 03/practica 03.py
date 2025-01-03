# -*- coding: utf-8 -*-
"""
Polinomios
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as pol

p = np.array([1., -1, 2, -3,  5, -2])
x0 = -0.5
print('Valor de P en el punto  ', x0)
print('Con polyval:            ', pol.polyval(x0, p)) # nos da el valor del polinomio en el punto x0


x = np.linspace(-1,1)

plt.figure()
plt.plot(x,pol.polyval(x0,p))
plt.plot(x,0*x,'k')
plt.title('Polinomio P')
plt.legend()
plt.show()
#%% Algoritmo de Horner

#n = len(p) # n == 6
#q[5] = p[5]
#for i in range(n-2,-1,-1):
#    q[n] = ... # q[4]
               # q[3]
#               ...

for i in range(4,-1,-1):
    print(i)

#%% Ejercicio 1
import numpy as np
import numpy.polynomial.polynomial as pol 

def horner(x0,p):
    q = np.zeros_like(p)
    n = len(p)
    q[n-1] = p[n-1]
    for i in range(n-2,-1,-1):
        q[i] = p[i] + q[i+1] * x0
    resto = q[0]
    cociente = q[1:]
    return cociente, resto

def main():
    p0 = np.array([1.,2,1])
    x0 = 1
    c, r = horner(x0,p0)
    rp   = pol.polyval(x0,p0) 

    print('Coeficientes de Q = ', c)
    print('P0(1)       = ', r)
    print('Con polyval = ', rp)

    # <--- ejemplos con polinomios p1 y p2
    p1 = np.array([1., -1, 2, -3,  5, -2])
    x1 = 1.
    c, r = horner(x1,p1)
    rp   = pol.polyval(x1,p1) 

    print('Coeficientes de Q = ', c)
    print('P0(1)       = ', r)
    print('Con polyval = ', rp)
    
    p2 = np.array([1., -1, -1, 1, -1, 0, -1, 1])
    x2 = -1.
    c, r = horner(x2,p2)
    rp   = pol.polyval(x2,p2) 

    print('Coeficientes de Q = ', c)
    print('P0(-1)       = ', r)
    print('Con polyval = ', rp)


if __name__ == "__main__":
    main()
    
#%% Ejercicio 2
import numpy as np
import numpy.polynomial.polynomial as pol
import matplotlib.pyplot as plt

def HornerV(x,p):
    y = np.zeros_like(x)
    
    for j in range(0,len(x),1):
        q = np.zeros_like(p)
        n = len(p)
        q[n-1] = p[n-1]
        for i in range(n-2,-1,-1):
            q[i] = p[i] + q[i+1] * x[j]
        resto = q[0]
        y[j] = resto
     
    return y


x = np.linspace(-1,1)
p = np.array([1., -1, 2, -3, 5, -2])
r = np.array([1., -1, -1, 1, -1, 0, -1, 1])

plt.figure()
plt.plot(x,HornerV(x, p), label = 'Polinomio P')
plt.plot(x,HornerV(x, r), label = 'Polinomio R')
plt.plot(x,0*x,'k')
plt.title('Polinomio P')
plt.legend()
plt.show()

#%% Ejercicio 3
import numpy as np
import numpy.polynomial.polynomial as pol
import matplotlib.pyplot as plt
np.set_printoptions(suppress = True)

def dersuc(x0,p):
    restos = np.zeros_like(p)
    nuevo_cociente = np.zeros(len(p)-1)
    
    nuevo_cociente, restos[0] = horner(x0,p)
    
    for i in range(1,len(restos)):
        nuevo_cociente, restos[i] = horner(x0, nuevo_cociente)
        
    return restos

def horner(x0,p):
    q = np.zeros_like(p)
    n = len(p)
    q[n-1] = p[n-1]
    for i in range(n-2,-1,-1):
        q[i] = p[i] + q[i+1] * x0
    resto = q[0]
    cociente = q[1:]
    return cociente, resto
    
p = np.array([1., -1, 2, -3,  5, -2])
x0 = 1.

print('Restos de dividir P una y otra vez por (x-x0): ', dersuc(x0,p))

r = np.array([1., -1, -1, 1, -1, 0, -1, 1])
x1 = -1.

print('Restos de dividir R una y otra vez por (x-x1): ', dersuc(x1,r))
    
#%% Ejercicio 3 B
import numpy as np
import numpy.polynomial.polynomial as pol
import matplotlib.pyplot as plt
np.set_printoptions(suppress = True)

def horner(x0,p):
    q = np.zeros_like(p)
    n = len(p)
    q[n-1] = p[n-1]
    for i in range(n-2,-1,-1):
        q[i] = p[i] + q[i+1] * x0
    resto = q[0]
    cociente = q[1:]
    return cociente, resto

def dersuc(x0,p):
    restos = np.zeros_like(p)

    fact = 1
    der = np.zeros_like(p)
    aux = np.zeros_like(p)
    
    
    nuevo_cociente, restos[0] = horner(x0,p)
    der[0] = restos[0]
    print('restos[0] = ',restos[0])
    for i in range(1,len(restos)):
        aux[i] = i
        fact = fact * aux[i]
        nuevo_cociente, restos[i] = horner(x0, nuevo_cociente)
        der[i] = restos[i] * fact
        
    return der

p = np.array([1., -1, 2, -3,  5, -2])
x0 = 1.

print('Derivadas sucesivas de P en x0 = 1: ', dersuc(x0,p))

r = np.array([1., -1, -1, 1, -1, 0, -1, 1])
x1 = -1.

print('Derivadas sucesivas de R en x1 = -1: ', dersuc(x1,r))
    
    
    
    
    
    
