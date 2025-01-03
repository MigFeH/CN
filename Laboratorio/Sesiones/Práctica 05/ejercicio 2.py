# -*- coding: utf-8 -*-
"""
Bisección
"""
import numpy as np
import matplotlib.pyplot as plt


def biseccion(f,a,b,tol=1.e-6,maxiter=100):
    niter = 1
    x = a
    
    a_k = a
    b_k = b
    
    for i in range(maxiter + 1):
        xant = x
        x = (a_k + b_k)/2
        
        if np.abs(x - xant) < tol:
            break
    
        if f(a_k)*f(x) < 0:
            a_k = a_k
            b_k = x
        
        elif f(x)*f(b_k) < 0:
            a_k = x
            b_k = b_k
            
        else:
            break
        
        niter = niter + 1
        
    return x, niter


#-------------------------------------------
# primer ejemplo
f = lambda x : x**5 - 3 * x**2 + 1.6


a = -0.7; b= -0.6;
x,i = biseccion(f,a,b)
print(x,i)


a = 0.8; b= 0.9;
x,i = biseccion(f,a,b)
print(x,i)


a = 1.2; b= 1.3;
x,i = biseccion(f,a,b)
print(x,i)

#-------------------------------------------
# segundo ejemplo
f2 = lambda x : ( (x**(3) + 1) / (x**(2) + 1) ) * np.cos(x)
a2 = -3. ; b2 = 3.;
x = np.linspace(-3,3,100)

# dibujar la función
plt.figure()
plt.plot(x,f(x))
plt.legend()
plt.show()

# estimar los tres intervalos sobre el dibujo
x,i = biseccion(f2,a2,b2)

print('%.5f' % x)













