# -*- coding: utf-8 -*-
"""
ejercicio 2
"""
import numpy as np
import sympy as sym
import scipy.optimize as op
import matplotlib.pyplot as plt

#%% Construir funciones lambda
x = sym.Symbol('x', real=True)

f_sim   = x**2 + sym.log(2*x+7) * sym.cos(3*x) + 0.1
df_sim  = sym.diff(f_sim,x)
d2f_sim = sym.diff(df_sim,x)

f   = sym.lambdify([x], f_sim,'numpy') 
df  = sym.lambdify([x], df_sim,'numpy') 
d2f = sym.lambdify([x], d2f_sim,'numpy')

#%% Dibujar df (la derivada de f)
x = np.linspace(-1,3)

plt.figure()
plt.plot(x,df(x)) 
plt.plot(x,0*x,'k')
plt.show()

#%% Calcular raíces de df
x0 = np.array([-1.,0,1,2,3])
m = op.newton(df,x0,tol=1.e-6)
print(m)

#%% Dibujar f con los Max y min
x = np.linspace(-1.5,3.5)

plt.figure()
plt.plot(x,f(x))
plt.plot(x,0*x,'k')

n = len(m)
for i in range(n):
    if d2f(m[i]) < 0: # Max
        plt.plot(m[i],f(m[i]),'ro')
    
    if d2f(m[i]) > 0: # min
        plt.plot(m[i],f(m[i]),'go')
        
# para puntos de inflexión
# plt.plot(pi,f(pi),'bo')
plt.show()


