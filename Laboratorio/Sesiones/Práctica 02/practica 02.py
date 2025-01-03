# -*- coding: utf-8 -*-
"""
Vectorización

"""

import numpy as np
import time

f = lambda x: np.exp(x)
x = np.linspace(-1,1,400000)
#%% vector que va creciendo dentro del bucle
y = np.array([])

t = time.time()
for i in range(len(x)):
    y = np.append(y,f(x[i]))
t1 = time.time() - t
#%% reservamos espacio antes de entrar en el bucle
y = np.zeros_like(x)

t = time.time()
for i in range(len(x)):
    y[i] = np.append(y,f(x[i]))
t2 = time.time() - t
#%% vectorizacion
t = time.time()
y = f(x)
t3 = time.time() - t
#%%
'''
Gráficas

'''

import numpy as np
import matplotlib.pyplot as plt

f = lambda x: np.exp(x)
x = np.linspace(-1,1)
y = f(x)
ox = 0*x

plt.figure()
plt.plot(x,y, label = 'f')
plt.plot(x,ox,'k', label = 'Eje OX')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ejemplo de gráfica')
plt.show()
#%% Taylor

import numpy as np

f = lambda x: np.exp(x)
x0 = 0.5

pol = 0.
fact = 1.

for i in range(10):
    term = x0**i / fact
    pol += term
    
    fact *= i+1
    
print('valor exacto = ', f(x0))
print('valor aprox. = ', pol)

#%% Con def
import numpy as np

def P(x0,grado):
    pol = 0.
    fact = 1.

    for i in range(grado+1):
        term = x0**i / fact
        pol += term
    
        fact *= i+1
    return pol
    

f = lambda x: np.exp(x)
x0 = 0.5
print('valor exacto = ', f(x0))
print('valor aprox. = ', P(x0,12))

#%% Con vectorización
import numpy as np

def P(x0,grado):
    pol = 0.
    fact = 1.

    for i in range(grado+1):
        term = x0**i / fact
        pol += term
    
        fact *= i+1
    return pol
    

f = lambda x: np.exp(x)
x0 = np.linspace(-1,1,4)
print('valor exacto = ', f(x0))
print('valor aprox. = ', P(x0,12))

#%% 
import numpy as np
import matplotlib.pyplot as plt

def P(x0,grado):
    pol = 0.
    fact = 1.

    for i in range(grado+1):
        term = x0**i / fact
        pol += term
    
        fact *= i+1
    return pol
    

f = lambda x: np.exp(x)
x = np.linspace(-1,1)
y = P(x,2)

plt.figure()
plt.plot(x,f(x), label='f')
plt.plot(x,0*x,'k')
plt.plot(x,y,label='P2')
plt.legend()
plt.show()

#%% 
import numpy as np
import matplotlib.pyplot as plt

def P(x0,grado):
    pol = 0.
    fact = 1.

    for i in range(grado+1):
        term = x0**i / fact
        pol += term
    
        fact *= i+1
    return pol
    

f = lambda x: np.exp(x)
x = np.linspace(-3,3) #<-------------------

plt.figure()
plt.plot(x,f(x), label='f')
plt.plot(x,0*x,'k')

for grado in range(1,6):    
    plt.plot(x,P(x,grado),label='P'+str(grado))
    plt.legend()
    plt.pause(2)
    
plt.show()
#%%
i = 0
while i < 5:
    print(i)
    i += 1
    
#%% Ejercicio 1
import numpy as np

f = lambda x: np.exp(x)

x0 = -0.4

pol = 0
fact = 1

tol = 1.e-8

numSum = 0
maxNumSum = 100

i = 0
while(True):
    termino = x0**i / fact
    pol = pol + termino
    
    numSum = numSum + 1
    
    if(np.abs(termino) < tol or numSum == maxNumSum):
        break
    
    fact = fact * (i+1)
    i = i+1
    
print('Valor de la función en x = ',x0,': ', f(x0))
print('Valor de la aproximación en x = ',x0,': ',pol)
print('Número de iteraciones: ', numSum)

#%% Ejercicio 2
import numpy as np
import matplotlib.pyplot as plt

f = lambda x: np.exp(x)

x = np.linspace(-1,1,50)
tol = 1.e-8
maxNumSum = 100
 
def funExp(x,tol,maxNumSum):
    y = np.zeros_like(x)
    for r in range(len(x)):
        fact = 1
        pol = 0
        
        numSum = 0
        i = 0
        
        cond = True
        while(cond):
            termino = x[r]**i / fact
            pol = pol + termino
            
            numSum = numSum + 1
            
            if(np.max(np.abs(x)) < tol or numSum == maxNumSum):
                cond = False
            
            
            fact = fact * (i+1)
            i = i+1
        
        y[r] = pol
    
    return y

plt.figure()
plt.plot(x,f(x),'y', linewidth = 4)
plt.plot(x,funExp(x,tol,maxNumSum),'b--')
plt.show()
