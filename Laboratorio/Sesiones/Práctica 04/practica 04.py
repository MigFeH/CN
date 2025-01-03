# -*- coding: utf-8 -*-
"""
Gauss
"""
import numpy as np
np.set_printoptions(precision = 2)   # solo dos decimales
np.set_printoptions(suppress = True) # no usar notación exponencial


def triangulariza(A,b):
    At = np.copy(A)
    bt = np.copy(b)
    
    for i in range(1,len(b)):
        pivote = At[i-1][i-1]
        numerador = At[i][i-1]
        
        f = numerador/pivote
        
        At[i][i-1] = At[i][i-1] - f*pivote
        At[i][i] = At[i][i] - f*At[i-1][i]
        
        bt[i] = bt[i] - f*bt[i-1]
    
    return At, bt


#------------------------
def sust_reg(At,bt):
    x = np.zeros_like(bt)
    
    x[len(bt)-1] = bt[-1]/At[-1][-1]
    
    for i in range(len(bt)-2,-1,-1):
        numerador = bt[i] - At[i][i+1]*x[i+1]
        denominador = At[i][i]
        
        x[i] = numerador/denominador
    
    return x


#----------------------
def triangulariza2(Ar,b):
    At = np.copy(Ar)
    bt = np.copy(b)
    
    for i in range(1,len(b)):
        pivote = At[i-1][1]
        numerador = At[i][0]
        
        f = numerador/pivote
        
        At[i][0] = numerador - f*pivote
        At[i][1] = At[i][1] - f*At[i-1][2]
        bt[i] = bt[i] - f*bt[i-1]
        
    
    return At, bt


#----------------------
def sust_reg2(At,bt):
    n = len(bt)
    x = np.zeros(n)
    
    x[n-1] = bt[n-1] / At[n-1][1]
    for k in range(n-2,-1,-1):
        x[k] = (bt[k] - At[k][2] * x[k+1]) / At[k][1]
    
    return x
    


print('EJERCICIO 1')
print('-------------  DATOS  -------------')
n = 7 

A1 = np.diag(np.ones(n))*3
A2 = np.diag(np.ones(n-1),1) 
A = A1 + A2 + A2.T 

b = np.arange(n,2*n)*1.

print('A')
print(A)
print('b')
print(b)


print('----  SISTEMA TRIANGULARIZADO ----')
At,bt = triangulariza(A,b)
print('At')
print(At)
print('bt')
print(bt)


x = sust_reg(At,bt)
print('X')
print(x)

print()
print()
print()
print('EJERCICIO 2')
print('-------------  DATOS  -------------')
n = 8 

np.random.seed(3)
A1 = np.diag(np.random.rand(n))
A2 = np.diag(np.random.rand(n-1),1)
A = A1 + A2 + A2.T 

b = np.random.rand(n)

print('A')
print(A)
print('b')
print(b)


print('----  SISTEMA TRIANGULARIZADO ----')
At,bt = triangulariza(A,b)
print('At')
print(At)
print('bt')
print(bt)

x = sust_reg(At,bt)
print('X')
print(x)


print()
print()
print()
print('EJERCICIO 3')
print('-------------  DATOS  -------------')
n = 7 

Ar = np.zeros((n,3))
Ar[:,0] = np.concatenate((np.array([0]),np.ones((n-1),)))
Ar[:,1] = np.ones((n),)*3
Ar[:,2] = np.concatenate((np.ones((n-1),),np.array([0])))

b = np.arange(n,2*n)*1.

print('Ar')
print(Ar)
print('b')
print(b)

print('----  SISTEMA TRIANGULARIZADO ----')
At,bt = triangulariza2(Ar,b)
print('At')
print(At)
print('bt')
print(bt)


x = sust_reg2(At,bt)
print('X')
print(x)


print()
print()
print('-------------  DATOS  -------------')
n = 8

np.random.seed(3)
Ar = np.zeros((n,3))
Ar[:,1] = np.random.rand(n)
Ar[:,0] = np.concatenate((np.array([0]),np.random.rand(n-1)))
Ar[0:n-1,2] = Ar[1:n,0]

b = np.random.rand(n)

print('Ar')
print(Ar)
print('b')
print(b)

print('----  SISTEMA TRIANGULARIZADO ----')
At,bt = triangulariza2(Ar,b)
print('At')
print(At)
print('bt')
print(bt)


x = sust_reg2(At,bt)
print('X')
print(x)
