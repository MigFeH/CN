# -*- coding: utf-8 -*-
"""
Ejercicio 1
"""
import numpy as np
np.set_printoptions(precision = 2) 
np.set_printoptions(suppress = True) 

def trisym(d,c,b):
    ct = np.copy(c)
    bt = np.copy(b)
    dt = np.copy(d)
    
    # K = 1
    f = ct[0]
    ct[0] = ct[0] / dt[0]
    bt[0] = bt[0] / dt[0]
    
    
    for i in range(1,len(bt)-1):
        dt[i] = dt[i] - (f * ct[i-1])
        bt[i] = bt[i] - (f * bt[i-1])
        
        f = ct[i]
        ct[i] = ct[i] / dt[i]
        bt[i] = bt[i] / dt[i]
    
    # K final
    dt[-1] = dt[-1] - (f * ct[-2])
    bt[-1] = bt[-1] - (f * bt[-2])
    
    bt[-1] = bt[-1] / dt[-1]
    return ct, bt    

print('-------------  DATOS 1  -------------')
n = 7 

d = np.ones(n)*3
c = np.ones(n-1)
b = np.arange(n,2*n)*1.

print('d')
print(d)
print('c')
print(c)

A1 = np.diag(d)
A2 = np.diag(c,1) 
A = A1 + A2 + A2.T 


print('\nA')
print(A)
print('b')
print(b)


print('\n-------  SISTEMA TRIANGULAR  1 -------') 
ct1, bt1 = trisym(d,c,b) 
print('ct1')
print(ct1)


print('\nAt1')
dt1 = np.ones(n)*1.
A1 = np.diag(dt1)
A2 = np.diag(ct1,1) 
At1 = A1 + A2 
print(At1)

print('bt1')
print(bt1) 


print('\n\n-------------  DATOS 2  -------------')
n = 8 

np.random.seed(3)
d = np.random.rand(n)
c = np.random.rand(n-1)
b = np.random.rand(n)

print('d')
print(d)
print('c')
print(c)

A1 = np.diag(d)
A2 = np.diag(c,1)
A = A1 + A2 + A2.T 

print('\nA')
print(A)
print('b')
print(b)


print('\n-------  SISTEMA TRIANGULAR 2 -------')  
ct2, bt2 = trisym(d,c,b) 
print('ct2')
print(ct2)


print('\nAt2')
dt2 = np.ones(n)*1.
A1 = np.diag(dt2)
A2 = np.diag(ct2,1) 
At2 = A1 + A2  
print(At2)

print('bt2')
print(bt2) 



