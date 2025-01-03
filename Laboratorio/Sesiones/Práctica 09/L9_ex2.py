#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaussian
"""
import numpy as np
import sympy as sym
from L9_ex1 import dibujo
#-----------------------------------------------------
def gauss(f,a,b,n,p=False):
    nodos = np.zeros(n)
    h = (b-a) / 2
    suma = 0
    [x,w] = np.polynomial.legendre.leggauss(n)
    for i in range(n):
        w_i = w[i]
        x_i = x[i]
        
        y_i = (h * x_i) + ((a+b) / 2)
        
        nodos[i] = y_i
        suma = suma + (w_i * f(y_i))
        
    Ia = h*suma
    
    if p:
        dibujo(f,a,b,nodos)

    return Ia

#----------------------------------------------------- 
#%%
def main():
    
    # VALOR EXACTO
    x = sym.Symbol('x', real=True) 
    f_sym = sym.log(x)
    Ie = sym.integrate(f_sym,(x,1,3))
    Ie = float(Ie)
    
    # DATOS
    f = lambda x: np.log(x)
    a = 1; b = 3 
    
    for n in range(1,4):
        Ia = gauss(f,a,b,n,p=True)
        if n == 1:
            print('\n',n,' nodo')
        else:    
            print('\n',n,' nodos')
        print('El valor aproximado es ',Ia)
        print('El valor exacto es     ',Ie)
        
if __name__ == "__main__":
    main()        
