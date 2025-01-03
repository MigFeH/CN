#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Degree of precision
"""
import numpy as np
import sympy as sym
from L9_ex1 import punto_medio, trapecio, simpson
from L9_ex2 import gauss
#-----------------------------------------------------
def newton_cotes(f,a,b,n):
    if n == 1:
        return punto_medio(f,a,b)
    elif n == 2:
        return trapecio(f,a,b)
    elif n == 3:
        return simpson(f,a,b)
    else:
        return 0

#-----------------------------------------------------
def grado_de_precision(formula,n):
    error = 0
    exponente = 0
    while error < 10**-10:    
        x = sym.Symbol('x', real=True) 
        I_exacta = sym.integrate(x**exponente,(x,1,3))
        I_exacta = float(I_exacta)
        f = lambda x: x**exponente
        Ia = formula(f,1,3,n)
        error = np.abs(I_exacta - Ia)
        
        print('f(x)^',exponente,'\t','error = ', error)
        
        exponente = exponente + 1
    print('El grado de precisión es ', exponente - 2)
#-----------------------------------------------------
print('----  Fórmula del punto medio (n = 1) ----\n') 
grado_de_precision(newton_cotes,1)

print('\n----  Fórmula del trapecio (n = 2) ----\n')       
grado_de_precision(newton_cotes,2) 

print('\n----  Fórmula de Simpson (n = 3) ----\n')  
grado_de_precision(newton_cotes,3)   

print('\n----  Fórmula Gauss n = 1  ----\n')
grado_de_precision(gauss,1)

print('\n----  Fórmula Gauss n = 2  ----\n')
grado_de_precision(gauss,2)

print('\n----  Fórmula Gauss n = 3  ----\n')
grado_de_precision(gauss,3)

print('\n----  Fórmula Gauss n = 4  ----\n')
grado_de_precision(gauss,4)