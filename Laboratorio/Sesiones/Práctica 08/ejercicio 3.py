# -*- coding: utf-8 -*-
"""
ejercicio 3
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as pol
from scipy.integrate import quad

def aprox2(f,g,a,b):
    # 1. Construir el sistema (C y d)
    C = np.ones((g+1,g+1))
    d = np.ones((g+1,1))
    
    for i in range(len(C)):
        for j in range(len(C[0])):
            
    
    # 2. Resolver el sistema
    p = np.linalg.solve(C,d)
    
    # 3. Dibujar los puntos y el polinomio de ajuste
    

#-----------
# primer ejemplo
f = lambda x: np.sin(x)
a = 0.; b = 2.; g = 2
aprox2(f,g,a,b)








