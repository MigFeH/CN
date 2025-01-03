#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Newton Cotes
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as pol
import sympy as sym
#-----------------------------------------------------
def dibujo(f,a,b,nodos):
    xp = np.linspace(a,b)
    plt.figure()
    # Area exacta
    plt.plot(xp,f(xp),'b', label = 'Área exacta')
    plt.plot([a,a,b,b],[f(a),0,0,f(b)],'b')
    
    # Puntos de interpolación
    plt.plot(nodos,f(nodos),'ro', label = 'Puntos de interpolación')
    
    # Area aproximada
    p = pol.polyfit(nodos, f(nodos), len(nodos)-1)
    yp = pol.polyval(xp,p)
    pa = pol.polyval(a,p) # yp[0]
    pb = pol.polyval(b,p) # yp[-1]
    
    plt.plot(xp,yp,'r--', label = 'Area aproximada')
    plt.plot([a,a,b,b],[pa,0,0,pb],'r--')
    
    plt.legend()
    plt.show()
    
#-----------------------------------------------------
def punto_medio(f,a,b,p=False):
    m = (a+b)/2
    Ia = (b-a) * f(m)
    if p:
        nodos = np.array([m])
        dibujo(f,a,b,nodos)
    return Ia

#-----------------------------------------------------
def trapecio(f,a,b,p=False):
    h = (b-a) / 2
    Ia = h * (f(a) + f(b))
    if p:
        nodos = np.array([f(a),f(b)])
        dibujo(f,a,b,nodos)
    return Ia

#-----------------------------------------------------
def simpson(f,a,b,p=False):
    h = (b-a) / 6
    Ia = h * (f(a) + 4*f((a+b) / 2) + f(b))
    #if p:
        #nodos = np.array([m])
        #dibujo(f,a,b,nodos)
    return Ia

#-----------------------------------------------------
def punto_medio_comp(f,a,b,n):
    h = (b-a) / n # longitud de cada subintervalo
    x_m = np.zeros(n)
    
    indice = 0
    suma = 0
    
    for i in range(1,n+1):
        
        # extremos de los subintervalos
        x_ant = a + (i-1) * h
        x_i = a + i*h
        
        # punto medio de los subintervalos
        x_m[indice] = ((x_ant + x_i) / 2)
        indice = indice + 1
        
    for i in range(len(x_m)):
        suma = suma + f(x_m[i])
        
    Ia = h * suma
    return Ia

#-----------------------------------------------------
def trapecio_comp(f,a,b,n):
    h = (b-a) / n # longitud de cada subintervalo
    suma = 0
    for i in range(1,n):
        x_i = a + i*h
        
        suma = suma + f(x_i)
    
    Ia = (h/2) * (f(a) + f(b)) + h * suma
    return Ia

#-----------------------------------------------------
def simpson_comp(f,a,b,n):
    h = (b-a) / n # longitud de cada subintervalo
    suma = 0
    for i in range(1,n+1):
        x_ant = a + (i-1)*h
        x_i = a + i*h
        x_m = (x_ant + x_i) / 2
        
        elemento = f(x_ant) + 4*f(x_m) + f(x_i)
        suma = suma + elemento
        
    Ia = (h/6) * suma
    return Ia
    
#----------------------------------------------------- 
#%%
def main():
    print('=======  Ejercicio 1 plot  =======')  
    # Ejemplo 1
    f = lambda x: np.exp(x)
    a = 0.; b = 3; nodos = np.array([1,2,2.5])
    dibujo(f,a,b,nodos)
    
    # Ejemplo 2
    f = lambda x: np.cos(x) + 1.5
    a = -3.; b = 3; nodos = np.array([-3.,-1,0,1,3])
    dibujo(f,a,b,nodos)
    
    # VALOR EXACTO
    x = sym.Symbol('x', real=True) 
    f_sym = sym.log(x)
    Ie = sym.integrate(f_sym,(x,1,3))
    Ie = float(Ie)
    
    # DATOS
    f = lambda x: np.log(x)
    a = 1; b = 3 
    
    
    print('\n=======  Ejercicio 1a  =======') 
    Ia = punto_medio(f,a,b,p=True)
    print('El valor aproximado es ',Ia)
    print('El valor exacto es     ',Ie)
    
    
    print('\n=======  Ejercicio 1b  =======') 
    Ia = trapecio(f,a,b,p=True)
    print('El valor aproximado es ',Ia)
    print('El valor exacto es     ',Ie)
    
    
    print('\n=======  Ejercicio 1c  =======') 
    Ia = simpson(f,a,b,p=True)
    print('El valor aproximado es ',Ia)
    print('El valor exacto es     ',Ie)
    
    
    print('\n=======  Ejercicio 1d  =======') 
    Ia = punto_medio_comp(f,a,b,5)
    print('El valor aproximado es ',Ia)
    print('El valor exacto es     ',Ie)
    
    
    print('\n=======  Ejercicio 1e  =======') 
    Ia = trapecio_comp(f,a,b,4)
    print('El valor aproximado es ',Ia)
    print('El valor exacto es     ',Ie)
    
    
    print('\n=======  Ejercicio 1f  =======') 
    Ia = simpson_comp(f,a,b,4)
    print('El valor aproximado es ',Ia)
    print('El valor exacto es     ',Ie)

    
if __name__ == "__main__":
    main()    
        