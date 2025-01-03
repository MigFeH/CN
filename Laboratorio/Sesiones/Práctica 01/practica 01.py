# -*- coding: utf-8 -*-
"""
Ejemplos
"""
print('Hola mundo')
#%%
import numpy as np
print(np.pi)
#%%
import numpy as np
a = np.array([1, 2, 3, 4])
b = np.array([(1.5, 2, 3), (4, 5, 6)])
c = np.zeros((3, 4))
d = np.ones((2, 3))
e = np.arange(1, 10, 2)
f = np.arange(1., 10)
g = np.linspace(1, 9, 5)


for i in range(5):
    print(i)
    
#%%
import numpy as np
a = np.array([1, 2, 3, 4])
b = np.array([(1.5, 2, 3), (4, 5, 6)])

#%%
import numpy as np
a = np.array([1, 2, 3, 4])
b = np.array([(1.5, 2, 3, 5)])

a1 = [1, 2, 3, 4]
b1 = [1.5, 2, 3, 5]


A = np.array( [[1, 1], [0, 1]] )
B = np.array( [[2, 3], [1, 4]] )


#%%
import numpy as np
a = np.arange(10)

#%%
import numpy as np
n = 6
s = n*n
a = np.arange(s)
a = np.reshape(a,(n,n))

b = np.copy(a)
b[0,0] = 123

c = np.sin(a)

import math as math
d = math.sin(1.)

#%%
import numpy as np
a = np.arange(5)

f1 = lambda x: x**3
f2 = lambda x,y: x**2 + y**2

c = f1(a)
d = f2(a,a)

#%%
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-1,2)              # malla
f = lambda x : x**3 - 2*x**2 + 1   # función 
OX = 0*x                           # eje OX

plt.figure()
plt.plot(x,f(x))                   # dibujar la función
plt.plot(x,OX,'k-')                # dibujar el eje X
plt.xlabel('x')
plt.ylabel('y')
plt.title('función')
plt.show()

#%%
# Ejercicio 1
import numpy as np

a = np.array([1,3,7])
b = np.array([(2,4,3),(0,1,6)])
c = np.ones((1,3))
d = np.zeros((1,4))
e = np.zeros((3,2))
f = np.ones((3,4))

#%%
# Ejercicio 2
import numpy as np
np.set_printoptions(precision=2,suppress=True)

a = np.arange(7,16,2)
a = np.linspace(7,15,5)

b = np.arange(10,5,-1)
b = np.linspace(10,6,5)

c = np.arange(15,-1,-5)
c = np.linspace(15,0,4)

d = np.arange(0,1.1,0.1)
d = np.linspace(0,1,11)

e = np.arange(-1,1.1,0.2)
e = np.linspace(-1,1,11)

f = np.arange(1,2.1,0.1)
f = np.linspace(1,2,11)

#%%
# Ejercicio 3
import numpy as np

v = np.arange(0.0,12.2,1.1)

vi = np.copy(v)
vi = vi[::-1]

v1 = np.copy(v)
v2 = np.copy(v)

v1 = v1[::2]
v2 = v2[1::2]

v1 = np.copy(v)
v2 = np.copy(v)
v3 = np.copy(v)

v1 = v1[::3]
v2 = v2[1::3]
v3 = v3[2::3]

v1 = np.copy(v)
v2 = np.copy(v)
v3 = np.copy(v)
v4 = np.copy(v)

v1 = v1[::4]
v2 = v2[1::4]
v3 = v3[2::4]
v4 = v4[3::4]

#%%
# Ejercicio 4
import numpy as np

a = np.array([1,2,3])

b = np.copy(a)
b = np.append(b,0)
b = b[::-1]
b = np.append(b,0)
b = b[::-1]

