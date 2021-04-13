# Parte Uno: Importar librerias
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.special

# Parte Dos: Definir Intervalo de Entradas
x = np.array(range(2, 100))

# Parte Tres: 0(1) = Acotadas por Funciones Constantes
y1 = np.full((x.size), 1)
y5 = np.full((x.size), np.sqrt(2))
y13 = 1/x
y15 = 1 + (np.sin(x))**2

plt.plot(x, y1, label='1')
plt.plot(x, y5, label='sqrt(2)')
plt.plot(x, y13, label='1/x')
plt.plot(x, y15, label='1 + sin^2 x')

plt.xlabel('n')
plt.ylabel('costo')
plt.legend(loc=4)

# Parte Cuatro: 0(log n) = Acotadas por Funciones Logaritmicas
y3 = np.log2(np.log2(x))
y8 = np.log2(x)
y10 = np.sqrt(x)
y11 = (np.log(x))**2
y14 = np.log2(x**2)

plt.plot(x, y3, label='log log x')
plt.plot(x, y8, label='log x')
plt.plot(x, y10, label='sqrt(x)')
plt.plot(x, y14, label='log x^2')
plt.plot(x, y11, label='(log x)^2')

plt.xlabel('n')
plt.ylabel('costo')
plt.legend()

# Parte Cinco: 0(n) = Acotadas por Funciones Lineales

y4 = x+1
y12 = x*np.log(x)
y9 = np.log2(sp.special.factorial(x))

plt.plot(x, y4, label='x + 1')
plt.plot(x, y12, label='x log x')
plt.plot(x, y9, label='log x!')

plt.xlabel('n')
plt.ylabel('costo')
plt.legend()

# Parte Seis: 0(n^2) = Acotadas por Funciones Cuadraticas
y7 = x**2
y16 = 2**x

plt.plot(x, y7, label='x^2')

plt.xlabel('n')
plt.ylabel('costo')
plt.legend()

plt.plot(x[1:10], y7[1:10], label='x^2')
plt.plot(x[1:10], y16[1:10], label='2^x')

plt.xlabel('n')
plt.ylabel('costo')
plt.legend()

# Parte Siete: 0(2^n) = Acotadas por Funciones Exponenciales
y2 = x**x
y6 = sp.special.factorial(x)

plt.plot(x, y6, label='x!')

plt.xlabel('n')
plt.ylabel('costo')
plt.legend()

plt.plot(x, y2, label='x^x')

plt.xlabel('n')
plt.ylabel('costo')
plt.legend()
