# Guia sobre Google Colaboration

El objetivo de esta guia es realizar una experimentación y análisis de las diferentes familias de complejidades, al mismo tiempo, utilizando una herramienta muy aplicada en la industria actualmente. En definitiva, armar y graficar algunas funciones.

## Introducción

Colab es una herramienta del Stack de Google que permite ejecutar código desde el navegador, con algunas características a destacar:
- IDE online y colaborativa en la nube
- Uso de librerías comunes del Stack de Ciencia de Datos
- Integración con Jupyter Notebooks
- Acceso "gratuito" a GPU

### IDE online y colaborativa
Desde [Google Colab](https://colab.research.google.com/), al igual que otras herramientas de Google, se puede editar y compartir los documentos creados entre diferentes usuaries.
Se considera una IDE porque permite escribir y ejecutar codigo en diferentes lenguajes de programacion, Python, Julia, JavaScript, entre otros, con una serie de librerias básicas instaladas por defecto. [Colab FAQ](https://research.google.com/colaboratory/faq.html)

### Uso de librerías comunes del Stack de Ciencia de Datos
Una de las caracteristicas principales de Colab es disponer de las librerias más conocidas en el mundo de la Ciencia de Datos en Python.
Las siguientes librerías están disponibles:
- [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/): generar modelos estadísticos de aprendizaje (AI, Machine Learning).
- [OpenCV](https://opencv.org/): generar modelos estadísticos basados en **computer vision**.
- [Pandas](https://pandas.pydata.org/): manejo de estructuras de datos eficientes.
- [NumPy](https://numpy.org/):  operaciones y algoritmos numéricos complejos.
- [MatPlotLib](https://matplotlib.org/): visualización y armado de gráficas.

### Integración con Jupyter Notebooks
[Google Colab](https://colab.research.google.com/) es una extensión de las famosas [Jupyter Notebooks](https://jupyter.org/) la cual le agrega la posibilidad de correr en la nube. Además, permite importar Notebooks previamente creadas.

### Acceso "gratuito" a GPU
Por último, pero no menor, Google Colab otorga el uso de una gran cantidad de recursos tanto de GPU como de RAM y CPU para ser utilizados por los programas corriendo en las Notebooks. Sin embargo, para mantener este servicio gratuito Google no garantiza la disponibilidad total de recursos en todo momento, esto puede variar dependiendo la hora, dia, pais, etc.

## Requerimientos

- Cuenta en Google, Gmail es suficiente

## Experimentación

### Proyecto Base

En este ejemplo vemos cómo crear una Notebook vacia con un programa en python.

- Creación de Notebook
![1-google-colab](https://user-images.githubusercontent.com/13955827/114624956-e9aed800-9c87-11eb-9adc-3746f4fb7a09.png)

- Algunos Ejemplos de código
![2-google-colab](https://user-images.githubusercontent.com/13955827/114624982-f59a9a00-9c87-11eb-8123-200a37790cc3.png)

![3-google-colab](https://user-images.githubusercontent.com/13955827/114625005-fe8b6b80-9c87-11eb-9644-54b574b5cefb.png)

- Codigo de ejemplo
```python
print("Hello World")

print(f"Jupyter Notebooks > Colab? {all([])}")
```

### Uso de MatPlotLib para graficar funciones

- Ejemplo de gráficos
![4-google-colab](https://user-images.githubusercontent.com/13955827/114625131-2d094680-9c88-11eb-9eba-e053a0ce2ce0.png)

- Codigo de ejemplo

```python
import matplotlib.pyplot as plt
 
x  = [1, 2, 3, 4, 5, 6, 7, 8, 9]
y1 = [1, 3, 5, 3, 1, 3, 5, 3, 1]
y2 = [2, 4, 6, 4, 2, 4, 6, 4, 2]
plt.plot(x, y1, label="line L")
plt.plot(x, y2, label="line H")
plt.plot()

plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Line Graph Example")
plt.legend()
plt.show()
```

### Gráficas "asintóticas" de funciones, Ejercicio 3 Práctica 1

- Resumen de las gráficas (Ver documento completo en [notebook](./1_complejidad.ipynb))

![5-google-colab](https://user-images.githubusercontent.com/13955827/114625972-6e4e2600-9c89-11eb-8f55-092d5e94aa9c.png)

```python
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

plt.plot(x, y1, label='1')
plt.plot(x, y5, label='sqrt(2)')

plt.xlabel('n')
plt.ylabel('costo')
plt.legend(loc=4)

# Parte Cuatro: 0(log n) = Acotadas por Funciones Logaritmicas
y8 = np.log2(x)
y10 = np.sqrt(x)
y14 = np.log2(x**2)

plt.plot(x, y8, label='log x')
plt.plot(x, y10, label='sqrt(x)')
plt.plot(x, y14, label='log x^2')

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

# Parte Seis: 0(n^2) y 0(2^n) = Acotadas por Funciones Cuadraticas y Exp
y7 = x**2
y16 = 2**x

plt.plot(x[1:10], y7[1:10], label='x^2')
plt.plot(x[1:10], y16[1:10], label='2^x')

plt.xlabel('n')
plt.ylabel('costo')
plt.legend()
```

### Referencias

- https://colab.research.google.com/
- https://research.google.com/colaboratory/faq.html
- https://jupyter.org/
- https://colab.research.google.com/notebooks/intro.ipynb#
- https://towardsdatascience.com/getting-started-with-google-colab-f2fff97f594c
