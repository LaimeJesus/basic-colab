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

En este ejemplo veremos cómo crear una Notebook vacia, y agregaremos algunos ejemplos.

- Creación de Notebook

- Algunos Ejemplos de código

```python
print("Hello World")

print(f"Jupyter Notebooks > Colab? {all([])}")

```

### Uso de MatPlotLib para graficar funciones

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

### Referencias

- https://colab.research.google.com/
- https://research.google.com/colaboratory/faq.html
- https://jupyter.org/
- https://colab.research.google.com/notebooks/intro.ipynb#
- https://towardsdatascience.com/getting-started-with-google-colab-f2fff97f594c


