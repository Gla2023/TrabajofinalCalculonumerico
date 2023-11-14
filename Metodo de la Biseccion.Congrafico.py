# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 09:59:48 2023

@author: glady
"""
 
import matplotlib.pyplot as plt

def bisection(f, a, b, tol=1e-6, max_iter=100):
    """
    Implementación del método de la bisección para encontrar una raíz de una función.

    Args:
    f: La función para la cual se busca la raíz.
    a: Extremo izquierdo del intervalo inicial.
    b: Extremo derecho del intervalo inicial.
    tol: Tolerancia para la convergencia.
    max_iter: Número máximo de iteraciones.

    Returns:
    x: La aproximación de la raíz.
    """

    if f(a) * f(b) >= 0:
        raise Exception("La función no cumple con el teorema del valor intermedio en el intervalo dado.")

    # Listas para almacenar datos de las iteraciones
    iteraciones = []
    aproximaciones = []

    for i in range(max_iter):
        x = (a + b) / 2
        iteraciones.append(i + 1)
        aproximaciones.append(x)

        if abs(f(x)) < tol:
            # Graficar las iteraciones
            plt.plot(iteraciones, aproximaciones, marker='o')
            plt.xlabel('Iteración')
            plt.ylabel('Aproximación de la raíz')
            plt.title('Convergencia del Método de la Bisección')
            plt.grid(True)
            plt.show()
            
            return x

        if f(x) * f(a) < 0:
            b = x
        else:
            a = x

    # Graficar las iteraciones si no converge
    plt.plot(iteraciones, aproximaciones, marker='o')
    plt.xlabel('Iteración')
    plt.ylabel('Aproximación de la raíz')
    plt.title('Convergencia del Método de la Bisección')
    plt.grid(True)
    plt.show()

    raise Exception("El método de la bisección no convergió después de {} iteraciones.".format(max_iter))

# Ejemplo de uso
def ejemplo_funcion(x):
    return x**2 - 4

raiz = bisection(ejemplo_funcion, 0, 3)
print("Aproximación de la raíz:", raiz)

