# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 09:59:48 2023

@author: glady
"""

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

    for i in range(max_iter):
        x = (a + b) / 2
        if abs(f(x)) < tol:
            return x
        if f(x) * f(a) < 0:
            b = x
        else:
            a = x

    raise Exception("El método de la bisección no convergió después de {} iteraciones.".format(max_iter))

# Ejemplo de uso:
if __name__ == "__main__":
    # Define la función
    def f(x):
        return x*x*x-13*x+18

    # Intervalo inicial
    a = -8
    b = 8

    # Encuentra una raíz
    root = bisection(f, a, b)
    print("Raíz encontrada:", root)
