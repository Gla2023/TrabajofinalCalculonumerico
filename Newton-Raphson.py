# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 09:53:46 2023

@author: glady
"""

import numpy as np

def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    """
    Implementación del método de Newton-Raphson para encontrar una raíz de una función.

    Args:
    f: La función para la cual se busca la raíz.
    df: La derivada de la función f.
    x0: Aproximación inicial.
    tol: Tolerancia para la convergencia.
    max_iter: Número máximo de iteraciones.

    Returns:
    x: La aproximación de la raíz.
    """
    x = x0
    for i in range(max_iter):
        x_new = x - f(x) / df(x)
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    raise Exception("El método de Newton-Raphson no convergió después de {} iteraciones.".format(max_iter))

# Ejemplo de uso:
if __name__ == "__main__":
    # Define la función y su derivada
    def f(x):
        return x**2 - 2

    def df(x):
        return 2*x

    # Valor inicial
    x0 = 3.0

    # Encuentra una raíz
    root = newton_raphson(f, df, x0)
    print("Raíz encontrada:", root)
