# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 10:41:31 2023

@author: glady
"""

import sympy as sp
import matplotlib.pyplot as plt

def punto_fijo(g, x0, tol=1e-6, max_iter=100):
    """
    Método de iteración de punto fijo para encontrar raíces de una función.
    
    Parameters:
        g (sympy.Function): Función auxiliar g(x).
        x0 (float): Punto inicial.
        tol (float): Tolerancia para la convergencia.
        max_iter (int): Número máximo de iteraciones permitidas.
        
    Returns:
        float: Aproximación de la raíz.
        int: Número de iteraciones realizadas.
    """
    x = sp.symbols('x')
   
    g = sp.diff(g, x)
    
    iteration = 0
    iteraciones = []
    aproximaciones = []
    
    while iteration < max_iter:
        x1 = g.subs(x, x0)
        
        # Actualizar listas para la gráfica
        iteraciones.append(iteration + 1)
        aproximaciones.append(x1)
        
        if abs(x1 - x0) < tol:
            # Graficar las iteraciones
            plt.plot(iteraciones, aproximaciones, marker='o')
            plt.xlabel('Iteración')
            plt.ylabel('Aproximación de la raíz')
            plt.title('Convergencia del Método de Iteración de Punto Fijo')
            plt.grid(True)
            plt.show()

            return x1, iteration + 1
        
        x0 = x1
        iteration += 1
    
    # Graficar las iteraciones si no converge
    plt.plot(iteraciones, aproximaciones, marker='o')
    plt.xlabel('Iteración')
    plt.ylabel('Aproximación de la raíz')
    plt.title('Convergencia del Método de Iteración de Punto Fijo')
    plt.grid(True)
    plt.show()

    raise ValueError("El método de iteración de punto fijo no convergió en el número máximo de iteraciones.")


