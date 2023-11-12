# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 10:41:31 2023

@author: glady
"""

import sympy as sp

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
    gx = g(x)
    g_prime = sp.diff(gx, x)
    
    iteration = 0
    while iteration < max_iter:
        x1 = g.subs(x, x0)
        
        if abs(x1 - x0) < tol:
            return x1, iteration + 1
        
        x0 = x1
        iteration += 1
    
    raise ValueError("El método de iteración de punto fijo no convergió en el número máximo de iteraciones.")

# Ejemplo de uso
if __name__ == "__main__":
    # Define la función g(x)
    x = sp.symbols('x')
    g = sp.sin(x)
    
    # Punto inicial
    x0 = 1.0
    
    # Llama al método de punto fijo
    root, iterations = punto_fijo(g, x0)
    
    # Imprime los resultados
    print(f"Aproximación de la raíz: {root}")
    print(f"Número de iteraciones: {iterations}")
