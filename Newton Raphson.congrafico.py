# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:58:23 2023

@author: glady
"""

import sympy as sp
import matplotlib.pyplot as plt

def newton_raphson(func, x0, tol=1e-6, max_iter=100):
    """
    Método de Newton-Raphson para encontrar la raíz de una función.
    
    Args:
        func (sympy.Function): La función para la cual queremos encontrar la raíz.
        x0 (float): Punto inicial.
        tol (float): Tolerancia para la convergencia (opcional).
        max_iter (int): Número máximo de iteraciones (opcional).
    
    Returns:
        float: La aproximación de la raíz.
    """
    x = sp.symbols('x')
    f_prime = sp.diff(func, x)
    
    iteration = 0
    iteraciones = []
    aproximaciones = []

    while iteration < max_iter:
        fx = func.subs(x, x0)
        f_prime_x = f_prime.subs(x, x0)

        if abs(fx) < tol:
            # Graficar las iteraciones
            plt.plot(iteraciones, aproximaciones, marker='o')
            plt.xlabel('Iteración')
            plt.ylabel('Aproximación de la raíz')
            plt.title('Convergencia del Método de Newton-Raphson')
            plt.grid(True)
            plt.show()

            return x0

        x1 = x0 - fx / f_prime_x
        
        # Actualizar listas para la gráfica
        iteraciones.append(iteration + 1)
        aproximaciones.append(x1)

        if abs(x1 - x0) < tol:
            # Graficar las iteraciones
            plt.plot(iteraciones, aproximaciones, marker='o')
            plt.xlabel('Iteración')
            plt.ylabel('Aproximación de la raíz')
            plt.title('Convergencia del Método de Newton-Raphson')
            plt.grid(True)
            plt.show()

            return x1

        x0 = x1
        iteration += 1
    
    # Graficar las iteraciones si no converge
    plt.plot(iteraciones, aproximaciones, marker='o')
    plt.xlabel('Iteración')
    plt.ylabel('Aproximación de la raíz')
    plt.title('Convergencia del Método de Newton-Raphson')
    plt.grid(True)
    plt.show()

    raise ValueError("El método de Newton-Raphson no convergió en el número máximo de iteraciones.")

# Ejemplo de uso
if __name__ == "__main__":
    # Define la función f(x)
    x = sp.symbols('x')
    f = x**2 - 4
    
    # Punto inicial
    x0 = 1.0
    
    # Llama al método de Newton-Raphson
    root = newton_raphson(f, x0)
    
    # Imprime el resultado
    print(f"Aproximación de la raíz: {root}")
