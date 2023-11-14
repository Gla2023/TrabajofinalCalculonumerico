# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:59:16 2023

@author: glady
"""
import matplotlib.pyplot as plt

def secant_method(func, x0, x1, tol=1e-6, max_iter=100):
    """
    Método de la secante para encontrar la raíz de una función.
    
    Args:
        func (function): La función para la cual queremos encontrar la raíz.
        x0 (float): Primer punto inicial.
        x1 (float): Segundo punto inicial.
        tol (float): Tolerancia para la convergencia (opcional).
        max_iter (int): Número máximo de iteraciones (opcional).
    
    Returns:
        float: La aproximación de la raíz de la función.
    """
    iter_count = 0
    iteraciones = []
    aproximaciones = []

    while iter_count < max_iter:
        fx0 = func(x0)
        fx1 = func(x1)

        if abs(fx1) < tol:
            # Graficar las iteraciones
            plt.plot(iteraciones, aproximaciones, marker='o')
            plt.xlabel('Iteración')
            plt.ylabel('Aproximación de la raíz')
            plt.title('Convergencia del Método de la Secante')
            plt.grid(True)
            plt.show()

            return x1

        x_next = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        iter_count += 1

        # Actualizar listas para la gráfica
        iteraciones.append(iter_count)
        aproximaciones.append(x_next)

        x0, x1 = x1, x_next

    # Graficar las iteraciones si no converge
    plt.plot(iteraciones, aproximaciones, marker='o')
    plt.xlabel('Iteración')
    plt.ylabel('Aproximación de la raíz')
    plt.title('Convergencia del Método de la Secante')
    plt.grid(True)
    plt.show()

    raise Exception("El método de la secante no convergió después de {} iteraciones.".format(max_iter))

# Ejemplo de uso
def ejemplo_funcion(x):
    return x**2 - 4

raiz_secante = secant_method(ejemplo_funcion, 0, 3)
print("Aproximación de la raíz:", raiz_secante)

