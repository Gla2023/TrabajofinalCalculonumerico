# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:57:30 2023

@author: glady
"""
import matplotlib.pyplot as plt
import numpy as np
def regula_falsi(func, a, b, tol=1e-6, max_iter=100):
    """
    Método de la Regla Falsi para encontrar la raíz de una función.
    
    Args:
        func (function): La función para la cual queremos encontrar la raíz.
        a (float): Extremo izquierdo del intervalo inicial.
        b (float): Extremo derecho del intervalo inicial.
        tol (float): Tolerancia para la convergencia (opcional).
        max_iter (int): Número máximo de iteraciones (opcional).
    
    Returns:
        float: La aproximación de la raíz de la función.
    """
    if func(a) * func(b) >= 0:
        raise ValueError("La función no cumple con el teorema de Bolzano en el intervalo dado.")
    
    iter_count = 0
    while iter_count < max_iter:
        c = (a * func(b) - b * func(a)) / (func(b) - func(a))
        
        if abs(func(c)) < tol:
            return c
        
        if func(c) * func(a) < 0:
            b = c
        else:
            a = c
        
        iter_count += 1
    
    raise Exception("El método de la Regla Falsa no convergió después de {} iteraciones.".format(max_iter))

def plot_regula_falsi_iterations(func, a, b, tol=1e-6, max_iter=100):
    x_vals = np.linspace(a, b, 1000)
    y_vals = func(x_vals)
    
    plt.plot(x_vals, y_vals, label='Función')
    
    iter_x_vals = []
    iter_y_vals = []
    
    try:
        root = regula_falsi(func, a, b, tol, max_iter)
        plt.scatter(root, func(root), color='red', marker='o', label='Aproximación de la raíz')
        
        # Graficar las iteraciones
        a_temp, b_temp = a, b
        for i in range(max_iter):
            c = (a_temp * func(b_temp) - b_temp * func(a_temp)) / (func(b_temp) - func(a_temp))
            
            iter_x_vals.append(c)
            iter_y_vals.append(func(c))
            
            if func(c) * func(a_temp) < 0:
                b_temp = c
            else:
                a_temp = c
        
        plt.scatter(iter_x_vals, iter_y_vals, color='green', marker='x', label='Iteraciones')
    
    except Exception as e:
        print(e)
    
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.title('Método de la Regla Falsi - Iteraciones')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

