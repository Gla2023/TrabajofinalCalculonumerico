# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:59:16 2023

@author: glady
"""

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
    while iter_count < max_iter:
        fx0 = func(x0)
        fx1 = func(x1)
        if abs(fx1) < tol:
            return x1
