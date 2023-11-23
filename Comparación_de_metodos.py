import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

def punto_fijo(g, x0, tol=1e-6, max_iter=100):
    x = sp.symbols('x')
    g_prime = sp.diff(g, x)

    iteration = 0
    iteraciones = []
    aproximaciones = []

    while iteration < max_iter:
        x1 = x0 - g.subs(x, x0) / g_prime.subs(x, x0)

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


def newton_raphson(func, x0, tol=1e-6, max_iter=100):
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


def secant_method(func, x0, x1, tol=1e-6, max_iter=100):
    x = sp.symbols('x')

    iter_count = 0
    iteraciones = []
    aproximaciones = []

    while iter_count < max_iter:
        fx0 = func.subs(x, x0)
        fx1 = func.subs(x, x1)

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


def bisection(f, a, b, tol=1e-6, max_iter=100):
    x = sp.symbols('x')

    if f.subs(x, a) * f.subs(x, b) >= 0:
        raise Exception("La función no cumple con el teorema del valor intermedio en el intervalo dado.")

    # Listas para almacenar datos de las iteraciones
    iteraciones = []
    aproximaciones = []

    for i in range(max_iter):
        x_val = (a + b) / 2
        iteraciones.append(i + 1)
        aproximaciones.append(x_val)

        if abs(f.subs(x, x_val)) < tol:
            # Graficar las iteraciones
            plt.plot(iteraciones, aproximaciones, marker='o')
            plt.xlabel('Iteración')
            plt.ylabel('Aproximación de la raíz')
            plt.title('Convergencia del Método de la Bisección')
            plt.grid(True)
            plt.show()

            return x_val

        if f.subs(x, x_val) * f.subs(x, a) < 0:
            b = x_val
        else:
            a = x_val

    # Graficar las iteraciones si no converge
    plt.plot(iteraciones, aproximaciones, marker='o')
    plt.xlabel('Iteración')
    plt.ylabel('Aproximación de la raíz')
    plt.title('Convergencia del Método de la Bisección')
    plt.grid(True)
    plt.show()

    raise Exception("El método de la bisección no convergió después de {} iteraciones.".format(max_iter))


def regula_falsi(func, a, b, tol=1e-6, max_iter=100):
    x = sp.symbols('x')

    if func.subs(x, a) * func.subs(x, b) >= 0:
        raise Exception("La función no cumple con el teorema del valor intermedio en el intervalo dado.")

    iter_count = 0
    iteraciones = []
    aproximaciones = []

    while iter_count < max_iter:
        c = (a * func.subs(x, b) - b * func.subs(x, a)) / (func.subs(x, b) - func.subs(x, a))

        if abs(func.subs(x, c)) < tol:
            return c

        if func.subs(x, c) * func.subs(x, a) < 0:
            b = c
        else:
            a = c

        iter_count += 1

    raise Exception("El método de la Regla Falsa no convergió después de {} iteraciones.".format(max_iter))


def plot_regula_falsi_iterations(func, a, b, tol=1e-6, max_iter=100):
    x = sp.symbols('x')
    x_vals = np.linspace(float(a), float(b), 1000)
    y_vals = [func.subs(x, val) for val in x_vals]

    plt.plot(x_vals, y_vals, label='Función')

    iter_x_vals = []
    iter_y_vals = []

    try:
        root = regula_falsi(func, a, b, tol, max_iter)
        plt.scatter(float(root), float(func.subs(x, root)), color='red', marker='o', label='Aproximación de la raíz')

        # Graficar las iteraciones
        a_temp, b_temp = float(a), float(b)
        for i in range(max_iter):
            c = (a_temp * func.subs(x, b_temp) - b_temp * func.subs(x, a_temp)) / (
                        func.subs(x, b_temp) - func.subs(x, a_temp))

            iter_x_vals.append(float(c))
            iter_y_vals.append(float(func.subs(x, c)))

            if func.subs(x, c) * func.subs(x, a_temp) < 0:
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


def calcular_error_relativo(aproximacion, real):
    return np.abs(aproximacion - real) / np.abs(real)


def comparar_metodos(funcion, derivada, x_inicial, x_real, iteraciones):
    resultados = []

    # Regula Falsi
    rf_resultado = regula_falsi(funcion, x_inicial, x_real, iteraciones)
    rf_error = calcular_error_relativo(rf_resultado, x_real)
    resultados.append(("Regula Falsi", rf_resultado, rf_error))

    # Bisección
    b_resultado = bisection(funcion, x_inicial, x_real, iteraciones)
    b_error = calcular_error_relativo(b_resultado, x_real)
    resultados.append(("Bisección", b_resultado, b_error))

    # Secante
    sec_resultado = secant_method(funcion, x_inicial, x_real, iteraciones)
    sec_error = calcular_error_relativo(sec_resultado, x_real)
    resultados.append(("Secante", sec_resultado, sec_error))

    # Punto Fijo
    pf_resultado = punto_fijo(funcion, x_inicial, iteraciones)
    pf_error = calcular_error_relativo(pf_resultado, x_real)
    resultados.append(("Punto Fijo", pf_resultado, pf_error))

    # Newton-Raphson
    nr_resultado = newton_raphson(funcion, derivada, x_inicial, iteraciones)
    nr_error = calcular_error_relativo(nr_resultado, x_real)
    resultados.append(("Newton-Raphson", nr_resultado, nr_error))

    return resultados


# Función de ejemplo: f(x) = x^2 - 2
x = sp.symbols('x')
funcion_ejemplo = x**2 - 2

# Derivada de la función de ejemplo: f'(x) = 2x
derivada_ejemplo = sp.diff(funcion_ejemplo, x)

# Parámetros
x_inicial = 1.0
x_real = np.sqrt(2)
iteraciones = 20

# Comparar métodos
resultados = comparar_metodos(funcion_ejemplo, derivada_ejemplo, x_inicial, x_real, iteraciones)

# Mostrar resultados
for metodo, aproximacion, error in resultados:
    print(f"{metodo}: Aproximación = {aproximacion}, Error Relativo = {error}")




