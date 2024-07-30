import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def runge_kutta_4(f, a, b, y0, m):
    """
    Implementação do método de Runge-Kutta de 4ª ordem para uma EDO.
    
    :param f: Função que representa a EDO dy/dx = f(x, y)
    :param a: Limite inferior do intervalo
    :param b: Limite superior do intervalo
    :param y0: Valor inicial y(a)
    :param m: Número de subintervalos
    :return: Vetores com as abscissas (x) e as soluções (y)
    """
    h = (b - a) / m
    x = np.linspace(a, b, m+1)
    y = np.zeros(m+1)
    y[0] = y0
    
    for i in range(m):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h / 2, y[i] + k1 / 2)
        k3 = h * f(x[i] + h / 2, y[i] + k2 / 2)
        k4 = h * f(x[i] + h, y[i] + k3)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return x, y

# Exemplo de uso
def f(x, y):
    return x - 2*y + 1

a = 0
b = 1
y0 = 1
m = 10

x, y = runge_kutta_4(f, a, b, y0, m)

# Imprimindo a tabela de resultados
data = {'x': x, 'y': y}
df = pd.DataFrame(data)
print(df)

# Gráfico
plt.plot(x, y, label='Aproximação Runge-Kutta')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Método de Runge-Kutta de 4ª Ordem para uma EDO')
plt.legend()
plt.grid(True)
plt.show()
