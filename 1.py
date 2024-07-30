import numpy as np
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt

def euler_implicito_edo(f, a, b, y0, h):
    # Número de passos
    N = int((b - a) / h)
    x = np.linspace(a, b, N+1)
    y = np.zeros(N+1)
    y[0] = y0
    
    for i in range(N):
        func = lambda y_next: y_next - y[i] - h * f(x[i+1], y_next)
        y[i+1] = opt.newton(func, y[i])
    
    return x, y

# Definindo a função f(x, y)
def f(x, y):
    return x - 2*y + 1

# Parâmetros de entrada
a = 0
b = 1
y0 = 1
h = 0.1

# Solucionando a EDO
x, y = euler_implicito_edo(f, a, b, y0, h)

# Criando a tabela de resultados
df = pd.DataFrame({'x': x, 'y': y})
print(df)

# Plotando o gráfico
plt.plot(x, y, marker='o', linestyle='-', color='b', label='Euler Implícito')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Método de Euler Implícito para uma EDO')
plt.legend()
plt.grid(True)
plt.show()
