import numpy as np
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt

def sistema_eq(Y, y_prev, z_prev, x_next, h, f, g):
    y_next, z_next = Y
    eq1 = y_next - y_prev - h * f(x_next, y_next, z_next)
    eq2 = z_next - z_prev - h * g(x_next, y_next, z_next)
    return [eq1, eq2]

def euler_implicito_sistema(f, g, a, b, y0, z0, h):
    # Número de passos
    N = int((b - a) / h)
    x = np.linspace(a, b, N+1)
    y = np.zeros(N+1)
    z = np.zeros(N+1)
    y[0] = y0
    z[0] = z0
    
    for i in range(N):
        x_next = x[i+1]
        y_prev = y[i]
        z_prev = z[i]
        Y_next = opt.root(sistema_eq, [y_prev, z_prev], args=(y_prev, z_prev, x_next, h, f, g)).x
        y[i+1] = Y_next[0]
        z[i+1] = Y_next[1]
    
    return x, y, z

# Definindo as funções f(x, y, z) e g(x, y, z)
def f(x, y, z):
    return -2 * y + z

def g(x, y, z):
    return -y + 3 * z

# Parâmetros de entrada
a = 0
b = 1
y0 = 1
z0 = 1
h = 0.1

# Solucionando o sistema de EDOs
x, y, z = euler_implicito_sistema(f, g, a, b, y0, z0, h)

# Criando a tabela de resultados
df = pd.DataFrame({'x': x, 'y': y, 'z': z})
print(df)

# Plotando os gráficos
plt.plot(x, y, marker='o', linestyle='-', color='r', label='y (Euler Implícito)')
plt.plot(x, z, marker='x', linestyle='-', color='g', label='z (Euler Implícito)')
plt.xlabel('x')
plt.ylabel('Valores de y e z')
plt.title('Método de Euler Implícito para um Sistema de duas EDOs')
plt.legend()
plt.grid(True)
plt.show()
