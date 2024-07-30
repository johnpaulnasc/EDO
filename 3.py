import numpy as np
import matplotlib.pyplot as plt

def runge_kutta_4th_order_single(f, a, b, m, y0):
    h = (b - a) / m
    x = a
    y = y0
    
    VetX = np.zeros(m + 1)
    VetY = np.zeros(m + 1)
    
    VetX[0] = x
    VetY[0] = y
    
    for i in range(1, m + 1):
        k1 = f(x, y)
        k2 = f(x + h / 2, y + h / 2 * k1)
        k3 = f(x + h / 2, y + h / 2 * k2)
        k4 = f(x + h, y + h * k3)
        
        x = x + h
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        VetX[i] = x
        VetY[i] = y
    
    return VetX, VetY

# Função definida pela EDO y' = y - x^2 + 1
def f(x, y):
    return y - x**2 + 1

# Parâmetros
a = 0       # Limite inferior
b = 1       # Limite superior
m = 10      # Número de subintervalos
y0 = 1    # Valor inicial y(a) = 0.5

# Resolvendo a EDO
VetX, VetY = runge_kutta_4th_order_single(f, a, b, m, y0)

# Plotando o resultado
plt.plot(VetX, VetY, label='Runge-Kutta de 4ª ordem')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solução da EDO usando Runge-Kutta de 4ª ordem')
plt.legend()
plt.grid(True)
plt.show()
