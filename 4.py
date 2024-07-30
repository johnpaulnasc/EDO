import numpy as np
import matplotlib.pyplot as plt

def runge_kutta_4th_order_system(f, g, a, b, m, y0, z0):
    h = (b - a) / m
    x = a
    y = y0
    z = z0
    
    VetX = np.zeros(m + 1)
    VetY = np.zeros(m + 1)
    VetZ = np.zeros(m + 1)
    
    VetX[0] = x
    VetY[0] = y
    VetZ[0] = z
    
    for i in range(1, m + 1):
        k1_y = f(x, y, z)
        k1_z = g(x, y, z)
        
        k2_y = f(x + h / 2, y + h / 2 * k1_y, z + h / 2 * k1_z)
        k2_z = g(x + h / 2, y + h / 2 * k1_y, z + h / 2 * k1_z)
        
        k3_y = f(x + h / 2, y + h / 2 * k2_y, z + h / 2 * k2_z)
        k3_z = g(x + h / 2, y + h / 2 * k2_y, z + h / 2 * k2_z)
        
        k4_y = f(x + h, y + h * k3_y, z + h * k3_z)
        k4_z = g(x + h, y + h * k3_y, z + h * k3_z)
        
        x = x + h
        y = y + h / 6 * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
        z = z + h / 6 * (k1_z + 2 * k2_z + 2 * k3_z + k4_z)
        
        VetX[i] = x
        VetY[i] = y
        VetZ[i] = z
    
    return VetX, VetY, VetZ

# Funções definidas pelas EDOs y' = y - x^2 + 1 e z' = z + y + x
def f(x, y, z):
    return y - x**2 + 1

def g(x, y, z):
    return z + y + x

# Parâmetros
a = 0       # Limite inferior
b = 2       # Limite superior
m = 20      # Número de subintervalos
y0 = 0.5    # Valor inicial y(a) = 0.5
z0 = 0.0    # Valor inicial z(a) = 0.0

# Resolvendo o sistema de EDOs
VetX, VetY, VetZ = runge_kutta_4th_order_system(f, g, a, b, m, y0, z0)

# Plotando os resultados
plt.plot(VetX, VetY, label='y(x) - Runge-Kutta de 4ª ordem')
plt.plot(VetX, VetZ, label='z(x) - Runge-Kutta de 4ª ordem')
plt.xlabel('x')
plt.ylabel('y, z')
plt.title('Solução do sistema de EDOs usando Runge-Kutta de 4ª ordem')
plt.legend()
plt.grid(True)
plt.show()
