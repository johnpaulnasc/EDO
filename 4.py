def runge_kutta_4_system(f, g, a, b, y0, z0, m):
    """
    Implementação do método de Runge-Kutta de 4ª ordem para um sistema de duas EDOs.
    
    :param f: Função que representa a primeira EDO dy/dx = f(x, y, z)
    :param g: Função que representa a segunda EDO dz/dx = g(x, y, z)
    :param a: Limite inferior do intervalo
    :param b: Limite superior do intervalo
    :param y0: Valor inicial y(a)
    :param z0: Valor inicial z(a)
    :param m: Número de subintervalos
    :return: Vetores com as abscissas (x) e as soluções (y, z)
    """
    h = (b - a) / m
    x = np.linspace(a, b, m+1)
    y = np.zeros(m+1)
    z = np.zeros(m+1)
    y[0] = y0
    z[0] = z0
    
    for i in range(m):
        k1_y = h * f(x[i], y[i], z[i])
        k1_z = h * g(x[i], y[i], z[i])
        k2_y = h * f(x[i] + h / 2, y[i] + k1_y / 2, z[i] + k1_z / 2)
        k2_z = h * g(x[i] + h / 2, y[i] + k1_y / 2, z[i] + k1_z / 2)
        k3_y = h * f(x[i] + h / 2, y[i] + k2_y / 2, z[i] + k2_z / 2)
        k3_z = h * g(x[i] + h / 2, y[i] + k2_y / 2, z[i] + k2_z / 2)
        k4_y = h * f(x[i] + h, y[i] + k3_y, z[i] + k3_z)
        k4_z = h * g(x[i] + h, y[i] + k3_y, z[i] + k3_z)
        y[i+1] = y[i] + (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
        z[i+1] = z[i] + (k1_z + 2*k2_z + 2*k3_z + k4_z) / 6
    
    return x, y, z

# Exemplo de uso
def f(x, y, z):
    return -2 * y + z

def g(x, y, z):
    return -y + 3 * z

a = 0
b = 1
y0 = 1
z0 = 1
m = 10

x, y, z = runge_kutta_4_system(f, g, a, b, y0, z0, m)

# Imprimindo a tabela de resultados
data = {'x': x, 'y': y, 'z': z}
df = pd.DataFrame(data)
print(df)

# Gráfico
plt.plot(x, y, label='y (solução 1)')
plt.plot(x, z, label='z (solução 2)')
plt.xlabel('x')
plt.ylabel('Soluções')
plt.title('Método de Runge-Kutta de 4ª Ordem para um sistema de duas EDOs')
plt.legend()
plt.grid(True)
plt.show()
