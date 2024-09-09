import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return x - 2*y + 1

def dopri(a, b, h, y0):
    VetX = np.arange(a, b + h, h)
    VetY = np.zeros(len(VetX))
    VetY[0] = y0
    for i in range(1, 4):
        k1 = h * f(VetX[i-1], VetY[i-1])
        k2 = h * f(VetX[i-1] + h/2, VetY[i-1] + k1/2)
        k3 = h * f(VetX[i-1] + h/2, VetY[i-1] + k2/2)
        k4 = h * f(VetX[i-1] + h, VetY[i-1] + k3)
        VetY[i] = VetY[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
    return VetX, VetY

def abm4(a, b, m, y0):
    h = (b - a) / m
    VetX, VetY = dopri(a, a + 3 * h, h, y0)
    erro = []
    
    for i in range(3, m):
        x = VetX[i]
        y = VetY[i]
        
        f0 = f(VetX[i-3], VetY[i-3])
        f1 = f(VetX[i-2], VetY[i-2])
        f2 = f(VetX[i-1], VetY[i-1])
        f3 = f(x, y)
        
        Ypre = y + h * (55 * f3 - 59 * f2 + 37 * f1 - 9 * f0) / 24
        VetX = np.append(VetX, x + h)
        VetY = np.append(VetY, Ypre)
        
        f4 = f(VetX[-1], Ypre)
        Ycor = y + h * (9 * f4 + 19 * f3 - 5 * f2 + f1) / 24
        
        erro.append(abs(Ycor - Ypre) * 19 / 270)
        
        VetY[-1] = Ycor
        
    return VetX, VetY, erro

a, b = 0, 1  
m = 10  
y0 = 1  

VetX, VetY, erro = abm4(a, b, m, y0)


plt.plot(VetX, VetY, label="ABM4")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Solução usando Adams-Bashforth-Moulton")
plt.legend()
plt.grid(True)
plt.show()
