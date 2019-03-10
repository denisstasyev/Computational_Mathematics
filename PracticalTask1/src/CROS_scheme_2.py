%matplotlib inline
import numpy as np
from matplotlib import pyplot as plt

import mpl_toolkits.mplot3d.axes3d as p3

# Различные значения h
hs = [10**-1, 7.5*10**-2, 5*10**-2, 2.5*10**-2,
      10**-2, 7.5*10**-3, 5*10**-3, 2.5*10**-3,
      10**-3, 7.5*10**-4, 5*10**-4, 2.5*10**-4, 10**-4]

lenhs = len(hs)

# drawing improvements
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15 #width
fig_size[1] = 5*len(hs) #height
plt.rcParams["figure.figsize"] = fig_size

# constants
x_min = 0
x_max = 10

cycles = 1
for h in hs:
    
    N = int((x_max - x_min) / h)

    u = np.zeros(N)
    v = np.zeros(N)

    u[0] = 1
    v[0] = 1

    for i in range(0, N-1):
        A = np.array([[complex(1 - 499*h, - h*499), complex(-999*h, - h*999)],
                      [complex(499.5*h, + 499.5*h), complex(1 + 999.5*h, + 999.5*h)]])
        B = np.array([998*u[i] + 1998*v[i], -999*u[i] - 1999*v[i]])
        W = np.linalg.solve(A, B) # решаем данную систему
    
        u[i+1] = u[i] + h*W[0].real
        v[i+1] = v[i] + h*W[1].real

    x = np.linspace(x_min, x_max, num=N)

    ax = plt.subplot(lenhs,2,cycles)
    plt.plot(x, u, color='blue', label='схема CROS', linewidth=2.0)
    ax.set_title(f"Решение схемой CROS для u(t) при h = {h}")
    ax.set_xlabel('t')
    ax.set_ylabel('u(t)')
    ax.legend()

    ax = plt.subplot(lenhs,2,cycles+1)
    plt.plot(x, v, color='red', label='схема CROS', linewidth=2.0)
    ax.set_title(f"Решение схемой CROS для v(t) при h = {h}")
    ax.set_xlabel('t')
    ax.set_ylabel('v(t)')
    ax.legend()

    # Точное решение
    x0 = np.arange(x_min, x_max, 0.01)
    u0 = 1996/1999 * np.exp(-x0) + 3/1999 * np.exp(-2000*x0)
    v0 = -998/1999 * np.exp(-x0) # + 2997/3 * np.exp(-2000*x0)

    ax = plt.subplot(lenhs,2,cycles)
    plt.plot(x0, u0, color='black', label='точное решение', linewidth=2.0)
    ax.legend()
    plt.grid()

    ax = plt.subplot(lenhs,2,cycles+1)
    plt.plot(x0, v0, color='black', label='точное решение', linewidth=2.0)
    ax.legend()
    plt.grid()
    
    cycles += 2
    
plt.savefig('CROS_scheme_2.png')