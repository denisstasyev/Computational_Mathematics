%matplotlib inline
import numpy as np
from matplotlib import pyplot as plt

# drawing improvements
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15 #width
fig_size[1] = 5 #height
plt.rcParams["figure.figsize"] = fig_size

# constants
x_min = 0
x_max = 10

h = 10**-1
N = int((x_max - x_min) / h)

u = np.zeros(N)
v = np.zeros(N)

u[0] = 1
v[0] = 1

for i in range(0, N-1):
    A = np.array([[1 - 499*h, -999*h], [499.5*h, 1 + 999.5*h]])
    B = np.array([(1 + 499*h)*u[i] + 999*h*v[i], -499.5*h*u[i] + (1 - 999.5*h)*v[i]])
    X = np.linalg.solve(A, B) # решаем данную систему
    
    u[i+1] = X[0]
    v[i+1] = X[1]

x = np.linspace(x_min, x_max, num=N)

ax = plt.subplot(1,2,1)
plt.plot(x, u, color='blue', label='метод трапеций', linewidth=2.0)
ax.set_title(f"Решение методом трапеций для u(t) при h = {h}")
ax.set_xlabel('t')
ax.set_ylabel('u(t)')
ax.legend()

ax = plt.subplot(1,2,2)
plt.plot(x, v, color='red', label='метод трапеций', linewidth=2.0)
ax.set_title(f"Решение методом трапеций для v(t) при h = {h}")
ax.set_xlabel('t')
ax.set_ylabel('v(t)')
ax.legend()

# Точное решение
x0 = np.arange(x_min, x_max, 0.01)
u0 = 1996/1999 * np.exp(-x0) + 3/1999 * np.exp(-2000*x0)
v0 = -998/1999 * np.exp(-x0) 

ax = plt.subplot(1,2,1)
plt.plot(x0, u0, color='black', label='точное решение', linewidth=2.0)
ax.legend()
plt.grid()

ax = plt.subplot(1,2,2)
plt.plot(x0, v0, color='black', label='точное решение', linewidth=2.0)
ax.legend()
plt.grid()

plt.savefig('trapezoid_method_analysis_1.png')

# Возможно более наглядное изображение решения
import mpl_toolkits.mplot3d.axes3d as p3
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.plot_trisurf(x, u, v, color='blue', label='метод трапеций', linewidth=0.0, antialiased=False)
ax.plot_trisurf(x0, u0, v0, color='black', label='точное решение', linewidth=0.0, antialiased=False)
ax.set_xlabel('t')
ax.set_ylabel('u(t)')
ax.set_zlabel('v(t)')