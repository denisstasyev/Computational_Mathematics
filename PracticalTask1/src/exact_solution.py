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

x = np.arange(x_min, x_max, 0.01)
u = 1996/1999 * np.exp(-x) + 3/1999 * np.exp(-2000*x)
v = -998/1999 * np.exp(-x) + 2997/3 * np.exp(-2000*x)

ax = plt.subplot(1,2,1)
plt.plot(x, u, color='blue', linewidth=2.0)
ax.set_title("Точное решение u(t)")
ax.set_xlabel('t')
ax.set_ylabel('u(t)')
plt.grid()

ax = plt.subplot(1,2,2)
plt.plot(x, v, color='red', linewidth=2.0)
ax.set_title("Точное решение v(t)")
ax.set_xlabel('t')
ax.set_ylabel('v(t)')
plt.grid()

plt.savefig('exact_solution.png')