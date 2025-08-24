import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# define the Van der Pol oscillator model:
def van_der_pol(t, z, mu):
    x, y = z
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]

# define the nullclines:
def y_nullcline(x, mu):
    return x/(mu*(1-x**2))
def x_nullcline(y, mu):
    return 0*y

# set time span:
eval_time = 100
t_iteration = 1000
t_span = [0, eval_time]
t_eval = np.linspace(*t_span, t_iteration)

# set initial conditions:
z0 = [2, 0]

# set Van der Pol oscillator parameter:
mu = 1 # stable: 1

# calculate the vector field:
mgrid_size = 8
x, y = np.meshgrid(np.linspace(-mgrid_size, mgrid_size, 15), 
                   np.linspace(-mgrid_size, mgrid_size, 15))
u = y
v = mu * (1 - x**2) * y - x

# calculating the trajectory for the Van der Pol oscillator:
sol_stable = solve_ivp(van_der_pol, t_span, z0, args=(mu,), t_eval=t_eval)

# define the x-array for the nullclines:
x_null = np.arange(-mgrid_size,mgrid_size,0.001)

# plot vector field and trajectory:
plt.figure(figsize=(6, 6))
plt.clf()
speed = np.sqrt(u**2 + v**2)
plt.streamplot(x, y, u, v, color=speed, cmap='cool', density=2.0)
plt.plot(x_null, y_nullcline(x_null, mu)  , '.', c="darkturquoise", markersize=2)
plt.plot(x_null, x_nullcline(x_null, mu)  , '.', c="darkturquoise", markersize=2)
plt.plot(sol_stable.y[0], sol_stable.y[1], 'r-', lw=3,
         label=f'Trajectory for $\mu$={mu}\nand $z_0$={z0}')  # trajectory
# indicate start point:
plt.plot(sol_stable.y[0][0], sol_stable.y[1][0], 'bo', label='start point', alpha=0.75, markersize=7)
plt.plot(sol_stable.y[0][-1], sol_stable.y[1][-1], 'o', c="yellow", label='end point', alpha=0.75, markersize=7)
plt.title('phase plane plot: Van der Pol oscillator')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right') #, bbox_to_anchor=(1, 0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.ylim(-mgrid_size, mgrid_size)
plt.tight_layout()
plt.show()

#Reference
#https://www.fabriziomusacchio.com/blog/2024-03-24-van_der_pol_oscillator/
