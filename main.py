import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def lorenz_equations(t, xyz, sigma, rho, beta):
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Parameters
sigma = 16
rho = 45
beta = 4

# Initial conditions
x0, y0, z0 = 12, 4, 1
initial_conditions = [x0, y0, z0]

# Time range
t_span1 = [0, 20]
t_span2 = [0, 30]
t_span3 = [0, 40]
t_span4 = [0, 60]

# Step size
h = 0.001

# Solve using solve_ivp
sol1 = solve_ivp(lorenz_equations, t_span1, initial_conditions, args=(sigma, rho, beta), dense_output=True)
sol2 = solve_ivp(lorenz_equations, t_span2, initial_conditions, args=(sigma, rho, beta), dense_output=True)
sol3 = solve_ivp(lorenz_equations, t_span3, initial_conditions, args=(sigma, rho, beta), dense_output=True)
sol4 = solve_ivp(lorenz_equations, t_span4, initial_conditions, args=(sigma, rho, beta), dense_output=True)

# Time points for evaluation
t_values1 = np.linspace(t_span1[0], t_span1[1], int((t_span1[1]-t_span1[0])/h)+1)
t_values2 = np.linspace(t_span2[0], t_span2[1], int((t_span2[1]-t_span2[0])/h)+1)
t_values3 = np.linspace(t_span3[0], t_span3[1], int((t_span3[1]-t_span3[0])/h)+1)
t_values4 = np.linspace(t_span4[0], t_span4[1], int((t_span4[1]-t_span4[0])/h)+1)

# Solve for each time range
xyz_values1 = sol1.sol(t_values1)
xyz_values2 = sol2.sol(t_values2)
xyz_values3 = sol3.sol(t_values3)
xyz_values4 = sol4.sol(t_values4)

# Plotting
fig, axs = plt.subplots(4, 1, figsize=(10, 20))

axs[0].plot(t_values1, xyz_values1[0], label='x(t)')
axs[0].plot(t_values1, xyz_values1[1], label='y(t)')
axs[0].plot(t_values1, xyz_values1[2], label='z(t)')
axs[0].set_title('t belongs to [0, 20]')
axs[0].legend()

axs[1].plot(t_values2, xyz_values2[0], label='x(t)')
axs[1].plot(t_values2, xyz_values2[1], label='y(t)')
axs[1].plot(t_values2, xyz_values2[2], label='z(t)')
axs[1].set_title('t belongs to [0, 30]')
axs[1].legend()

axs[2].plot(t_values3, xyz_values3[0], label='x(t)')
axs[2].plot(t_values3, xyz_values3[1], label='y(t)')
axs[2].plot(t_values3, xyz_values3[2], label='z(t)')
axs[2].set_title('t belongs to [0, 40]')
axs[2].legend()

axs[3].plot(t_values4, xyz_values4[0], label='x(t)')
axs[3].plot(t_values4, xyz_values4[1], label='y(t)')
axs[3].plot(t_values4, xyz_values4[2], label='z(t)')
axs[3].set_title('t belongs to [0, 60]')
axs[3].legend()

plt.tight_layout()
plt.show()
