# lotka_volterra_sim.py
# Simulate and visualize the Lotka–Volterra predator–prey system.
# dx/dt = α x - β x y
# dy/dt = δ x y - γ y

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -------------------------------
# Model and utilities
# -------------------------------
def lotka_volterra(t, z, alpha, beta, delta, gamma):
    x, y = z
    dx = alpha * x - beta * x * y
    dy = delta * x * y - gamma * y
    return [dx, dy]

def equilibrium_points(alpha, beta, delta, gamma):
    # (0, 0) and (gamma/delta, alpha/beta)
    return (0.0, 0.0), (gamma / delta, alpha / beta)

def jacobian_at(x, y, alpha, beta, delta, gamma):
    # J = [[α - β y,   -β x],
    #      [δ y,       δ x - γ]]
    return np.array([[alpha - beta * y, -beta * x],
                     [delta * y,        delta * x - gamma]])

# -------------------------------
# Parameters (feel free to tweak)
# -------------------------------
alpha = 1.0   # prey growth rate
beta  = 0.1   # predation rate
delta = 0.075 # predator growth per prey eaten
gamma = 1.5   # predator death rate

t_span = (0.0, 60.0)
t_eval = np.linspace(t_span[0], t_span[1], 2000)

# Initial populations (x0=prey, y0=predator)
x0, y0 = 10.0, 5.0

# -------------------------------
# Solve ODE
# -------------------------------
sol = solve_ivp(
    fun=lambda t, z: lotka_volterra(t, z, alpha, beta, delta, gamma),
    t_span=t_span,
    y0=[x0, y0],
    method="RK45",            # try "Radau" or "BDF" if you run into stiffness
    t_eval=t_eval,
    rtol=1e-8,
    atol=1e-10
)

if not sol.success:
    raise RuntimeError(f"Integration failed: {sol.message}")

x, y = sol.y

# -------------------------------
# Print equilibria and local linearization info
# -------------------------------
e0, e1 = equilibrium_points(alpha, beta, delta, gamma)
print("Equilibria:")
print(f"  E0 = (0, 0)")
print(f"  E1 = ({e1[0]:.4f}, {e1[1]:.4f})  # (gamma/delta, alpha/beta)")

J_e0 = jacobian_at(e0[0], e0[1], alpha, beta, delta, gamma)
J_e1 = jacobian_at(e1[0], e1[1], alpha, beta, delta, gamma)
eig_e0 = np.linalg.eigvals(J_e0)
eig_e1 = np.linalg.eigvals(J_e1)
print("\nJacobian eigenvalues:")
print(f"  at E0: {eig_e0}")
print(f"  at E1: {eig_e1}")

# -------------------------------
# Plot: time series
# -------------------------------
plt.figure(figsize=(8, 4.8))
plt.plot(sol.t, x, label="Prey (x)")
plt.plot(sol.t, y, label="Predator (y)")
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Lotka–Volterra Time Series")
plt.legend()
plt.tight_layout()

# -------------------------------
# Plot: phase portrait with nullclines and vector field
# -------------------------------
plt.figure(figsize=(5.6, 5.6))

# Trajectory
plt.plot(x, y, lw=2, label="Trajectory")
plt.scatter([x0], [y0], s=40, marker="o", label="Start")

# Equilibria
plt.scatter([e0[0], e1[0]], [e0[1], e1[1]], s=60, marker="x", label="Equilibria")

# Nullclines: y = α/β  (dx/dt = 0 when x ≠ 0),  x = γ/δ (dy/dt = 0 when y ≠ 0)
y_nc = alpha / beta
x_nc = gamma / delta

# Axes limits based on data and nullclines
xmax = max(np.max(x), x_nc) * 1.2
ymax = max(np.max(y), y_nc) * 1.2
plt.xlim(0, xmax)
plt.ylim(0, ymax)

plt.axhline(y_nc, linestyle="--", linewidth=1, label="dx/dt = 0 (y = α/β)")
plt.axvline(x_nc, linestyle="--", linewidth=1, label="dy/dt = 0 (x = γ/δ)")

# Vector field
nx, ny = 20, 20
X, Y = np.meshgrid(np.linspace(0, xmax, nx), np.linspace(0, ymax, ny))
U = alpha * X - beta * X * Y
V = delta * X * Y - gamma * Y
# Normalize arrows for readability
N = np.hypot(U, V)
N[N == 0] = 1.0
plt.quiver(X, Y, U / N, V / N, angles="xy", width=0.003)

plt.xlabel("Prey (x)")
plt.ylabel("Predator (y)")
plt.title("Lotka–Volterra Phase Portrait")
plt.legend(loc="upper right")
plt.tight_layout()

plt.show()

# -------------------------------
# Optional: estimate oscillation periods (prey) via peaks
# -------------------------------
try:
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(x, distance=10)
    if len(peaks) > 1:
        periods = np.diff(sol.t[peaks])
        print(f"\nEstimated prey oscillation periods (n={len(periods)}):")
        print(periods)
        print(f"Mean period ≈ {np.mean(periods):.3f}")
except Exception as err:
    print(f"(Skipping period estimation: {err})")

#REFERENCE: GPT-5
