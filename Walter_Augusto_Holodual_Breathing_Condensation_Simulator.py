# 🔥 Walter Augusto’s Holodual Breathing Condensation Simulator
# Conceptualized and introduced by Walter Augusto on April 10, 2025.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Parameters ---
epsilon = 2.5  # Nonlinear self-densification strength (adjust as needed)

# --- Define the Breathing Condensation Equation ---
def breathing_condensation(t, y):
    phi, phi_dot = y
    phi_ddot = -np.sin(phi) * (1 + epsilon * phi**2)
    return [phi_dot, phi_ddot]

# --- Initial Conditions ---
phi_0 = 0.1     # Initial breathing field value
phi_dot_0 = 0.0 # Initial breathing rate
initial_conditions = [phi_0, phi_dot_0]

# --- Time Span ---
t_start = 0.0
t_end = 100.0
t_points = 5000
t_eval = np.linspace(t_start, t_end, t_points)

# --- Solve the System ---
sol = solve_ivp(
    breathing_condensation,
    (t_start, t_end),
    initial_conditions,
    t_eval=t_eval,
    method='RK45',
    rtol=1e-8,
    atol=1e-10
)

# --- Extract Results ---
t = sol.t
phi = sol.y[0]
phi_dot = sol.y[1]

# --- Plot the Results ---
plt.figure(figsize=(12, 6))
plt.plot(t, phi, label='Breathing Field φ(t)', color='blue')
plt.title('Walter Augusto’s Holodual Breathing Condensation Simulation')
plt.xlabel('Time (t)')
plt.ylabel('Breathing Field φ')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
