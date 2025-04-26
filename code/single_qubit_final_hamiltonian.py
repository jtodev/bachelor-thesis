import qutip as qt
import numpy as np
import matplotlib.pyplot as plt


# CREATE THE HAMILTONIAN FOR THE SYSTEM

# Define the parameters
omega_c = 1.0  # Cavity frequency
omega_q = 1.5  # Qubit frequency
omega_d = 0.95 * omega_c  # Driving frequency for cavity (near cavity frequency)
epsilon_d = 0.2  # Driving amplitude for cavity
epsilon_r = 0.3  # Driving amplitude for qubit
g = 0.05   # Coupling strength (must be small for approximations)

# Define the operators
a = qt.destroy(2)  # Cavity annihilation operator
                   # (assuming 2 levels for the cavity for simplicity, can be larger)
sigma_minus = qt.destroy(2) # Qubit lowering operator
sigma_plus = qt.create(2)  # Qubit raising operator
sigma_z = qt.sigmaz()  # Qubit Z operator

kappa = 0.01  # Proportional to the Hamiltonian
Delta = omega_q - omega_c
Delta_c = omega_d - omega_c
Omega_R = -2 * epsilon_r * g / Delta
chi = g**2 / Delta

abar = epsilon_d / (Delta_c + 0.5j * kappa)
d = a - abar

# Define the Hamiltonian terms
term1 = -Delta_c * d.dag() * d
term2 = -0.5 * Omega_R * sigma_z
term3 = -chi * (np.conjugate(abar) * d * sigma_plus + abar * d.dag() * sigma_minus)

H = term1 + term2 + term3


# SOLVE THE LINDBLAD EQUATION

# Define additional operators
sigma_x = qt.sigmax()  # Qubit X operator

# Define parameters for the master eqation solver
psi0 = qt.basis(2, 0)  # Initial state
times = np.linspace(0.0, 10.0, 1000)  # Time span to evaluate
e_ops = dict(z=sigma_z, x=sigma_x)  # Evaluation operators

result = qt.mesolve(H, psi0, times, e_ops=e_ops)  # Result of the master equation

# Visualize result
for k,v in result.e_data.items():
    plt.plot(result.times, np.array(v)**2, label=f'$\\langle\\sigma_{k}\\rangle$')
plt.legend()
plt.xscale('log')
plt.show()