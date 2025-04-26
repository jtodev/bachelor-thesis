import qutip as qt
import numpy as np
import matplotlib.pyplot as plt


# CREATE THE HAMILTONIAN FOR THE SYSTEM

# Define the parameters
omega_c = 2 * np.pi * 6.826  # Cavity frequency (nu_c = 6.826 GHz)
omega_q = 2 * np.pi * 5.0258  # Qubit frequency (nu_q = 5.0258 GHz)
kappa = 2 * np.pi * 4.3 * 1e-3  # Linewidth (kappa / 2 pi = 4.3 MHz)
omega_d = 0.99 * omega_c  # Driving frequency for cavity (near cavity frequency)
omega_r = 0.99 * omega_q  # Driving frequency for qubit (near qubit frequency)
epsilon_d = 0.2  # Driving amplitude for cavity
epsilon_r = 0.3  # Driving amplitude for qubit
g = kappa * 1e-3  # Coupling strength (must be much smaller than kappa for approximations)

Delta = omega_q - omega_c  # Detuning
Delta_c = omega_d - omega_c
Omega_R = -2 * epsilon_r * g / Delta
chi = g**2 / Delta

# Define the operators
a = qt.destroy(2)  # Cavity photon annihilation operator
                   # (assuming 2 levels for the cavity for simplicity, can be larger)
sigma_minus = qt.destroy(2) # Qubit lowering operator
sigma_plus = qt.create(2)  # Qubit raising operator
sigma_z = qt.sigmaz()  # Qubit Z operator
abar = epsilon_d / (Delta_c + 0.5j * kappa)
d = a - abar

# Define the different Hamiltonians
H_c = -Delta_c * d.dag() * d  # Cavity
H_q = -0.5 * Omega_R * sigma_z  # Qubit
H_int = -chi * (np.conj(abar) * d * sigma_plus + abar * d.dag() * sigma_minus)  # Interaction

H = H_c + H_q * H_int  # Jaynes-Cummings Hamiltonian


# SOLVE THE LINDBLAD EQUATION

# Define additional operators
sigma_x = qt.sigmax()  # Qubit X operator

# Define parameters for the master eqation solver
psi0 = (qt.basis(2, 0) + qt.basis(2, 1)).unit()  # Initial state (|+>)
times = np.linspace(0.0, 10.0, 1000)  # Time span to evaluate
e_ops = dict(z=sigma_z, x=sigma_x, h=H)  # Evaluation operators

result = qt.mesolve(H, psi0, times, e_ops=e_ops)  # Result of the master equation

# Visualize result
plt.plot(result.times, np.array(result.e_data['z'])**2, label=f'$\\langle\\sigma_z\\rangle$')
plt.plot(result.times, np.array(result.e_data['x'])**2, label=f'$\\langle\\sigma_x\\rangle$')
# plt.plot(result.times, result.e_data['h'], label='$E$')
plt.legend()
# plt.xscale('log')
plt.show()