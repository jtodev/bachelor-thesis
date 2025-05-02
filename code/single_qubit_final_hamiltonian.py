import qutip as qt
import numpy as np
import matplotlib.pyplot as plt


# SETUP

n_vals = np.linspace(0, 10, 100)  # n-values to evaluate
times = np.linspace(0, 10, 10)  # Time span to evaluate
x_data = np.zeros(shape=(n_vals.size, times.size))  # For storing qupit X operator evaluation output
z_data = x_data.copy()  # For storing qupit Z operator evaluation output

# Define the operators
a = qt.destroy(2)  # Cavity photon annihilation operator
                    # (assuming 2 levels for the cavity for simplicity, can be larger)
sigma_minus = qt.destroy(2)  # Qubit lowering operator
sigma_plus = qt.create(2)  # Qubit raising operator
sigma_x = qt.sigmax()  # Qubit X operator
sigma_z = qt.sigmaz()  # Qubit Z operator

# Define constant parameters
chi = 2 * np.pi * -0.66 * 1e-3  # Dispersive coupling strength (chi / 2 pi = -0.66 MHz)
omega_c = 2 * np.pi * 6.826  # Cavity frequency (nu_c = 6.826 GHz)
kappa = 2 * np.pi * 4.3 * 1e-3  # Linewidth (kappa / 2 pi = 4.3 MHz)
omega_d = 0.99 * omega_c  # Driving frequency for cavity (near cavity frequency)
epsilon_r = 0.3  # Driving amplitude for qubit
# g = kappa * 1e-3  # Coupling strength (must be much smaller than kappa for approximations)

# Define parameters for the master equation solver
psi0 = (qt.basis(2, 0) + qt.basis(2, 1)).unit()  # Initial state (|+>)
c_ops = []  # Collapse operators
e_ops = dict(x=sigma_x, z=sigma_z)  # Evaluation operators


for i,n in enumerate(n_vals):
    

    # CREATE THE n-DEPENDENT HAMILTONIAN FOR THE SYSTEM

    # Define n-dependent parameters
    omega_q = 2 * np.pi * 5.0258 + (2 * chi * n)  # Qubit frequency (nu_q = 5.0258 GHz)
    omega_r = 0.99 * omega_q  # Driving frequency for qubit (near qubit frequency)
    Delta = omega_q - omega_c  # Detuning
    # chi = g**2 / Delta
    g = np.sqrt(chi * Delta)
    Omega_R = -2 * epsilon_r * g / Delta
    Delta_c = -Omega_R
    Delta_q = omega_q - omega_r
    epsilon_d = np.sqrt(n) * np.abs(Delta_c + 0.5j * kappa)
    abar = epsilon_d / (Delta_c + 0.5j * kappa)  # Driving amplitude for cavity

    # Define d operator
    d = a - abar

    # Define the different Hamiltonians
    H_c = -Delta_c * d.dag() * d  # Cavity Hamiltonian
    H_q = -0.5 * (Delta_q + chi) * sigma_z  # Qubit Hamiltonian (n-dependent)
    H_int1 = -0.5 * Omega_R * sigma_x  # Interaction Hamiltonian pt. 1
    H_int2 = -chi * ( (np.conj(abar) * d) + (abar * d.dag()) + (d.dag() * d) ) * sigma_z  # Interaction Hamiltonian pt. 2

    H = H_c + H_q + H_int1 + H_int2  # Final n-dependent Hamiltonian


    # SOLVE THE LINDBLAD MASTER EQUATION

    result = qt.mesolve(H, psi0, times, c_ops, e_ops)  # Result of the master equation

    x_data[i, :] = result.e_data['x']  # Store <X> for all times
    z_data[i, :] = result.e_data['z']  # Store <Z> for all times


# Visualize result
plt.plot(n_vals, x_data[:, -1]**2, label='$\\langle\\sigma_x\\rangle$')
plt.plot(n_vals, z_data[:, -1]**2, label='$\\langle\\sigma_z\\rangle$')

plt.ylabel('$\\langle\\sigma_x\\rangle, \\langle\\sigma_z\\rangle$')
plt.xlabel('$\\overline{n}$')
plt.xscale('log')
plt.legend()
plt.show()