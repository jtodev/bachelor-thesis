import qutip as qt
import numpy as np
import matplotlib.pyplot as plt


# CREATE THE TIME-DEPENDENT HAMILTONIAN FOR THE SYSTEM

# Define the parameters
omega_c = 1.0  # Cavity frequency
omega_q = 1.5  # Qubit frequency
omega_d = 0.95 * omega_c  # Driving frequency for cavity (near cavity frequency)
omega_r = 0.95 * omega_q  # Driving frequency for qubit (near qubit frequency)
epsilon_d = 0.2  # Driving amplitude for cavity
epsilon_r = 0.3  # Driving amplitude for qubit
g = 0.05   # Qubit-cavity coupling constant (must be small for approximations)

# Define the operators
a = qt.destroy(2)  # Cavity annihilation operator
                   # (assuming 2 levels for the cavity for simplicity, can be larger)
sigma_minus = qt.destroy(2) # Qubit lowering operator
sigma_plus = qt.create(2)  # Qubit raising operator
sigma_z = qt.sigmaz()  # Qubit Z operator

# Define the terms of the Jaynes-Cummings Hamiltonian
H_c = omega_c * a.dag() * a  # Cavity Hamiltonian
H_q = -1/2 * omega_q * sigma_z  # Qubit Hamiltonian
H_int = g * (a * sigma_plus + a.dag() * sigma_minus)  # Interaction Hamiltonian
# Note: The first term describes the annihilation of a photon (absorption) and
#       creation of an atomic excitation while the second term describes the
#       creation of a photon (emission) and annihilation of an atomic excitation.

H_jc = H_c + H_q + H_int  # Jaynes-Cummings Hamiltonian

# Define the time-dependent coefficients of the two external drives
def term4_coeff(t, args):
    coeff_d = args['epsilon_d'] * np.exp(-1j * args['omega_d'] * t)
    coeff_r = args['epsilon_r'] * np.exp(-1j * args['omega_r'] * t)
    return coeff_d + coeff_r

def term4_coeff_conj(t, args):
    return np.conj(term4_coeff(t, args))

args = dict(epsilon_d=epsilon_d, omega_d=omega_d, epsilon_r=epsilon_r, omega_r=omega_r)

# Combine the terms into a list for the time-dependent Hamiltonian
H_terms = [H_jc, [a.dag(), term4_coeff], [a, term4_coeff_conj]]

# Create the time-dependent Hamiltonian object
H_t = qt.QobjEvo(H_terms, args=args)


# SOLVE THE LINDBLAD EQUATION

# Define additional operators
sigma_x = qt.sigmax()  # Qubit X operator

# Define parameters for the master eqation solver
psi0 = (qt.basis(2, 0) + qt.basis(2, 1)).unit()  # Initial state (|+>)
times = np.linspace(0.0, 10.0, 1000)  # Time span to evaluate
e_ops = dict(z=sigma_z, x=sigma_x)  # Evaluation operators

result = qt.mesolve(H_t, psi0, times, e_ops=e_ops)  # Result of the master equation

# Visualize result
for k,v in result.e_data.items():
    plt.plot(result.times, np.array(v)**2, label=f'$\\langle\\sigma_{k}\\rangle$')
plt.legend()
plt.xscale('log')
plt.show()