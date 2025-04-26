import qutip as qp
import numpy as np

# Define the parameters
omega_c = 1.0  # Cavity frequency
omega_q = 1.5  # Qubit frequency
g = 0.1   # Coupling strength
epsilon_d = 0.2  # Driving amplitude for cavity
epsilon_r = 0.3  # Driving amplitude for qubit
omega_d = 1.1  # Driving frequency for cavity
omega_r = 1.6  # Driving frequency for qubit

# Define the operators
a = qp.destroy(2)  # Cavity annihilation operator (assuming 2 levels for the cavity for simplicity, can be larger)
sigma_minus = qp.destroy(2) # Qubit lowering operator
sigma_plus = qp.create(2)  # Qubit raising operator

# Define the Hamiltonian terms
term1 = omega_c * a.dag() * a
term2 = -1/2 * omega_q * qp.sigmaz()
term3 = g * (a.dag() * sigma_minus + a * sigma_plus)

# Define the time-dependent coefficients as strings for QuTiP's t-dependence
def term4_coeff(t, args):
    coeff_d = args['epsilon_d'] * np.exp(-1j * args['omega_d'] * t)
    coeff_r = args['epsilon_r'] * np.exp(-1j * args['omega_r'] * t)
    return coeff_d + coeff_r

def term4_coeff_conj(t, args):
    return np.conjugate(term4_coeff(t, args))

args = dict(epsilon_d=epsilon_d, omega_d=omega_d, epsilon_r=epsilon_r, omega_r=omega_r)

# Combine the terms into a list for the time-dependent Hamiltonian
H = [term1, term2, term3, [a.dag(), term4_coeff], [a, term4_coeff_conj]]

# Create the time-dependent Hamiltonian onject
H_t = qp.QobjEvo(H, args=args)