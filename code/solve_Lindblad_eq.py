from qutip import *
import numpy as np

# Define the parameters
wc = 1.0  # Cavity frequency
wq = 1.5  # Qubit frequency
g = 0.1   # Coupling strength
ed = 0.2  # Driving amplitude for cavity
ef = 0.3  # Driving amplitude for qubit
wd = 1.1  # Driving frequency for cavity
wf = 1.6  # Driving frequency for qubit

# Define the operators
a = destroy(2)  # Cavity annihilation operator (assuming 2 levels for the cavity for simplicity, can be larger)
sm = destroy(2) # Qubit lowering operator
sz = sigmaz()   # Qubit Z operator
sp = create(2)  # Qubit raising operator

# Define the Hamiltonian terms
term1 = wc * a.dag() * a
term2 = -0.5 * wq * sz
term3 = g * (a.dag() * sm + a * sp)
term4_coeff_d = ed
term4_op_d = a.dag()
term4_coeff_d_conj = np.conjugate(ed)
term4_op_d_dag = a
term5_coeff_f = ef
term5_op_f = sp
term5_coeff_f_conj = np.conjugate(ef)
term5_op_f_dag = sm

# Define the time-dependent coefficients as strings for QuTiP's t-dependence
def coeff_d(t, args):
    return ed * np.exp(-1j * wd * t)

def coeff_d_conj(t, args):
    return np.conjugate(ed) * np.exp(1j * wd * t)

def coeff_f(t, args):
    return ef * np.exp(-1j * wf * t)

def coeff_f_conj(t, args):
    return np.conjugate(ef) * np.exp(1j * wf * t)

# Combine the terms into a list for the time-dependent Hamiltonian
H = [
    term1,
    term2,
    term3,
    [term4_op_d, coeff_d],
    [term4_op_d, coeff_d].dag(),
    [term5_op_f, coeff_f],
    [term5_op_f, coeff_f].dag()
]

print(H)