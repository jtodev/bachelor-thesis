import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# ==================================================================================================
# System Parameters
# ==================================================================================================

# Physical parameters
omega_c = 2 * np.pi * 6.826  # Cavity frequency (nu_c = 6.826 GHz)
omega_q = 2 * np.pi * 5.0258  # Qubit frequency (nu_q = 5.0258 GHz)
omega_r = 0.9 * omega_q  # Qubit drive frequency (near qubit frequency)
omega_d = 0.9 * omega_c  # Drive frequency (near cavity frequency)
chi = 2 * np.pi * -0.66 * 1e-3  # Dispersive coupling strength (chi / 2 pi = -0.66 MHz)
kappa = 2 * np.pi * 4.3 * 1e-3  # Linewidth (kappa / 2 pi = 4.3 MHz)
Omega_R = 1  # Rabi frequency (Nu_R = 9 MHz) (same as -Delta_c)

T_phi = 1  # Phase relaxation time
Gamma_phi = 1 / T_phi  # Phase relaxation rate
T_1 = 0.0001  # Qubit relaxation time
Gamma_plus = 1 / (2 * T_1)  # Qubit relaxation rate
Gamma_minus = 1 / (2 * T_1)  # Qubit relaxation rate

# System dimensions
N_cavity = 10  # Number of Fock states for cavity
N_qubit = 2  # Number of states for qubit

# Time range
time_range = np.linspace(0, 10, 100)

# Calculate detuning
Delta_c = omega_d - omega_c  # Cavity detuning (same as -Omega_R)
Delta_q = omega_q - omega_r  # Qubit detuning

# ==================================================================================================
# Define Operators
# ==================================================================================================

# Identity operators
I_c = qt.qeye(N_cavity)  # Cavity identity operator
I_q = qt.qeye(N_qubit)  # Qubit identity operator

# Cavity operators
a = qt.tensor(qt.destroy(N_cavity), I_q)  # Cavity annihilation operator

# Qubit operators
sigma_x = qt.tensor(I_c, qt.sigmax())  # Qubit X operator
sigma_z = qt.tensor(I_c, qt.sigmaz())  # Qubit Z operator
sigma_plus = qt.tensor(I_c, qt.create(N_qubit))  # Qubit raising operator
sigma_minus = qt.tensor(I_c, qt.destroy(N_qubit))  # Qubit lowering operator

# ==================================================================================================
# Define Hamiltonian Function
# ==================================================================================================

def hamiltonian(displacement):

    # Define displacement operator
    a = qt.destroy(N_cavity)
    d = qt.tensor(a - displacement, I_q)


    # Define the Hamiltonian components
    H_cavity = -Delta_c * d.dag() * d
    H_qubit = -0.5 * (Delta_q + chi * (2 * np.abs(displacement)**2)) * sigma_z
    H_rabi = -0.5 * Omega_R * sigma_x
    H_int = -chi * (np.conj(displacement) * d + displacement * d.dag() + d.dag() * d) * sigma_z
    
    return H_cavity + H_qubit + H_rabi + H_int

# ==================================================================================================
# Simulation Parameters
# ==================================================================================================

# Range of average photon numbers to simulate
blah = Delta_c + 0.5j * kappa
displacement_range = np.sqrt(np.linspace(0, 10, 10)) * np.abs(blah) / blah

# ==================================================================================================
# Simulation and Analysis
# ==================================================================================================

# Arrays to store results
sx_values = np.zeros_like(displacement_range)
sz_values = np.zeros_like(displacement_range)

# Define initial state (|0⟩_c ⊗ |+⟩_q)
psi0_cavity = qt.basis(N_cavity, 0)                                # Cavity ground state
psi0_qubit = (qt.basis(N_qubit, 0) + qt.basis(N_qubit, 1)).unit()  # Qubit |+⟩ state
psi0 = qt.tensor(psi0_cavity, psi0_qubit)

# Collapse operators (dissipation)
c_ops = [
    np.sqrt(kappa) * a,
    np.sqrt(Gamma_phi / 2) * sigma_z,
    np.sqrt(Gamma_plus) * sigma_plus,
    np.sqrt(Gamma_minus) * sigma_minus
]

# Evaluation operators (observables)
e_ops = {
    'x': sigma_x,
    'z': sigma_z,
}

for i, displacement in enumerate(displacement_range):
    H = hamiltonian(displacement)
    print(i)
    result = qt.mesolve(H, psi0, time_range, c_ops, e_ops=e_ops, options={'store_states': True})
    
    # Store final expectation values from time evolution
    sx_values[i] = result.expect[0][-1]  # Take last value from time evolution
    sz_values[i] = result.expect[1][-1]

n_bar_values = abs(displacement_range)**2

# ==================================================================================================
# Visualization
# ==================================================================================================

fig = plt.figure()

# Plot qubit observables
ax1 = fig.add_subplot(111)
ax1.plot(n_bar_values, sx_values, label='$\\langle\\sigma_x\\rangle$ simulation')
ax1.plot(n_bar_values, sz_values, label='$\\langle\\sigma_z\\rangle$ simulation')
ax1.set_xlabel('$\\overline{n}$')
ax1.set_ylabel('$\\langle\\sigma_x\\rangle, \\langle\\sigma_z\\rangle$')
ax1.legend()

# ax2 = fig.add_subplot(122)
# ax2.plot(time_range, n_values, label='$\\langle n\\rangle$ simulation')
# ax2.set_xlabel('Time')
# ax2.set_ylabel('$\\langle n\\rangle$')
# ax2.legend()

plt.show()