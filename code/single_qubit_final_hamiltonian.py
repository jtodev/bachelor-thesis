import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from cache_utils import get_hash, save_result, load_result

# SETUP

# Define dimensions
N_cavity_range = np.arange(2,21)  # Range of Fock states for cavity
N_qubit = 2                       # Number of states for qubit

# Define the operators for qubit
sigma_minus = qt.destroy(N_qubit)  # Qubit lowering operator
sigma_plus = qt.create(N_qubit)    # Qubit raising operator
sigma_x = qt.sigmax()              # Qubit X operator
sigma_z = qt.sigmaz()              # Qubit Z operator

# Define constant parameters
chi = 2 * np.pi * -0.66 * 1e-3  # Dispersive coupling strength (chi / 2 pi = -0.66 MHz)
omega_c = 2 * np.pi * 6.826     # Cavity frequency (nu_c = 6.826 GHz)
omega_q = 2 * np.pi * 5.0258    # Qubit frequency (nu_q = 5.0258 GHz)
kappa = 2 * np.pi * 4.3 * 1e-3  # Linewidth (kappa / 2 pi = 4.3 MHz)
epsilon_d = 0.9 * omega_c       # Driving frequency for cavity
epsilon_r = 0.3                 # Driving amplitude for qubit

# Define time points for evolution
times = np.linspace(0, 10, 100)  # Time span to evaluate

# Define solver options
options = {
    'nsteps': 10000,
    'store_states': True,
    'rtol': 1e-8,
    'atol': 1e-8
}

# Create parameter dictionary for hashing
params = {
    'N_qubit': N_qubit,
    'omega_q': omega_q,
    'omega_c': omega_c,
    'chi': chi,
    'kappa': kappa,
    'epsilon_d': epsilon_d,
    'times': times.tolist(),
    'options': options,
}

# Store expectation values for different photon numbers
expect_x = np.zeros_like(N_cavity_range, dtype=float)
expect_z = np.zeros_like(N_cavity_range, dtype=float)

for i, N_cavity in enumerate(N_cavity_range):
    print(f"Calculating for N_cavity = {N_cavity}")

    # Add N_cavity to params
    params['N_cavity'] = N_cavity
    
    # Define the operators for cavity
    a = qt.destroy(N_cavity)  # Cavity photon annihilation operator
    n = a.dag() * a           # Cavity number operator

    # Define initial state (|0⟩_c ⊗ |+⟩_q)
    psi0_cavity = qt.basis(N_cavity, 0)                                # Cavity ground state
    psi0_qubit = (qt.basis(N_qubit, 0) + qt.basis(N_qubit, 1)).unit()  # Qubit |+⟩ state
    psi0 = qt.tensor(psi0_cavity, psi0_qubit)                          # Composite initial state

    # Define collapse operators
    c_ops = [np.sqrt(kappa) * qt.tensor(a, qt.qeye(N_qubit))]  # Cavity decay

    # Define evaluation operators
    e_ops = {
        'x': qt.tensor(qt.qeye(N_cavity), sigma_x),
        'z': qt.tensor(qt.qeye(N_cavity), sigma_z),
        'n': qt.tensor(n, qt.qeye(N_qubit))
    }

    # Define the Hamiltonian
    # Cavity Hamiltonian
    H_c = omega_c * qt.tensor(n, qt.qeye(N_qubit))
    # Qubit Hamiltonian
    H_q = 0.5 * omega_q * qt.tensor(qt.qeye(N_cavity), sigma_z)
    # Interaction Hamiltonian (dispersive coupling)
    H_int = chi * qt.tensor(n, sigma_z)
    # Drive Hamiltonian
    H_drive = epsilon_d * qt.tensor(a + a.dag(), qt.qeye(N_qubit))

    # Total Hamiltonian
    H = H_c + H_q + H_int + H_drive

    # Get hash of parameters
    hash_value = get_hash(params)

    # Try to load cached result
    result = load_result(hash_value, prefix="cavity_single_qubit")

    # If no cached result exists, calculate and save
    if result is None:
        print("Calculating new result...")
        result = qt.mesolve(H, psi0, times, c_ops, e_ops, options=options)
        save_result(result, hash_value, prefix="cavity_single_qubit")
        print(f"Result saved with hash: {hash_value}")
    else:
        print(f"Loaded cached result with hash: {hash_value}")

    # Get the final density matrix
    rho_final = result.states[-1]

    # Create projection operator for |n⟩ state
    P_n = qt.tensor(qt.basis(N_cavity, i) * qt.basis(N_cavity, i).dag(), qt.qeye(N_qubit))
    
    # Calculate expectation values conditioned on photon number n
    rho_n = P_n * rho_final * P_n
    trace = rho_n.tr()
    
    # Calculate expectation values if there's significant population
    if abs(trace) > 1e-10:  # Only calculate if there's significant population
        rho_n = rho_n / trace  # Normalize
        # Take real part of expectation values
        expect_x[i] = np.real(qt.expect(qt.tensor(qt.qeye(N_cavity), sigma_x), rho_n))
        expect_z[i] = np.real(qt.expect(qt.tensor(qt.qeye(N_cavity), sigma_z), rho_n))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(N_cavity_range, expect_x, 'o-', label='$\\langle\\sigma_x\\rangle$')
plt.plot(N_cavity_range, expect_z, 's-', label='$\\langle\\sigma_z\\rangle$')
plt.xlabel('Cavity photon number $n$')
plt.ylabel('Qubit expectation values')
plt.title('Qubit expectation values conditioned on cavity photon number')
plt.grid(True)
plt.legend()
plt.xscale('log')
plt.show()