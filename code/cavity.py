import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from cache_utils import get_hash, save_result, load_result

# Define the cavity dimension (number of Fock states)
N = 15

# Create the annihilation operator for the cavity
a = qt.destroy(N)

# Create number operator
n = a.dag() * a

# Define parameters
omega_c = 2 * np.pi * 6.826  # Cavity frequency (nu_c = 6.826 GHz)
epsilon_d = 0.9 * omega_c  # Driving frequency for cavity (near cavity frequency)
kappa = 2 * np.pi * 4.3 * 1e-3  # Linewidth (kappa / 2 pi = 4.3 MHz)

# Define epsilon_d range (as fraction of omega_c)
epsilon_d_range = np.linspace(0.5, 1.5, 10) * omega_c
expect = np.zeros((len(epsilon_d_range), N))

# Plot the results
fig = plt.figure()
ax1 = fig.add_subplot(111)

for j, epsilon_d in enumerate(epsilon_d_range):
    # Define the Hamiltonian
    H = omega_c * a.dag() * a + epsilon_d * (a + a.dag())  # Free Hamiltonian + drive

    # Define collapse operators for Lindblad master equation
    c_ops = [np.sqrt(kappa) * a]  # Cavity decay

    # Define initial state (ground state)
    psi0 = qt.basis(N, 0)

    # Define time points for evolution
    tlist = np.linspace(0, 10, 100)  # 10 seconds evolution

    # Define solver options
    options = {
        'nsteps': 10000,
        'store_states': True,
        'rtol': 1e-8,
        'atol': 1e-8
    }

    # Create parameter dictionary for hashing
    params = {
        'N': N,
        'omega_c': omega_c,
        'epsilon_d': epsilon_d,
        'kappa': kappa,
        'tlist': tlist.tolist(),
        'options': options
    }

    # Get hash of parameters
    hash_value = get_hash(params)

    # Try to load cached result
    result = load_result(hash_value, prefix="cavity")

    # If no cached result exists, calculate and save
    if result is None:
        print("Calculating new result...")
        result = qt.mesolve(H, psi0, tlist, c_ops, options=options)
        save_result(result, hash_value, prefix="cavity")
        print(f"Result saved with hash: {hash_value}")
    else:
        print(f"Loaded cached result with hash: {hash_value}")

    # Get the final density matrix
    rho_final = result.states[-1]

    # Calculate expectation value of |1⟩ over time
    for i in range(N):
        # Create projection operator for |1⟩ state
        P = qt.basis(N, i) * qt.basis(N, i).dag()
        # Calculate expectation value
        expect[j, i] = qt.expect(P, rho_final)

    ax1.plot(expect[j], label=f'$\\epsilon_d/\\omega_c = {epsilon_d/omega_c:.2f}$')

ax1.set_xlabel('$n$')
ax1.set_ylabel('$\\langle n|\\rho_\\mathrm{final}|n\\rangle$')
ax1.grid(True)
ax1.legend()

plt.show()