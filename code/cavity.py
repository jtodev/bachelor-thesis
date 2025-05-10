import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from cache_utils import get_hash, save_result, load_result

# Define the cavity dimension (number of Fock states)
N = 20

# Create the annihilation operator for the cavity
a = qt.destroy(N)

# Define parameters
omega = 1  # Cavity frequency
epsilon = 0.9 * omega  # Driving frequency for cavity (near cavity frequency)
kappa = 0.2

# Define collapse operators for Lindblad master equation
c_ops = [np.sqrt(kappa) * a]  # Cavity decay

# Define initial state (ground state)
psi0 = qt.basis(N, 0)

# Define time points for evolution
tlist = np.linspace(0, 10, 100)  # 10 seconds evolution

# Solve the master equation
options = qt.Options(nsteps=10000, atol=1e-8, rtol=1e-6)

# Plot the results
fig = plt.figure()
ax1 = fig.add_subplot(111)

# Simulate for different displacements
displacements = -np.linspace(0.9, 2.8, 10) * omega / (omega + 0.5j * kappa)
expect = np.zeros((len(displacements), N))
for j, displacement in enumerate(displacements):

    # Displace destroy operator
    d = a - displacement

    # Define the Hamiltonian
    H = omega * d.dag() * d  - omega * np.conj(displacement) * displacement
    result = qt.mesolve(H, psi0, tlist, c_ops, options=options)

    # Get the final density matrix
    rho_final = result.states[-1]

    # Calculate expectation value of |1⟩ over time
    for i in range(N):

        # Create projection operator for |1⟩ state
        P = qt.basis(N, i) * qt.basis(N, i).dag()
        
        # Calculate expectation value
        expect[j, i] = qt.expect(P, rho_final)

    ax1.plot(expect[j], label='$|\\overline{a}|^2$ = ' + f'{np.conj(displacement) * displacement:.2f}')

ax1.set_xlabel('$n$')
ax1.set_ylabel('$\\langle n|\\rho_\\mathrm{final}|n\\rangle$')
ax1.grid(True)
ax1.legend()

plt.show()