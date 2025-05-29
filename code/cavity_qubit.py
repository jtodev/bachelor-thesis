# %% 
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt


# %%
# Define simulation parameters

# Define Hilbert space dimensions
N_cavity = 5  # Cavity dimension (number of Fock states)
N_qubit = 2  # Qubit dimension

# Define identity operatos
I_c = qt.qeye(N_cavity)  # Cavity identity operator
I_q = qt.qeye(N_qubit)  # Qubit identity operator

# Define the annihilation operator for the cavity
d = qt.tensor(qt.destroy(N_cavity), I_q)

# Define qubit operators
X = qt.tensor(I_c, qt.sigmax())  # Qubit X operator
Z = qt.tensor(I_c, qt.sigmaz())  # Qubit Z operator
plus = qt.tensor(I_c, qt.create(N_qubit))  # Qubit raising operator
minus = qt.tensor(I_c, qt.destroy(N_qubit))  # Qubit lowering operator

# Define time points for evolution
t_range = np.linspace(0, 10000, 100)  # Time in ns

# Master equation solver options
options = dict(
    store_states=True,
    progress_bar=True,
    nsteps=int(1e6),
    rtol=1e-4,
    atol=1e-4,
    max_step=1.0,
)

# Define initial state
g = qt.tensor(qt.basis(N_cavity, 0), qt.basis(N_qubit, 0))  # Qubit ground state |0, g⟩
e = qt.tensor(qt.basis(N_cavity, 0), qt.basis(N_qubit, 1))  # Qubit excited state |0, e⟩
psi0 = (g + 1j * e).unit()  # Initial state |0, i⟩


# %%
# Simulate displaced cavity

n_range = np.logspace(-3, 1, num=10)

def solve_master_equation(n):

    # Define parameters
    Omega_R = 2 * np.pi * 9e-3
    Delta_c = - Omega_R
    chi = - 2 * np.pi * 0.66e-3
    kappa = 2 * np.pi * 4.3e-3

    epsilon_r = np.sqrt(n) * abs(Delta_c + 0.5j * kappa)
    alpha = epsilon_r / (Delta_c + 0.5j * kappa)

    # omega_q = 2 * np.pi * 5.0258
    # omega_q_prime = omega_q + chi * (2 * n)
    # omega_r = omega_q_prime
    # Delta_q = omega_q - omega_r
    # Delta_q_prime = Delta_q + chi * (2 * n)
    Delta_q_prime = 0

    # Define cooling rates
    T_1 = 10e3  # Qubit energy decay time (T_1 = 10 μs)
    T_2 = 10.6e3  # Qubit lab frame dephasing time (T_2 = 10.6 μs)
    T_phi = 1 / ((1 / T_2) - (1 / 2 / T_1))  # Qubit pure dephasing time
    Gamma_phi = 1 / T_phi  # Qubit pure dephasing (phase relaxation) rate
    Gamma_minus = (4 * chi**2 * n / kappa) + (1 / 2 / T_2)  # Net cooling rate
    Gamma_plus = (kappa * chi**2 * n) / ((2 * Omega_R)**2 + (kappa / 2)**2) + (1 / 2 / T_2)  # Net heating rate

    # Define collapse operators
    c_ops = [
        np.sqrt(kappa) * d,
        np.sqrt(Gamma_phi / 2) * Z,
        np.sqrt(Gamma_minus) * minus,
        np.sqrt(Gamma_plus) * plus,
        ]

    # Define Hamiltonian
    H_c = - Delta_c * d.dag() * d
    H_q = - 0.5 * Delta_q_prime * Z
    H_R = - 0.5 * Omega_R * X
    H_chi = - chi * (np.conj(alpha) * d + alpha * d.dag() + d.dag() * d) * Z

    H = H_c + H_q + H_R + H_chi

    return qt.mesolve(H, psi0, t_range, c_ops, options=options), Gamma_minus, Gamma_plus

# Solve the master equations
results, Gamma_minuses, Gamma_pluses = np.array(qt.parallel_map(solve_master_equation, n_range)).T


# %%
# Get expectation values

states = [r.states for r in results]

n_exps = np.array([[qt.expect(d.dag() * d, s) for s in r] for r in states])
x_exps = np.array([[qt.expect(X, s) for s in r] for r in states])
z_exps = np.array([[qt.expect(Z, s) for s in r] for r in states])


# %%
# Plot the results

fig = plt.figure(figsize=(16, 23))
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)

for n, n_exp in list(zip(n_range, n_exps)):
    ax1.plot(t_range * 1e-3, n_exp, label=n)
ax1.set_xlabel('$t$ in μs')
ax1.set_ylabel('$\\langle n \\rangle$')
ax1.grid()
# ax1.legend()

ax2.plot(n_range, x_exps[:, -1], label='$\\langle X \\rangle$')
ax2.plot(n_range, z_exps[:, -1], label='$\\langle Z \\rangle$')
# ax2.plot(n_range, Gamma_minuses/(Gamma_minuses+Gamma_pluses), label='State purity')
ax2.set_xlabel('$\\overline{n}$')
ax2.grid()
ax2.set_xscale('log')
ax2.legend()

ax3.plot(n_range, Gamma_minuses / 2 / np.pi * 1e3, label='Cooling rate $\\Gamma_{-}$')
ax3.plot(n_range, Gamma_pluses / 2 / np.pi * 1e3, label='Heating rate $\\Gamma_{+}$')
ax3.set_xlabel('$\\overline{n}$')
ax3.set_ylabel('$\\Gamma_\\alpha / 2 \\pi$ in MHz')
ax3.grid()
ax3.set_xscale('log')
ax3.legend()

for n, x_exp in list(zip(n_range, x_exps)):
    ax4.plot(t_range * 1e-3, x_exp, label=n)
ax4.set_xlabel('$t$ in μs')
ax4.grid()
# ax4.legend()

plt.show()
