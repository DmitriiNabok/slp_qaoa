import numpy as np
import sys, os
import matplotlib.pyplot as plt

from qiskit import Aer, execute


def linear_ramp_params(p: int, slope: float = 0.7, beta_sign: float = 1.0) -> np.ndarray:
    """Linear ramp scheme for the QAOA parameters initialization"""
    if not isinstance(p, int) or p <= 0:
        raise ValueError("p must be a positive integer")
    if not isinstance(slope, float) or slope <= 0:
        raise ValueError("slope must be a positive float")

    time = slope * p
    # create evenly spaced timelayers at the centers of p intervals
    dt = time / p
    # fill betas, gammas_singles and gammas_pairs
    betas = np.linspace(
        (dt / time) * (time * (1 - 0.5 / p)), (dt / time) * (time * 0.5 / p), p
    )
    gammas = betas[::-1]
    params = np.vstack((beta_sign * betas, gammas)).ravel(order="F")
    return params


def get_counts(
    ansatz, params, backend=Aer.get_backend("qasm_simulator"), n_shots=1024, seed=12345
):
    """ """
    qc = ansatz.copy()
    qc.measure_all()

    job = execute(
        qc.assign_parameters(parameters=params),
        backend,
        shots=n_shots,
        seed_simulator=seed,
        seed_transpiler=seed,
    )
    counts = job.result().get_counts()

    return counts


def plot_solutions(dict_sols, exact, width=0.1):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    for label, sol in dict_sols.items():
        x = [s["obj"] for s in sol]
        y = [s["prob"] for s in sol]
        z = [s["feas"] for s in sol]

        ax[0].plot(x, color="grey")
        ax[0].set_xlabel("Samples")
        ax[0].set_ylabel("Objective")

        ax[1].bar(x, y, width=width, label=label, color="grey")

        x_feas, y_feas = [], []
        for _x, _y, _z in zip(x, y, z):
            if _z:
                x_feas.append(_x)
                y_feas.append(_y)

        ax[1].bar(x_feas, y_feas, width=width, label=label + " (feas)", color="blue")

        ax[1].set_xlabel("Objective")
        ax[1].set_ylabel("Quasi probability")

    # exact solution
    ax[0].axhline(y=exact, ls=":", color="k")
    ax[1].axvline(x=exact, ls=":", color="k", label="exact")

    plt.legend()
    plt.show()