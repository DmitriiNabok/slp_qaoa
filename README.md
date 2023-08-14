# Segmentation and Labeling problem solved with QAOA

Basic QAOA workflow applied to solve the QUBO formulation of the Segmentation and Labeling Problem (SLP).

Benchmarking runs include:
* Computing the cost function energy surface for p=1.
* Computing the energy surface for p=2, 3, 4 for the last 2 values keeping the rest of the variational parameters fixed at their optimal values.
* Resources (circuit depths and the number of the CNOT gates) for different problem sizes.
* Benchmarking two versions of QAOA using the standard X and XY mixers.
