# Physics-Lab-Work

This repository contains several physics projects completed during lab sessions at ESPCI Paris-PSL. Each project involves numerical modeling and simulation of physical systems.

---

## Projects overview

### Fluid Mechanics – Bénard–von Kármán Vortex Street

**File:** `rapport-fluid-mechanics.pdf`

This project investigates the numerical modeling of the Bénard–von Kármán instability — the formation of alternating vortices in the wake of a bluff body subject to fluid flow. The study uses the finite element method to solve the Navier–Stokes equations in confined geometries. By varying the Reynolds number, the transition from stable flow to vortex shedding is captured, and the influence of confinement on the critical instability threshold is analyzed.

---

### Thermal conduction in particle systems

**Folder:** `Conduction/`  
**Contains:**  
- `rapport-conduction.pdf`  
- `codes/` (Python scripts)

This project models a gas of hard spheres bouncing in a square box to simulate heat conduction between a hot and cold wall. The kinetic energy of particles is adjusted upon collision with the thermal walls, inducing a density gradient. The thermal conductivity λ is computed from energy flux measurements. The study explores how λ depends on particle number and size, and investigates the effect of a large, heavy intruder particle on mobility and energy transfer.

---

### Quantum harmonic oscillator – MATLAB simulation

**File:** `rapport-quantum-matlab.pdf`

This project focuses on the numerical solution of the confined quantum harmonic oscillator using a basis of infinite square well eigenstates. The goal is to compute and analyze the energy eigenvalues and eigenfunctions numerically and compare them with analytical results. The study shows how the harmonic potential influences the system’s spectrum and eigenstates as the matrix size increases.

---

### Solving the 2D Poisson equation

**Folder:** `Poisson-equation/`  
**Contains:**  
- `notebook-poisson.ipynb`  
- `rapport-poisson.pdf`

This work involves solving the 2D Poisson equation, relevant in electrostatics, for a square domain representing a parallel-plate capacitor. Dirichlet boundary conditions are applied, and the equation is discretized using a second-order Taylor expansion. Multiple solution methods are implemented and compared: sparse matrix solvers (`spsolve`), Gauss pivoting, and Jacobi iteration. The performance and accuracy of these methods are discussed.

---

## Tools & Languages

- Python
- MATLAB
- Finite element method
- Jupyter notebooks
- PDF reports

---

All reports are written in French.
