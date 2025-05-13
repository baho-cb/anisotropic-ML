# nep_simulator

Anisotropic particle simulation code with NeuroEvolutionPotential.

## Project Structure
---

### 01. **Data Generation**  
Scripts and utilities for generating training and validation datasets for anisotropic particle systems.

**Packages:**  
- `pyDOE` – for Latin Hypercube Sampling  
- `cupy` – for GPU-accelerated computations  
- `gsd` – for reading particle data

---

### 02. **Comparisons with SOAP**  
Generate SOAP descriptors and compare the performance of the following models:  
Linear Regression, Kernel Ridge Regression, Gaussian Process Regression, and Feedforward Neural Networks.  
SOAP hyperparameters are optimized via Bayesian optimization using `scikit-optimize`.

**Packages:**  
- `skopt`  
- `torch`  
- `sklearn`  
- `dscribe` – for SOAP descriptor generation  
- `ase` – for SOAP descriptor generation

---

### 03. **SchNet vs DimeNet++ vs FFNN**  
Evaluation of the predictive performance of message-passing neural networks (MPNNs) and feedforward neural networks (FFNNs).

**Packages:**  
- `torch`  
- `torch_geometric`

---

### 04. **MLFlow Experiments for Tetrahedra**  
MLFlow setup for logging and tracking model training sessions and performance metrics on tetrahedral geometries.

---

### 05. **MLFlow – Point Placement Experiments for Twisted Cylinders**  
Experiments using MLFlow to investigate the effect of particle placement strategies on twisted cylindrical systems.

