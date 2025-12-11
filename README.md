# Electrochemical Kinetics Parameter Fitting

A Python framework for analyzing electrochemical kinetics data using Butler-Volmer (BV), Marcus, and Marcus-Hush-Chidsey (MHC) models. This project implements parameter fitting via differential evolution optimization and MCMC sampling to extract kinetic parameters from Tafel plot data.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Models](#models)
- [Data Format](#data-format)
- [Notebooks](#notebooks)
- [Results](#results)
- [References](#references)
- [License](#license)

## Overview

This project provides tools for fitting electrochemical kinetic models to experimental Tafel data (ln(k) vs. overpotential η). The primary focus is on the Marcus-Hush-Chidsey (MHC) model, which provides a quantum mechanical treatment of electron transfer kinetics.

### Key Parameters

- **k₀** (k01, k02): Exchange rate constants (s⁻¹)
- **λ**: Reorganization energy (dimensionless, λ̃ = λ/(RT))
- **C**: Concentration ratio parameter (for asymmetric MHC2 model)
- **α**: Transfer coefficient (Butler-Volmer model)

## Features

- **Multiple kinetic models**: Butler-Volmer, Marcus, MHC, and MHC2 (asymmetric)
- **Global optimization**: Differential evolution and particle swarm optimization (PSO)
- **Bayesian inference**: MCMC sampling for parameter uncertainty quantification
- **Temperature analysis**: Arrhenius analysis for activation energy extraction
- **Electrolyte comparison**: Systematic comparison across different electrolyte systems
- **Publication-quality plots**: Automated visualization of fits and parameter trends

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Dependencies

```bash
pip install numpy scipy matplotlib pandas
```

Or create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Required packages

- `numpy` - Numerical computations
- `scipy` - Optimization and statistical functions
- `matplotlib` - Visualization
- `pandas` - Data handling

## Project Structure

```
2.625-Project/
├── models.py                 # Core electrochemical kinetic models
├── optimization.py           # PSO, MCMC, and fitting utilities
├── mcmc_analysis.py          # MCMC sampler implementation
├── reorganisation_energy.py  # Reorganization energy calculations
│
├── notebooks/                # Legacy notebooks
├── 01_model_setup.ipynb      # Initial model exploration
├── 02_newer_models.ipynb     # Extended model development
├── 03_adjusted_eta_residuals.ipynb  # Residual analysis
├── 04_parameter_fitting.ipynb       # MHC2 parameter fitting
├── 05_mhc_parameter_fitting.ipynb   # MHC (symmetric) fitting
├── 06_electrolyte_anaylysis.ipynb   # Electrolyte comparison
├── 07_temperature_analysis.ipynb    # Temperature dependence
│
├── Cell-A-Tafel.csv          # Experimental data (Cell A)
├── Cell-B-Tafel.csv          # Experimental data (Cell B)
├── Cell-C-Tafel.csv          # Experimental data (Cell C)
├── Electrolytes/             # Electrolyte-specific data
├── Temps-data/               # Temperature-dependent data
│   ├── Cell A/               # 30°C, 40°C, 50°C data
│   ├── Cell B/
│   └── Cell C/
│
├── Papers/                   # Reference literature
├── MCMC MATLAB Scripts/      # Original MATLAB implementation
└── README.md
```

## Usage

### Basic Model Evaluation

```python
from models import Model
import numpy as np

# Create an MHC2 model
model = Model(
    model='MHC2',
    k01=2.1e-4,      # s⁻¹
    k02=2.1e-4,      # s⁻¹
    lambda_=8.3,     # dimensionless
    C=0.5,           # concentration ratio
    eta=np.linspace(-0.5, 0.5, 200)
)

# Get rate constants
eta, k = model.rate_constant()

# Get ln(k) for Tafel plot
eta, lnk = model.ln_k()
```

### Parameter Fitting

```python
from scipy.optimize import differential_evolution
from models import Model

# Define objective function
def objective(params, data):
    k01, C = params
    model = Model(model='MHC2', k01=k01, k02=2.093e-4, 
                  lambda_=8.3, C=C, eta=data.eta, origin_eta=True)
    eta_model, lnk_model = model.ln_k()
    lnk_pred = np.interp(data.eta, eta_model, lnk_model)
    return np.sum((data.lnk - lnk_pred)**2)

# Run optimization
bounds = [(1e-6, 1e-2), (0.1, 2.0)]  # [k01, C]
result = differential_evolution(objective, bounds, args=(data,))
```

## Models

### Butler-Volmer (BV)

Classical electrochemical kinetics with exponential dependence on overpotential:

$$k = k_0 \left[ \exp(-\alpha \tilde{\eta}) - \exp((1-\alpha) \tilde{\eta}) \right]$$

### Marcus

Quantum mechanical electron transfer with parabolic free energy surfaces:

$$k = k_0 \exp\left(-\frac{(\tilde{\lambda} - \tilde{\eta})^2}{4\tilde{\lambda}}\right)$$

### Marcus-Hush-Chidsey (MHC)

Accounts for electronic coupling to metal electrode states:

$$k(\tilde{\lambda}, \tilde{\eta}) = k_0 \tanh\left(\frac{\tilde{\eta}}{2}\right) \frac{\text{erfc}(\text{arg}(\tilde{\eta}))}{\text{erfc}(\text{arg}(0))}$$

### MHC2 (Asymmetric)

Extended MHC model with asymmetric concentrations (parameter C):

$$k = k_0 \cdot f(\tilde{\eta}, \tilde{\lambda}, C)$$

## Data Format

Input CSV files should contain two columns:

| Column | Description |
|--------|-------------|
| `x` | Overpotential η (V) |
| `y` | ln(k) (s⁻¹) |

Example:
```csv
x,y
-0.15,-5.2
-0.10,-5.8
-0.05,-7.1
...
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_model_setup.ipynb` | Initial model exploration and validation |
| `04_parameter_fitting.ipynb` | MHC2 model fitting with fixed λ comparison |
| `05_mhc_parameter_fitting.ipynb` | Symmetric MHC fitting with MCMC |
| `06_electrolyte_anaylysis.ipynb` | Multi-electrolyte comparison (PC, DEC, EC/DEC, FEC, DME) |
| `07_temperature_analysis.ipynb` | Temperature dependence and Arrhenius analysis |

## Results

### Key Findings

**MHC Model Parameters (averaged across cells):**

| Temperature | k₀₁ (s⁻¹) | C | R² |
|-------------|-----------|------|------|
| 30°C | 2.09×10⁻⁴ | 0.51 | 0.956 |
| 40°C | 2.06×10⁻⁴ | 0.64 | 0.950 |
| 50°C | 2.52×10⁻⁴ | 0.62 | 0.940 |

**Activation Energy (Arrhenius analysis):**
- Combined Ea ≈ 76 meV (7.4 kJ/mol)
- Fixed λ = 8.3 (consistent with literature)

### Output Files

- `temperature_analysis_results_individual.csv` - Fitted parameters
- `electrolyte_model_fitting_results.csv` - Electrolyte comparison
- `*.png` - Generated figures

## References

1. Bai, P., & Bazant, M. Z. (2014). Charge transfer kinetics at the solid–solid interface in porous electrodes. *Nature Communications*, 5, 3585.

2. Zeng, Y., Smith, R. B., Bai, P., & Bazant, M. Z. (2014). Simple formula for Marcus-Hush-Chidsey kinetics. *Journal of Electroanalytical Chemistry*, 735, 77-83.

3. Marcus, R. A. (1956). On the theory of oxidation-reduction reactions involving electron transfer. *The Journal of Chemical Physics*, 24(5), 966-978.

## License

This project is part of MIT course 2.625 (Electrochemical Energy Systems).

## Authors

- Felix Wang

## Acknowledgments

- Based on MCMC MATLAB scripts by Matthew Ashner (2018)
- Inspired by theoretical frameworks from Bazant group at MIT
