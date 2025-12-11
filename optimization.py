"""
Optimization algorithms for electrochemical model parameter fitting.

This module provides various optimization methods including particle swarm
optimization and MCMC for fitting parameters of electrochemical kinetics models.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from models import Model
import warnings
warnings.filterwarnings('ignore')


class TafelData:
    """Container for Tafel plot data"""
    def __init__(self, eta, lnk):
        self.eta = np.array(eta)
        self.lnk = np.array(lnk)


def load_datasets():
    """Load all three datasets and return as TafelData objects"""
    datasets = {}
    
    # Load Cell A data
    data_A = pd.read_csv('Cell-A-Tafel.csv')
    datasets['Cell A'] = TafelData(data_A['Overpotential'].values, data_A['Ln(k)'].values)
    
    # Load Cell B data  
    data_B = pd.read_csv('Cell-B-Tafel.csv')
    datasets['Cell B'] = TafelData(data_B['Overpotential'].values, data_B['Ln(k)'].values)
    
    # Load Cell C data
    data_C = pd.read_csv('Cell-C-Tafel.csv')
    datasets['Cell C'] = TafelData(data_C['Overpotential'].values, data_C['Ln(k)'].values)
    
    return datasets


def average_datasets(datasets):
    """
    Average the three datasets by interpolating all to a common overpotential grid
    and taking the mean ln(k) values.
    """
    # Create a common overpotential grid spanning all datasets
    all_eta = np.concatenate([data.eta for data in datasets.values()])
    eta_min, eta_max = np.min(all_eta), np.max(all_eta)
    eta_common = np.linspace(eta_min, eta_max, 100)
    
    # Interpolate each dataset to the common grid
    lnk_interpolated = []
    for data in datasets.values():
        lnk_interp = np.interp(eta_common, data.eta, data.lnk)
        lnk_interpolated.append(lnk_interp)
    
    # Average the interpolated ln(k) values
    lnk_avg = np.mean(lnk_interpolated, axis=0)
    
    return TafelData(eta_common, lnk_avg)


def residual_function(params, data, model_type='MHC2', mask_threshold=-8.0, k02_fixed=2.093e-4):
    """
    Calculate residuals between experimental data and model predictions.
    
    Parameters:
    -----------
    params : array-like
        Model parameters depending on model type:
        - BV: [alpha, k01, k02]
        - Marcus: [lambda_, k01, k02]
        - MHC: [lambda_, k01, k02]
        - MHC2: [k01, C, lambda_] (k02 is fixed)
    data : TafelData
        Experimental data object
    model_type : str
        Type of model ('BV', 'Marcus', 'MHC', or 'MHC2')
    mask_threshold : float
        Exclude data points with ln(k) below this value
    k02_fixed : float
        Fixed value for k02 parameter (only used for MHC2)
        
    Returns:
    --------
    residuals : array
        Residuals between model and experimental data
    """
    try:
        if model_type == 'BV':
            alpha, k01, k02 = params
            if alpha <= 0 or alpha >= 1 or k01 <= 0 or k02 <= 0:
                return np.array([1e6])
            model = Model(model='BV', k01=k01, k02=k02, alpha=alpha, 
                         eta=data.eta, origin_eta=True)
        elif model_type == 'Marcus':
            lambda_, k01, k02 = params
            if lambda_ <= 0 or k01 <= 0 or k02 <= 0:
                return np.array([1e6])
            model = Model(model='Marcus', k01=k01, k02=k02, lambda_=lambda_, 
                         eta=data.eta, origin_eta=True)
        elif model_type == 'MHC':
            lambda_, k01, C = params
            if lambda_ <= 0 or k01 <= 0 or C <= 0:
                return np.array([1e6])
            # MHC with k01 and C (concentration ratio) - use MHC2 model
            model = Model(model='MHC2', k01=k01, k02=k01, lambda_=lambda_, 
                         eta=data.eta, C=C, origin_eta=True)
        elif model_type == 'MHC2' and len(params) == 3:
            k01, C, lambda_ = params
            k02 = k02_fixed  # Use fixed k02 value
            if k01 <= 0 or C <= 0 or lambda_ <= 0:
                return np.array([1e6])
            model = Model(model='MHC2', k01=k01, k02=k02, lambda_=lambda_, 
                         eta=data.eta, C=C, origin_eta=True)
        else:
            raise ValueError(f"Invalid model type {model_type} or parameter count {len(params)}")
        
        # Get model predictions
        eta_model, lnk_model = model.ln_k()
        
        # Interpolate model predictions to experimental overpotentials
        lnk_pred = np.interp(data.eta, eta_model, lnk_model)
        
        # Apply mask to exclude low ln(k) values
        mask = data.lnk >= mask_threshold
        residuals = (data.lnk[mask] - lnk_pred[mask])
        
        # Handle case where mask excludes all data
        if len(residuals) == 0:
            return np.array([1e6])  # Large residual if no valid data
            
        return residuals
        
    except Exception as e:
        # Return large residuals for invalid parameters
        return np.array([1e6])


def objective_function(params, data, model_type='MHC2', mask_threshold=-8.0, k02_fixed=2.093e-4):
    """
    Objective function (sum of squared residuals) for optimization.
    """
    residuals = residual_function(params, data, model_type, mask_threshold, k02_fixed)
    return np.sum(residuals**2)


class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization for model parameter fitting.
    """
    
    def __init__(self, n_particles=30, n_iterations=100, w=0.5, c1=1.5, c2=1.5):
        """
        Parameters:
        -----------
        n_particles : int
            Number of particles in the swarm
        n_iterations : int
            Number of iterations
        w : float
            Inertia weight
        c1, c2 : float
            Acceleration coefficients
        """
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
    def optimize(self, data, model_type='MHC2', bounds=None, mask_threshold=-8.0, k02_fixed=2.093e-4, verbose=True):
        """
        Run particle swarm optimization.
        
        Parameters:
        -----------
        data : TafelData
            Experimental data
        model_type : str
            Type of model ('BV', 'Marcus', 'MHC', or 'MHC2')
        bounds : list of tuples
            Parameter bounds [(min1, max1), (min2, max2), ...]
        mask_threshold : float
            Exclude data points with ln(k) below this value
        k02_fixed : float
            Fixed value for k02 (only used for MHC2)
        verbose : bool
            Print progress information
            
        Returns:
        --------
        best_params : array
            Best parameters found
        best_cost : float
            Best objective function value
        cost_history : list
            History of best costs per iteration
        """
        if bounds is None:
            if model_type == 'BV':
                bounds = [(0.1, 0.9), (1e-8, 1e-1), (1e-8, 1e-1)]  # alpha, k01, k02
            elif model_type == 'Marcus':
                bounds = [(1.0, 30.0), (1e-8, 1e-1), (1e-8, 1e-1)]  # lambda_, k01, k02
            elif model_type == 'MHC':
                bounds = [(1.0, 30.0), (1e-8, 1e-1), (0.1, 2.0)]  # lambda_, k01, C
            elif model_type == 'MHC2':
                bounds = [(1e-6, 1e-2), (0.1, 1.0), (5.0, 20.0)]  # k01, C, lambda_
            else:
                bounds = [(1e-6, 1e-2), (1e-6, 1e-2), (5.0, 20.0)]  # default
        
        n_params = len(bounds)
        
        # Initialize particles
        particles = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(self.n_particles, n_params)
        )
        
        velocities = np.zeros((self.n_particles, n_params))
        
        # Initialize personal and global bests
        personal_best_positions = particles.copy()
        personal_best_costs = np.array([
            objective_function(p, data, model_type, mask_threshold, k02_fixed) 
            for p in particles
        ])
        
        global_best_idx = np.argmin(personal_best_costs)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_cost = personal_best_costs[global_best_idx]
        
        cost_history = [global_best_cost]
        
        if verbose:
            print(f"PSO: Initial best cost: {global_best_cost:.6f}")
        
        # Main PSO loop
        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                # Update velocity
                r1, r2 = np.random.random(n_params), np.random.random(n_params)
                
                velocities[i] = (self.w * velocities[i] + 
                               self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                               self.c2 * r2 * (global_best_position - particles[i]))
                
                # Update position
                particles[i] += velocities[i]
                
                # Apply bounds
                for j in range(n_params):
                    particles[i, j] = np.clip(particles[i, j], bounds[j][0], bounds[j][1])
                
                # Evaluate fitness
                cost = objective_function(particles[i], data, model_type, mask_threshold, k02_fixed)
                
                # Update personal best
                if cost < personal_best_costs[i]:
                    personal_best_costs[i] = cost
                    personal_best_positions[i] = particles[i].copy()
                    
                    # Update global best
                    if cost < global_best_cost:
                        global_best_cost = cost
                        global_best_position = particles[i].copy()
            
            cost_history.append(global_best_cost)
            
            if verbose and (iteration + 1) % 20 == 0:
                print(f"PSO: Iteration {iteration + 1}/{self.n_iterations}, "
                      f"Best cost: {global_best_cost:.6f}")
        
        if verbose:
            print(f"PSO: Final best cost: {global_best_cost:.6f}")
            
        return global_best_position, global_best_cost, cost_history


class MCMCOptimizer:
    """
    Markov Chain Monte Carlo optimization for model parameter fitting.
    """
    
    def __init__(self, n_samples=5000, burn_in=1000, proposal_std=None):
        """
        Parameters:
        -----------
        n_samples : int
            Number of MCMC samples
        burn_in : int
            Number of burn-in samples to discard
        proposal_std : array-like
            Standard deviation for proposal distribution
        """
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.proposal_std = proposal_std
        
    def log_likelihood(self, params, data, model_type='MHC2', mask_threshold=-8.0, k02_fixed=2.093e-4, sigma=1.0):
        """
        Log-likelihood function assuming Gaussian errors.
        """
        residuals = residual_function(params, data, model_type, mask_threshold, k02_fixed)
        
        if len(residuals) == 1 and residuals[0] == 1e6:
            return -np.inf  # Invalid parameters
        
        # Gaussian log-likelihood
        log_lik = -0.5 * np.sum(residuals**2) / (sigma**2)
        log_lik -= 0.5 * len(residuals) * np.log(2 * np.pi * sigma**2)
        
        return log_lik
    
    def log_prior(self, params, bounds):
        """
        Log-prior (uniform within bounds).
        """
        for i, (param, (low, high)) in enumerate(zip(params, bounds)):
            if not (low <= param <= high):
                return -np.inf
        return 0.0
    
    def log_posterior(self, params, data, bounds, model_type='MHC2', mask_threshold=-8.0, k02_fixed=2.093e-4, sigma=1.0):
        """
        Log-posterior = log-likelihood + log-prior.
        """
        log_prior_val = self.log_prior(params, bounds)
        if log_prior_val == -np.inf:
            return -np.inf
            
        log_lik_val = self.log_likelihood(params, data, model_type, mask_threshold, k02_fixed, sigma)
        return log_lik_val + log_prior_val
    
    def optimize(self, data, model_type='MHC2', bounds=None, initial_params=None, 
                mask_threshold=-8.0, k02_fixed=2.093e-4, sigma=1.0, verbose=True):
        """
        Run MCMC sampling.
        
        Parameters:
        -----------
        data : TafelData
            Experimental data
        model_type : str
            Type of model ('MHC' or 'MHC2')
        bounds : list of tuples
            Parameter bounds
        initial_params : array-like
            Starting parameters
        mask_threshold : float
            Exclude data points with ln(k) below this value
        k02_fixed : float
            Fixed value for k02 (only used for MHC2)
        sigma : float
            Assumed measurement error standard deviation
        verbose : bool
            Print progress information
            
        Returns:
        --------
        best_params : array
            MAP (maximum a posteriori) parameters
        samples : array
            MCMC samples (after burn-in)
        log_posterior_history : list
            History of log-posterior values
        """
        if bounds is None:
            if model_type == 'BV':
                bounds = [(0.1, 0.9), (1e-8, 1e-1), (1e-8, 1e-1)]  # alpha, k01, k02
            elif model_type == 'Marcus':
                bounds = [(1.0, 30.0), (1e-8, 1e-1), (1e-8, 1e-1)]  # lambda_, k01, k02
            elif model_type == 'MHC':
                bounds = [(1.0, 30.0), (1e-8, 1e-1), (0.1, 2.0)]  # lambda_, k01, C
            elif model_type == 'MHC2':
                bounds = [(1e-6, 1e-2), (0.1, 1.0), (5.0, 20.0)]  # k01, C, lambda_
            else:
                bounds = [(1e-6, 1e-2), (1e-6, 1e-2), (5.0, 20.0)]  # default
        
        n_params = len(bounds)
        
        if self.proposal_std is None:
            # Set proposal std to ~1% of parameter range
            self.proposal_std = np.array([(b[1] - b[0]) * 0.01 for b in bounds])
        
        if initial_params is None:
            # Random initial parameters within bounds
            initial_params = np.array([
                np.random.uniform(b[0], b[1]) for b in bounds
            ])
        
        # Initialize
        current_params = initial_params.copy()
        current_log_posterior = self.log_posterior(
            current_params, data, bounds, model_type, mask_threshold, k02_fixed, sigma
        )
        
        samples = []
        log_posterior_history = []
        n_accepted = 0
        
        if verbose:
            print(f"MCMC: Starting with log-posterior: {current_log_posterior:.2f}")
        
        for i in range(self.n_samples):
            # Propose new parameters
            proposal = current_params + np.random.normal(0, self.proposal_std)
            
            # Calculate log-posterior for proposal
            proposal_log_posterior = self.log_posterior(
                proposal, data, bounds, model_type, mask_threshold, k02_fixed, sigma
            )
            
            # Accept/reject step
            if proposal_log_posterior > current_log_posterior:
                # Accept
                current_params = proposal
                current_log_posterior = proposal_log_posterior
                n_accepted += 1
            else:
                # Accept with probability exp(delta)
                delta = proposal_log_posterior - current_log_posterior
                if np.log(np.random.random()) < delta:
                    current_params = proposal
                    current_log_posterior = proposal_log_posterior
                    n_accepted += 1
            
            # Store sample (after burn-in)
            if i >= self.burn_in:
                samples.append(current_params.copy())
            
            log_posterior_history.append(current_log_posterior)
            
            if verbose and (i + 1) % 1000 == 0:
                acceptance_rate = n_accepted / (i + 1)
                print(f"MCMC: Sample {i + 1}/{self.n_samples}, "
                      f"Acceptance rate: {acceptance_rate:.3f}, "
                      f"Log-posterior: {current_log_posterior:.2f}")
        
        samples = np.array(samples)
        
        # Find MAP estimate (maximum a posteriori)
        best_idx = np.argmax(log_posterior_history[self.burn_in:])
        best_params = samples[best_idx]
        
        acceptance_rate = n_accepted / self.n_samples
        if verbose:
            print(f"MCMC: Final acceptance rate: {acceptance_rate:.3f}")
            print(f"MCMC: MAP estimate found")
        
        return best_params, samples, log_posterior_history


def compare_optimizers(data, model_type='MHC2', mask_threshold=-8.0, k02_fixed=2.093e-4, verbose=True):
    """
    Compare PSO and MCMC optimization results.
    """
    if verbose:
        print("="*60)
        print(f"COMPARING OPTIMIZATION METHODS FOR {model_type} MODEL")
        if model_type == 'MHC2':
            print(f"Fitting parameters: k01, C, λ (k02 fixed at {k02_fixed:.6e})")
        print("="*60)
    
    # Run PSO
    if verbose:
        print("\n--- PARTICLE SWARM OPTIMIZATION ---")
    
    pso = ParticleSwarmOptimizer(n_particles=30, n_iterations=100)
    pso_params, pso_cost, pso_history = pso.optimize(
        data, model_type=model_type, mask_threshold=mask_threshold, 
        k02_fixed=k02_fixed, verbose=verbose
    )
    
    # Run MCMC (use PSO result as starting point)
    if verbose:
        print("\n--- MARKOV CHAIN MONTE CARLO ---")
    
    mcmc = MCMCOptimizer(n_samples=3000, burn_in=500)
    mcmc_params, mcmc_samples, mcmc_history = mcmc.optimize(
        data, model_type=model_type, initial_params=pso_params,
        mask_threshold=mask_threshold, k02_fixed=k02_fixed, verbose=verbose
    )
    
    # Calculate final costs
    pso_final_cost = objective_function(pso_params, data, model_type, mask_threshold, k02_fixed)
    mcmc_final_cost = objective_function(mcmc_params, data, model_type, mask_threshold, k02_fixed)
    
    if verbose:
        print("\n--- RESULTS COMPARISON ---")
        print(f"PSO  - Final cost: {pso_final_cost:.6f}")
        print(f"MCMC - Final cost: {mcmc_final_cost:.6f}")
        
        if model_type == 'MHC2':
            param_names = ['k01', 'C', 'lambda_']
        else:
            param_names = ['k01', 'k02', 'lambda_']
        
        print("\nParameter estimates:")
        for i, name in enumerate(param_names):
            if i < len(pso_params) and i < len(mcmc_params):
                print(f"{name:8s}: PSO={pso_params[i]:.6f}, MCMC={mcmc_params[i]:.6f}")
        
        if model_type == 'MHC2':
            print(f"{'k02':8s}: Fixed at {k02_fixed:.6e}")
    
    return {
        'pso': {'params': pso_params, 'cost': pso_final_cost, 'history': pso_history},
        'mcmc': {'params': mcmc_params, 'cost': mcmc_final_cost, 'samples': mcmc_samples, 'history': mcmc_history}
    }


def plot_optimization_results(results, data, model_type='MHC2', mask_threshold=-8.0, k02_fixed=2.093e-4):
    """
    Plot optimization results and model fits.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: PSO convergence
    axes[0, 0].plot(results['pso']['history'])
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Objective Function')
    axes[0, 0].set_title('PSO Convergence')
    axes[0, 0].set_yscale('log')
    
    # Plot 2: MCMC convergence
    axes[0, 1].plot(results['mcmc']['history'])
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('Log-Posterior')
    axes[0, 1].set_title('MCMC Log-Posterior History')
    
    # Plot 3: Model fits comparison
    mask = data.lnk >= mask_threshold
    axes[1, 0].scatter(data.eta[mask], data.lnk[mask], alpha=0.7, label='Experimental Data', s=50)
    
    # PSO fit
    pso_params = results['pso']['params']
    if model_type == 'MHC2':
        k01, C, lambda_ = pso_params
        k02 = k02_fixed
        model_pso = Model(model='MHC2', k01=k01, k02=k02, lambda_=lambda_, 
                         eta=data.eta, C=C, origin_eta=True)
    else:
        k01, k02, lambda_ = pso_params
        model_pso = Model(model='MHC', k01=k01, k02=k02, lambda_=lambda_, 
                         eta=data.eta, origin_eta=True)
    
    eta_model, lnk_pso = model_pso.ln_k()
    axes[1, 0].plot(eta_model, lnk_pso, label='PSO Fit', linewidth=2)
    
    # MCMC fit
    mcmc_params = results['mcmc']['params']
    if model_type == 'MHC2':
        k01, C, lambda_ = mcmc_params
        k02 = k02_fixed
        model_mcmc = Model(model='MHC2', k01=k01, k02=k02, lambda_=lambda_, 
                          eta=data.eta, C=C, origin_eta=True)
    else:
        k01, k02, lambda_ = mcmc_params
        model_mcmc = Model(model='MHC', k01=k01, k02=k02, lambda_=lambda_, 
                          eta=data.eta, origin_eta=True)
    
    eta_model, lnk_mcmc = model_mcmc.ln_k()
    axes[1, 0].plot(eta_model, lnk_mcmc, label='MCMC Fit', linewidth=2, linestyle='--')
    
    axes[1, 0].set_xlabel('Overpotential η (V)')
    axes[1, 0].set_ylabel('ln(k) (s⁻¹)')
    axes[1, 0].set_title(f'{model_type} Model Fits')
    axes[1, 0].legend()
    
    # Plot 4: MCMC parameter distributions
    if 'samples' in results['mcmc']:
        samples = results['mcmc']['samples']
        if model_type == 'MHC2':
            param_names = ['k01', 'C', 'λ']
        else:
            param_names = ['k01', 'k02', 'λ']
        
        # Plot first parameter distribution as example
        axes[1, 1].hist(samples[:, 0], bins=50, alpha=0.7, density=True)
        axes[1, 1].axvline(mcmc_params[0], color='red', linestyle='--', 
                          label=f'MAP: {mcmc_params[0]:.6f}')
        axes[1, 1].set_xlabel(f'{param_names[0]}')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title(f'MCMC: {param_names[0]} Distribution')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig