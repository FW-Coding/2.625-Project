"""
MCMC Analysis for Electrochemical Kinetics
==========================================

Python implementation of MCMC-based parameter estimation for electrochemical
kinetic models, converted from MATLAB scripts by Matthew Ashner (2018).

This module provides:
1. Biexciton/exciton kinetic model evaluation with Poisson statistics
2. Affine-invariant ensemble MCMC sampler (Goodman & Weare algorithm)
3. Tempered MCMC for finding global likelihood maxima
4. Parameter estimation with uncertainty quantification

References:
- Ashner and Tisdale, JPC 2018
- Goodman & Weare (2010), Comm. App. Math. Comp. Sci.
- Foreman-Mackey et al. (2013), emcee: The MCMC Hammer
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.linalg import eig
from scipy.stats import gaussian_kde
import warnings
from tqdm import tqdm
from typing import Callable, List, Tuple, Optional, Union
import time


class BiexcitonKinetics:
    """
    Kinetic model evaluator for biexciton (BX) and exciton (X) states.
    
    Model: BX --k1--> X --k2--> (decay)
    
    Uses Poisson statistics for fluence-dependent initial conditions:
    - BX yield: 1 - P(N≤1) where N ~ Poisson(XC * power)
    - X yield: P(N=1) where N ~ Poisson(XC * power)
    """
    
    def __init__(self, data: np.ndarray, sigma: np.ndarray, t: np.ndarray, powers: np.ndarray):
        """
        Initialize the kinetic model evaluator.
        
        Parameters:
        -----------
        data : np.ndarray, shape (wavelengths, time_points, fluences)
            Fluence-dependent transient absorption data
        sigma : np.ndarray, same shape as data
            Standard deviations for each data point
        t : np.ndarray, shape (time_points,)
            Time delays in ps
        powers : np.ndarray, shape (fluences,)
            Pump powers in µW
        """
        self.data = data
        self.sigma = sigma
        self.t = t
        self.powers = powers
        self.specpts, self.tpts, self.numtrace = data.shape
        
    def evaluate_log_likelihood(self, params: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate log likelihood for given parameters.
        
        Parameters:
        -----------
        params : np.ndarray, shape (3,)
            [log10(k1), log10(k2), XC] where:
            - k1, k2: rate constants in ps⁻¹
            - XC: cross section parameter in 1/µW
            
        Returns:
        --------
        log_prob : float
            Log likelihood value
        spec : np.ndarray, shape (wavelengths, 2)
            Component spectra for BX and X states
        datafit : np.ndarray, same shape as self.data
            Fitted TA data
        """
        # Convert log rate constants to linear space
        k1, k2 = 10**params[0], 10**params[1]
        XC = params[2]
        
        # Define kinetic model matrix: BX --k1--> X --k2--> decay
        K = np.array([[-k1, 0], [k1, -k2]])
        eigenvals, eigenvecs = eig(K)
        
        # Pre-compute eigenvalue dynamics
        krep = np.outer(eigenvals, self.t)
        eigdyn = np.exp(krep)  # Shape: (2, time_points)
        
        # Initialize arrays
        dyn = np.zeros((2, self.tpts, self.numtrace))
        datafit = np.zeros_like(self.data)
        
        # Evaluate kinetics with fluence-dependent initial conditions
        for i, power in enumerate(self.powers):
            # Poisson statistics for initial state populations
            lambda_param = XC * power
            BX_yield = 1 - poisson.cdf(1, lambda_param)  # P(N > 1)
            X_yield = poisson.pmf(1, lambda_param)       # P(N = 1)
            
            # Source term in component basis
            source = np.array([BX_yield, X_yield])
            
            # Transform to eigenbasis, apply dynamics, transform back
            # This is equivalent to: U * diag(U\source) * eigdyn
            source_eigen = np.linalg.solve(eigenvecs, source)
            A2 = eigenvecs @ np.diag(source_eigen)
            dyn[:, :, i] = A2 @ eigdyn
        
        # Reshape for linear least squares: (spectra, time*fluence)
        data_2d = self.data.reshape(self.specpts, self.tpts * self.numtrace)
        dyn_2d = dyn.reshape(2, self.tpts * self.numtrace)
        
        # Linear least squares to find component spectra
        spec = data_2d @ dyn_2d.T @ np.linalg.inv(dyn_2d @ dyn_2d.T)
        
        # Reconstruct fitted data
        for i in range(self.numtrace):
            datafit[:, :, i] = spec @ dyn[:, :, i]
        
        # Calculate log likelihood (negative chi-squared)
        log_prob = -np.sum((datafit - self.data)**2 / self.sigma**2)
        
        return log_prob, spec, datafit


class TemperedMCMC:
    """
    Tempered affine-invariant ensemble MCMC sampler.
    
    Implementation of the Goodman & Weare (2010) algorithm with optional
    tempering for finding global likelihood maxima.
    """
    
    def __init__(self, step_size: float = 2.5, thin_chain: int = 10, 
                 progress_bar: bool = True, parallel: bool = False):
        """
        Initialize the MCMC sampler.
        
        Parameters:
        -----------
        step_size : float, default 2.5
            Unit-less step size parameter
        thin_chain : int, default 10
            Thin chains by storing every N-th step
        progress_bar : bool, default True
            Show progress bar during sampling
        parallel : bool, default False
            Run walkers in parallel (not yet implemented)
        """
        self.step_size = step_size
        self.thin_chain = thin_chain
        self.progress_bar = progress_bar
        self.parallel = parallel
        
    def sample(self, minit: np.ndarray, log_prob_funs: List[Callable], 
               mccount: int, temper: bool = False, 
               burn_in: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run ensemble MCMC sampling.
        
        Parameters:
        -----------
        minit : np.ndarray, shape (n_params, n_walkers)
            Initial positions for walkers
        log_prob_funs : list of callables
            Functions returning log probabilities [log_prior, log_likelihood]
        mccount : int
            Total number of Monte Carlo proposals
        temper : bool, default False
            Use tempering with cooling schedule
        burn_in : float, default 0.0
            Fraction of chain to discard as burn-in
            
        Returns:
        --------
        models : np.ndarray, shape (n_params, n_walkers, n_samples)
            Sampled parameter chains
        log_p : np.ndarray, shape (n_log_p_funs, n_walkers, n_samples)
            Log probability chains
        """
        n_params, n_walkers = minit.shape
        n_keep = int(np.ceil(mccount / self.thin_chain / n_walkers))
        mccount = (n_keep - 1) * self.thin_chain + 1
        
        # Pre-allocate arrays
        models = np.full((n_params, n_walkers, n_keep), np.nan)
        models[:, :, 0] = minit
        
        n_p_fun = len(log_prob_funs)
        log_p = np.full((n_p_fun, n_walkers, n_keep), np.nan)
        
        # Calculate initial log probabilities
        for w in range(n_walkers):
            for f, func in enumerate(log_prob_funs):
                val = func(minit[:, w])
                if isinstance(val, bool):  # Handle logical constraints
                    val = -np.inf if not val else 0
                log_p[f, w, 0] = val
                
        if not np.all(np.isfinite(log_p[:, :, 0])):
            raise ValueError("Starting points must have finite log probabilities")
        
        # Set up tempering schedule
        if temper:
            T0 = 4000  # Initial temperature
            slope = 0.97
            beta = 1.0 / (T0 * slope**(np.arange(n_keep)) + 1)
        else:
            beta = np.ones(n_keep)
        
        # Initialize current state
        reject_count = np.zeros(n_walkers)
        cur_m = models[:, :, 0].copy()
        cur_log_p = log_p[:, :, 0].copy()
        
        total_count = n_walkers
        
        # Progress bar setup
        if self.progress_bar:
            pbar = tqdm(total=n_keep, desc="MCMC Sampling")
        
        # Main sampling loop
        for row in range(1, n_keep):
            for thin_step in range(self.thin_chain):
                # Generate proposals for all walkers
                partner_idx = (np.arange(n_walkers) + 
                              np.random.randint(1, n_walkers, n_walkers)) % n_walkers
                zz = ((self.step_size - 1) * np.random.rand(n_walkers) + 1)**2 / self.step_size
                
                proposed_m = (cur_m[:, partner_idx] - 
                             (cur_m[:, partner_idx] - cur_m) * zz[np.newaxis, :])
                
                log_rand = np.log(np.random.rand(n_p_fun + 1, n_walkers))
                
                # Evaluate proposals for each walker
                for w in range(n_walkers):
                    accept_step = True
                    proposed_log_p = np.full(n_p_fun, np.nan)
                    
                    # Check geometric constraint
                    if log_rand[0, w] < (n_params - 1) * np.log(zz[w]):
                        # Evaluate log probability functions in cascade
                        for f, func in enumerate(log_prob_funs):
                            proposed_log_p[f] = func(proposed_m[:, w])
                            
                            # Check acceptance criterion with tempering
                            if (log_rand[f + 1, w] > 
                                beta[row] * (proposed_log_p[f] - cur_log_p[f, w]) or
                                not np.isreal(proposed_log_p[f]) or 
                                np.isnan(proposed_log_p[f])):
                                accept_step = False
                                break
                    else:
                        accept_step = False
                    
                    # Update walker state
                    if accept_step:
                        cur_m[:, w] = proposed_m[:, w]
                        cur_log_p[:, w] = proposed_log_p
                    else:
                        reject_count[w] += 1
                
                total_count += n_walkers
            
            # Store thinned samples
            models[:, :, row] = cur_m
            log_p[:, :, row] = cur_log_p
            
            # Update progress
            if self.progress_bar:
                reject_rate = np.sum(reject_count) / total_count
                pbar.set_postfix(reject_rate=f"{reject_rate:.1%}")
                pbar.update(1)
        
        if self.progress_bar:
            pbar.close()
        
        # Apply burn-in
        if burn_in > 0:
            crop = int(np.ceil(n_keep * burn_in))
            models = models[:, :, crop:]
            log_p = log_p[:, :, crop:]
        
        return models, log_p


def run_mcmc_analysis(data: np.ndarray, sigma: np.ndarray, t: np.ndarray, 
                     powers: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray],
                     is_log: np.ndarray, n_walkers_1: int = 100, 
                     n_steps_1: int = 200000, n_walkers_2: int = 100, 
                     n_steps_2: int = 500000) -> dict:
    """
    Complete MCMC analysis workflow for biexciton kinetics.
    
    This function implements the full analysis pipeline:
    1. Tempered MCMC to find global likelihood maximum
    2. Focused MCMC sampling around the maximum
    3. Parameter estimation and uncertainty quantification
    
    Parameters:
    -----------
    data : np.ndarray, shape (wavelengths, time_points, fluences)
        Transient absorption data
    sigma : np.ndarray, same shape as data
        Data uncertainties
    t : np.ndarray, shape (time_points,)
        Time delays in ps
    powers : np.ndarray, shape (fluences,)
        Pump powers in µW
    bounds : tuple of np.ndarray
        (lower_bounds, upper_bounds) for parameters
    is_log : np.ndarray, dtype bool
        Which parameters are in log space
    n_walkers_1, n_steps_1 : int
        Number of walkers and steps for tempered MCMC
    n_walkers_2, n_steps_2 : int
        Number of walkers and steps for main MCMC
        
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'models_1', 'log_p_1': Tempered MCMC results
        - 'models_2', 'log_p_2': Main MCMC results  
        - 'best_params': Most likely parameter values
        - 'param_samples': Final parameter samples
        - 'spec_samples': Sample component spectra
        - 'data_samples': Sample fitted data
    """
    lb, ub = bounds
    n_params = len(lb)
    
    # Initialize kinetic model
    kinetics = BiexcitonKinetics(data, sigma, t, powers)
    
    # Define prior and likelihood functions
    def log_prior(params):
        return 0.0 if np.all((params >= lb) & (params <= ub)) else -np.inf
    
    def log_likelihood(params):
        try:
            log_prob, _, _ = kinetics.evaluate_log_likelihood(params)
            return log_prob
        except:
            return -np.inf
    
    log_prob_funs = [log_prior, log_likelihood]
    
    # Step 1: Tempered MCMC to find global maximum
    print("Step 1: Running tempered MCMC to find global likelihood maximum...")
    rand_init_1 = (np.random.rand(n_params, n_walkers_1) * 
                   (ub - lb)[:, np.newaxis] + lb[:, np.newaxis])
    
    sampler = TemperedMCMC(step_size=1.7)
    models_1, log_p_1 = sampler.sample(rand_init_1, log_prob_funs, n_steps_1, 
                                      temper=True, burn_in=0.25)
    
    # Find best region from tempered MCMC
    mean_log_p = np.mean(log_p_1[1, :, :], axis=1)  # Average likelihood per walker
    sorted_probs = np.sort(mean_log_p)
    prob_diffs = np.diff(sorted_probs)
    
    # Find threshold for high-likelihood region
    threshold_idx = np.where(prob_diffs >= 10)[0]
    if len(threshold_idx) > 0:
        bound = threshold_idx[-1] + 1
        good_walkers = np.argsort(mean_log_p)[bound:]
        post_sub = models_1[:, good_walkers, :]
    else:
        post_sub = models_1
    
    # Estimate most likely parameter values using kernel density estimation
    best_params = np.zeros(n_params)
    for i in range(n_params):
        samples = post_sub[i, :, :].flatten()
        kde = gaussian_kde(samples)
        x_test = np.linspace(samples.min(), samples.max(), 1000)
        density = kde(x_test)
        best_params[i] = x_test[np.argmax(density)]
    
    print(f"Best parameters found: {best_params}")
    
    # Step 2: Main MCMC around the best region
    print("Step 2: Running main MCMC around best region...")
    rand_init_2 = np.zeros((n_params, n_walkers_2))
    
    for i in range(n_params):
        if is_log[i]:
            # For log parameters, add small Gaussian noise
            rand_init_2[i, :] = best_params[i] + 0.002 * np.random.randn(n_walkers_2)
        else:
            # For linear parameters, add multiplicative noise
            rand_init_2[i, :] = best_params[i] * 10**(0.002 * np.random.randn(n_walkers_2))
    
    sampler.step_size = 2.2  # Adjust step size for better acceptance rate
    models_2, log_p_2 = sampler.sample(rand_init_2, log_prob_funs, n_steps_2, 
                                      temper=False, burn_in=0.1)
    
    # Extract final samples
    param_samples = models_2.reshape(n_params, -1)
    
    # Convert logged rate constants to time constants
    param_samples_converted = param_samples.copy()
    param_samples_converted[is_log, :] = 10**(-param_samples[is_log, :])  # Convert to time constants
    
    # Generate sample fits for visualization
    n_viz_samples = min(100, param_samples.shape[1])
    sample_indices = np.random.choice(param_samples.shape[1], n_viz_samples, replace=False)
    
    spec_samples = []
    data_samples = []
    
    print("Generating sample fits...")
    for i, idx in enumerate(sample_indices):
        _, spec, data_fit = kinetics.evaluate_log_likelihood(param_samples[:, idx])
        spec_samples.append(spec)
        data_samples.append(data_fit)
    
    spec_samples = np.array(spec_samples)  # Shape: (n_samples, wavelengths, 2)
    data_samples = np.array(data_samples)  # Shape: (n_samples, wavelengths, time, fluence)
    
    return {
        'models_1': models_1,
        'log_p_1': log_p_1,
        'models_2': models_2, 
        'log_p_2': log_p_2,
        'best_params': best_params,
        'param_samples': param_samples,
        'param_samples_converted': param_samples_converted,
        'spec_samples': spec_samples,
        'data_samples': data_samples,
        'kinetics': kinetics
    }


def plot_mcmc_results(results: dict, param_names: List[str] = None, 
                     save_figs: bool = False, fig_prefix: str = "mcmc"):
    """
    Generate diagnostic and result plots from MCMC analysis.
    
    Parameters:
    -----------
    results : dict
        Results from run_mcmc_analysis
    param_names : list of str, optional
        Names for parameters (default: ['log(k1)', 'log(k2)', 'XC'])
    save_figs : bool, default False
        Save figures to files
    fig_prefix : str, default "mcmc"
        Prefix for saved figure filenames
    """
    if param_names is None:
        param_names = ['log(k₁)', 'log(k₂)', 'XC']
    
    param_samples = results['param_samples']
    n_params = param_samples.shape[0]
    
    # Plot 1: Parameter traces
    fig, axes = plt.subplots(n_params, 1, figsize=(12, 4*n_params))
    if n_params == 1:
        axes = [axes]
    
    for i in range(n_params):
        axes[i].plot(param_samples[i, :1000])  # Plot first 1000 samples
        axes[i].set_ylabel(param_names[i])
        axes[i].set_title(f'Parameter Trace: {param_names[i]}')
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Sample Number')
    plt.tight_layout()
    if save_figs:
        plt.savefig(f'{fig_prefix}_traces.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Parameter distributions
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 4*n_params))
    if n_params == 1:
        axes = [axes]
    
    for i in range(n_params):
        axes[i].hist(param_samples[i, :], bins=50, alpha=0.7, density=True)
        axes[i].axvline(results['best_params'][i], color='red', linestyle='--', 
                       label='Best fit')
        axes[i].set_xlabel(param_names[i])
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'Parameter Distribution: {param_names[i]}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_figs:
        plt.savefig(f'{fig_prefix}_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Corner plot (parameter correlations)
    fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
    
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histograms
                ax.hist(param_samples[i, :], bins=30, alpha=0.7, density=True)
                ax.set_ylabel('Density')
            elif i > j:
                # Lower triangle: scatter plots
                idx = np.random.choice(param_samples.shape[1], 1000, replace=False)
                ax.scatter(param_samples[j, idx], param_samples[i, idx], 
                          alpha=0.3, s=1)
                ax.set_xlabel(param_names[j])
                ax.set_ylabel(param_names[i])
            else:
                # Upper triangle: hide
                ax.set_visible(False)
            
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_figs:
        plt.savefig(f'{fig_prefix}_corner.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print parameter statistics
    print("\nParameter Statistics:")
    print("=" * 50)
    for i, name in enumerate(param_names):
        samples = param_samples[i, :]
        mean_val = np.mean(samples)
        std_val = np.std(samples)
        percentiles = np.percentile(samples, [2.5, 16, 50, 84, 97.5])
        
        print(f"{name}:")
        print(f"  Mean ± Std: {mean_val:.4f} ± {std_val:.4f}")
        print(f"  Median: {percentiles[2]:.4f}")
        print(f"  68% CI: [{percentiles[1]:.4f}, {percentiles[3]:.4f}]")
        print(f"  95% CI: [{percentiles[0]:.4f}, {percentiles[4]:.4f}]")
        print()


# Example usage function
def example_analysis():
    """
    Example of how to use the MCMC analysis pipeline.
    """
    # This would be replaced with real data loading
    print("Example MCMC Analysis")
    print("=" * 40)
    print("This is a template for running MCMC analysis.")
    print("To use with real data:")
    print("1. Load your TA data, uncertainties, time points, and powers")
    print("2. Set parameter bounds and specify which are in log space")
    print("3. Run the analysis and plot results")
    print()
    print("Example code:")
    print("""
# Load data
data = np.load('ta_data.npy')  # shape: (wavelengths, times, fluences)
sigma = np.load('ta_sigma.npy')  # same shape as data
t = np.load('time_delays.npy')  # time delays in ps
powers = np.load('pump_powers.npy')  # pump powers in µW

# Set parameter bounds
lb = np.array([-2.5, -6, 1e-5])  # [log(k1), log(k2), XC]
ub = np.array([-1, -2.5, 0.01])
bounds = (lb, ub)
is_log = np.array([True, True, False])  # First two are in log space

# Run analysis
results = run_mcmc_analysis(data, sigma, t, powers, bounds, is_log)

# Plot results
plot_mcmc_results(results, param_names=['log(k₁/ps⁻¹)', 'log(k₂/ps⁻¹)', 'XC (µW⁻¹)'])
""")


if __name__ == "__main__":
    example_analysis()