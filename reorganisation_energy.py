import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import special

def reorganisation_energy(dimension=False, epsilon_op=4.74, epsilon_s=11.58, a_0=0.21e-9, d=0.21e-9, T=293):
    """
    Calculate the reorganisation energy using the Born energy of solvation formula:
    
    λ = e²/(8π ε₀ kᵦ T) (1/a₀ - 1/2d)(1/ε_op - 1/ε_s)
    
    This implementation reproduces the values from the paper:
    - Direct contact (d = a₀): λ = 8.3 (dimensionless)
    - Separated by PO₄ (d = 0.44 nm): λ = 12.64 (dimensionless)
    
    Parameters:
    -----------
    dimension : bool
        If True, return dimensional value in Joules. If False, return dimensionless λ
    epsilon_op : float
        Optical dielectric constant (default 4.74 from first-principles calculations)
    epsilon_s : float  
        Static dielectric constant (default 11.58 from first-principles calculations)
    a_0 : float
        Effective radius of reactant in meters (default 0.21 nm for Fe-O bond in FeO₆)
    d : float
        Distance from reactant center to electrode surface in meters 
        (default 0.21 nm for direct contact)
    T : float
        Temperature in Kelvin (default 300 K)
        
    Returns:
    --------
    lambda_reorg : float
        Reorganization energy (dimensionless λ if dimension=False, Joules if dimension=True)
        
    Examples:
    ---------
    >>> # Direct contact case
    >>> lambda_direct = reorganisation_energy()
    >>> print(f"λ = {lambda_direct:.2f}")  # Should give ~8.26
    
    >>> # Separated by PO₄ case  
    >>> lambda_separated = reorganisation_energy(d=0.44e-9)
    >>> print(f"λ = {lambda_separated:.2f}")  # Should give ~12.58
    """
    # Physical constants
    e = 1.602176634e-19      # Elementary charge (C)
    epsilon_0 = 8.854187817e-12  # Permittivity of free space (F/m)
    k_b = 1.380649e-23       # Boltzmann constant (J/K)
    
    # Born energy of solvation formula from the paper:
    # λ = e²/(8π ε₀ kᵦ T) (1/a₀ - 1/2d)(1/ε_op - 1/ε_s)
    
    # Energy scale factor (has units of meters, making result dimensionless)
    energy_scale = e**2 / (8 * np.pi * epsilon_0 * k_b * T)
    
    # Geometric factor (1/meters)
    geometric_factor = (1/a_0 - 1/(2*d))
    
    # Dielectric factor (dimensionless)
    dielectric_factor = (1/epsilon_op - 1/epsilon_s)
    
    # Dimensionless reorganization energy
    lambda_dimensionless = energy_scale * geometric_factor * dielectric_factor
    
    if dimension:
        # Convert to dimensional form (Joules)
        lambda_dimensional = lambda_dimensionless * k_b * T
        return lambda_dimensional
    else:
        return lambda_dimensionless


def demonstrate_paper_calculations():
    """
    Demonstrate the calculations from the paper showing λ = 8.3 and λ = 12.64
    """
    print("Born Energy of Solvation - Reorganization Energy Calculations")
    print("=" * 60)
    
    # Paper values
    epsilon_op = 4.74  # Optical dielectric constant
    epsilon_s = 11.58  # Static dielectric constant
    a_0 = 0.21e-9     # Fe-O bond length (m)
    T = 300           # Room temperature (K)
    
    print(f"Parameters from paper:")
    print(f"  ε_op = {epsilon_op}")
    print(f"  ε_s = {epsilon_s}")
    print(f"  a_0 = {a_0*1e9:.2f} nm (Fe-O bond length)")
    print(f"  T = {T} K")
    
    # Case 1: Direct contact (d = a_0)
    print(f"\n1. Direct contact case (d = a_0 = {a_0*1e9:.2f} nm):")
    d1 = a_0
    lambda_1_dimensional = reorganisation_energy(dimension=True, epsilon_op=epsilon_op, 
                                                epsilon_s=epsilon_s, a_0=a_0, d=d1, T=T)
    lambda_1_dimensionless = reorganisation_energy(dimension=False, epsilon_op=epsilon_op, 
                                                  epsilon_s=epsilon_s, a_0=a_0, d=d1, T=T)
    
    print(f"  λ = {lambda_1_dimensional/1.602176634e-19*1000:.0f} meV")
    print(f"  λ (dimensionless) = {lambda_1_dimensionless:.2f}")
    
    # Case 2: Separated by PO4 tetrahedrons (d = 0.44 nm)
    print(f"\n2. Separated by PO4 case (d = 0.44 nm):")
    d2 = 0.44e-9
    lambda_2_dimensional = reorganisation_energy(dimension=True, epsilon_op=epsilon_op, 
                                                epsilon_s=epsilon_s, a_0=a_0, d=d2, T=T)
    lambda_2_dimensionless = reorganisation_energy(dimension=False, epsilon_op=epsilon_op, 
                                                  epsilon_s=epsilon_s, a_0=a_0, d=d2, T=T)
    
    print(f"  d = {d2*1e9:.2f} nm (a_0 + O-O bond ~0.23 nm)")
    print(f"  λ = {lambda_2_dimensional/1.602176634e-19*1000:.0f} meV")
    print(f"  λ (dimensionless) = {lambda_2_dimensionless:.2f}")
    
    print(f"\nComparison with paper values:")
    print(f"  Paper: λ = 8.3 (direct), λ = 12.64 (separated)")
    print(f"  Calculated: λ = {lambda_1_dimensionless:.2f} (direct), λ = {lambda_2_dimensionless:.2f} (separated)")


if __name__ == "__main__":
    # Run demonstration
    demonstrate_paper_calculations()
    
    # Additional examples
    print(f"\n" + "="*60)
    print("Additional Examples:")
    print("="*60)
    
    # Test with different parameters
    lambda_test = reorganisation_energy(dimension=False, epsilon_op=4.74, epsilon_s=11.58, 
                                       a_0=0.21e-9, d=0.21e-9, T=300)
    print(f"λ (dimensionless, direct contact): {lambda_test:.2f}")
    
    # Test dimensional output
    lambda_dimensional = reorganisation_energy(dimension=True, epsilon_op=4.74, epsilon_s=11.58, 
                                              a_0=0.21e-9, d=0.21e-9, T=300)
    print(f"λ (dimensional): {lambda_dimensional:.2e} J")
    print(f"λ in meV: {lambda_dimensional/1.602176634e-19*1000:.1f} meV")

    d = np.linspace(0.2e-9, 1.0e-9, 100)
    lambda_values = [reorganisation_energy(dimension=False, epsilon_op=4.74, epsilon_s=11.58, a_0=0.21e-9, d=di, T=300) for di in d] 

    plt.plot(d*1e9, lambda_values)
    plt.hlines(8.3, xmin=0.2, xmax=1.0, colors='r', linestyles='dashed', label='λ = 8.3 (direct contact)')
    plt.hlines(12.64, xmin=0.2, xmax=1.0, colors='g', linestyles='dashed', label='λ = 12.64 (separated by PO₄)')
    plt.xlabel("Distance from reactant center to electrode surface (nm)")
    plt.ylabel("Dimensionless Reorganization Energy (λ)")
    plt.title("Reorganization Energy vs. Distance")
    plt.grid(True)
    plt.show()