import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import special


class Model:
    """
    model : 'BV', 'Marcus', or 'MHC'
    k0    : exchange rate constant (s^-1)
    lambda_: for Marcus / MHC this is the *dimensionless* reorganisation energy
             λ̃ = λ / (R T) (e.g. 8.3 in the Bai & Bazant paper)
    """

    def __init__(
        self,
        model=None,
        k01=None,
        k02=None,
        alpha=None,
        lambda_=None,
        n=1,
        T=298.15,
        eta=np.linspace(-0.5, 0.5, 200),
        C=1.0,
        origin_eta = False
    ):
        self.model = model
        self.k01 = k01
        self.k02 = k02
        self.lambda_ = lambda_
        self.alpha = alpha
        self.n = n
        self.T = T
        self.F = 96485.3329  # Faraday constant, C/mol
        self.R = 8.314462618  # Gas constant, J/(mol·K)
        self.e = 1.602176634e-19  # Elementary charge, C
        # exchange current density if you want currents rather than rates
        self.i01 = k01 * n * self.F
        self.i02 = k02 * n * self.F
        self.eta = eta  # overpotential (V)
        self.C = C  # concentration ratio for MHC2 model
        self.origin_eta = origin_eta  # wether to use the original eta range and points or to generate new points with higher fidelity

    # ------------------------------------------------------------------
    # RATE CONSTANT k(η)
    # ------------------------------------------------------------------
    def rate_constant(self):
        """
        Forward rate constant k(η) for the chosen model.

        We always write k(η) = k0 * f(η), so that k(0) = k0.
        """
        if self.origin_eta == True:
            g = self.eta
        else:
            g = np.linspace(np.max(self.eta)+2, np.min(self.eta)-2, len(self.eta)*10)  # interpolate over a finer grid and larger range

        lam_tilde = self.lambda_
        k = np.zeros_like(g)
        positive_mask = g > 0
        negative_mask = g <= 0

        if self.model in ("BV", None):
            # Butler-Volmer rate constant - use different k0 for positive vs negative eta
            
            # For positive eta (oxidation), use k02
            k[positive_mask] = self.k02 * (np.exp(-self.alpha * g[positive_mask]) - np.exp((1.0 - self.alpha) * g[positive_mask]))
            
            # For negative eta (reduction), use k01  
            k[negative_mask] = self.k01 * (np.exp(-self.alpha * g[negative_mask]) - np.exp((1.0 - self.alpha) * g[negative_mask]))

        elif self.model == "Marcus":
            # Marcus rate constant: k ∝ exp(- (λ̃ - η)^2 / 4λ̃) - exp(- (λ̃ + η)^2 / 4λ̃)
            # helper: forward and backward shape functions
            def marcus_fwd(eta):
                return np.exp(-((lam_tilde - eta) ** 2) / (4.0 * lam_tilde))

            def marcus_bwd(eta):
                return np.exp(-((lam_tilde + eta) ** 2) / (4.0 * lam_tilde))

            # positive η: use k02
            gp = g[positive_mask]
            k[positive_mask] = self.k02 * np.exp(lam_tilde/4.0) * (marcus_fwd(gp) - marcus_bwd(gp))

            # negative η: use k01
            gn = g[negative_mask]
            k[negative_mask] = self.k01 * np.exp(lam_tilde/4.0) * (marcus_fwd(gn) - marcus_bwd(gn))
            
        elif self.model == "MHC":
            # MHC net rate constant (symmetric concentrations):
            # k(λ̃, η̃) = -k0(η̃) * tanh(η̃/2) * erfc(arg(η̃)) / erfc(arg(0))

            lam_tilde = float(self.lambda_)        # dimensionless λ̃
            g = np.asarray(g, dtype=float)         # dimensionless overpotential η̃

            sqrt_lam = np.sqrt(lam_tilde)

            # arg(η̃) and arg(0) in the erfc
            inner_sqrt = np.sqrt(1.0 + np.sqrt(lam_tilde) + g**2)
            arg = (lam_tilde - inner_sqrt) / (2.0 * sqrt_lam)

            inner_sqrt_0 = np.sqrt(1.0 + np.sqrt(lam_tilde))  # g = 0
            arg_0 = (lam_tilde - inner_sqrt_0) / (2.0 * sqrt_lam)

            # Dimensionless shape factor S(g) with S(0) = 1
            S = special.erfc(arg) / (special.erfc(arg_0))

            # Piecewise k0: k01 for η̃<0, k02 for η̃>0 (Table 1 values)
            k0_arr = np.where(g >= 0, float(self.k02), float(self.k01))
            f = np.where(g>=0, 1/(1+np.exp(-g)), 1/(1+np.exp(g)))

            # Overall MHC net rate
            k = -k0_arr * -2* np.tanh(g / 2.0) * S

        elif self.model == "MHC2":
            # Alternative MHC model (not used in this work)
            lam_tilde = float(self.lambda_)        # dimensionless λ̃
            g_eff = g+np.log(self.C)
            g_eff = np.asarray(g_eff, dtype=float)         # dimensionless overpotential η̃

            sqrt_lam = np.sqrt(lam_tilde)

            # arg(η̃) and arg(0) in the erfc
            inner_sqrt = np.sqrt(1.0 + np.sqrt(lam_tilde) + g_eff**2)
            arg = (lam_tilde - inner_sqrt) / (2.0 * sqrt_lam)

            inner_sqrt_0 = np.sqrt(1.0 + np.sqrt(lam_tilde))  # g = 0
            arg_0 = (lam_tilde - inner_sqrt_0) / (2.0 * sqrt_lam)

            # Dimensionless shape factor S(g) with S(0) = 1
            S = special.erfc(arg) / (special.erfc(arg_0))

            # Piecewise k0: k01 for η̃<0, k02 for η̃>0 (Table 1 values)
            k0_arr = float(self.k01)

            # Overall MHC net rate
            k = -k0_arr * -2* (self.C/(1+np.exp(g_eff))-1/(1+np.exp(-g_eff))) * S

        else:
            raise ValueError("Unsupported model type")

        return g, k

    # ------------------------------------------------------------------
    # CURRENT (net Faradaic current, if you need it)
    # ------------------------------------------------------------------
    def current_density(self):
        """
        Return current density i(η) = n * F * k(η)
        
        Returns:
        - eta: overpotential array
        - i: current density array (A/m²)
        """
        eta, k = self.rate_constant()
        
        # Convert rate constant to current density: i = n * F * k
        i = self.n * self.F * k
        
        return eta, i

    # ------------------------------------------------------------------
    # ln k helper
    # ------------------------------------------------------------------
    def ln_k(self, normalised=False):
        """
        Return (eta, y) where y is:

          normalised=True  -> ln(k / k0)
          normalised=False -> ln(k)

        This is what you want for Bai & Bazant-style Tafel plots.
        """
        eta, k = self.rate_constant()

        if normalised:
            # piecewise k0 array for correct normalisation
            k0_arr = np.where(eta > 0, self.k02, self.k01)
            x = np.abs(k / k0_arr)
        else:
            x = np.abs(k)

        x = np.clip(x, np.finfo(float).tiny, None)  # avoid log(0)
        y = np.log(x)
        return eta, y

    # ------------------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------------------
    def plot_current(self):
        eta, i = self.current_density()

        plt.figure(figsize=(8, 6))
        plt.plot(eta, i, label=f"{self.model} equation", color="blue")
        plt.axhline(0, color="black", lw=0.5, ls="--")
        plt.axvline(0, color="black", lw=0.5, ls="--")
        plt.title(f"{self.model} current–overpotential")
        plt.xlabel("Overpotential (V)")
        plt.ylabel("Current density (A/m²)")
        plt.yscale("symlog", linthresh=1e-6)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_tafel(self, data=None, data_1=None, data_2=None, normalised=True):
        """
        Plot theoretical ln(k) or ln(k/k0) vs overpotential and optionally
        overlay experimental data from TafelData (which stores ln(k)).
        """
        eta, y = self.ln_k(normalised=normalised)

        plt.figure(figsize=(8, 6))
        plt.plot(eta, y, label=f"{self.model} (theory)", color="red")

        if data is not None:
            # data['lnk'] is already ln(k); normalise if requested
            if normalised:
                # subtract value at closest to eta=0 so it behaves like ln(k/k0)
                idx0 = data["eta"].abs().idxmin()
                lnk0 = data["lnk"].iloc[idx0]
                y_data = data["lnk"] - lnk0
            else:
                y_data = data["lnk"]
            plt.plot(data["eta"], y_data, "o", label="Cell A", color="green")

        if data_1 is not None:
            if normalised:
                idx0 = data_1["eta"].abs().idxmin()
                lnk0 = data_1["lnk"].iloc[idx0]
                y_data = data_1["lnk"] - lnk0
            else:
                y_data = data_1["lnk"]
            plt.plot(data_1["eta"], y_data, "s", label="Cell B", color="orange")

        if data_2 is not None:
            if normalised:
                idx0 = data_2["eta"].abs().idxmin()
                lnk0 = data_2["lnk"].iloc[idx0]
                y_data = data_2["lnk"] - lnk0
            else:
                y_data = data_2["lnk"]
            plt.plot(data_2["eta"], y_data, "^", label="Cell C", color="purple")

        plt.axvline(0, color="black", lw=0.5, ls="--")
        plt.title(f"{self.model} Tafel plot")
        plt.xlabel("Overpotential (V)")
        plt.ylabel("ln(k/k0)" if normalised else "ln(k)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


class TafelData:
    """Class for loading and processing Tafel plot data from CSV files."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def load_and_process(self):
        """
        Load CSV data and process it for comparison with the kinetic models.
        Assumes CSV has columns:
          - 'Overpotential' or 'eta'  (in mV)
          - 'Ln(k)' or 'lnk'          (natural log of rate constant)
        """
        data = pd.read_csv(self.filepath)
        data.columns = data.columns.str.strip()

        if "Overpotential" in data.columns:
            data.rename(columns={"Overpotential": "eta"}, inplace=True)
        if "Ln(k)" in data.columns:
            data.rename(columns={"Ln(k)": "lnk"}, inplace=True)

        if "eta" not in data.columns or "lnk" not in data.columns:
            raise ValueError(
                "CSV must have 'eta'/'Overpotential' and 'lnk'/'Ln(k)' columns. "
                f"Found: {data.columns.tolist()}"
            )

        # store k for convenience if needed
        data["k"] = np.exp(data["lnk"])

        self.data = data
        return data

    def get_summary(self):
        if self.data is None:
            print("No data loaded. Call load_and_process() first.")
            return

        print("Data Summary:")
        print(f"  Number of points: {len(self.data)}")
        print(f"  Eta range: {self.data['eta'].min():.4f} to {self.data['eta'].max():.4f} V")
        print(f"  ln(k) range: {self.data['lnk'].min():.4f} to {self.data['lnk'].max():.4f}")
        print(f"  k range: {self.data['k'].min():.6e} to {self.data['k'].max():.6e}")
        print("\nFirst few rows:")
        print(self.data[["eta", "lnk", "k"]].head())

def error_function(model, data):
    """
    Compute the sum of squared errors between model predictions and experimental data.

    Parameters:
    - model: an instance of Model class
    - data: a pandas DataFrame with 'eta' and 'lnk' columns

    Returns:
    - sse: sum of squared errors
    """
    eta_data = data["eta"].values
    lnk_data = data["lnk"].values

    # Get model predictions at the same eta points
    eta_model, lnk_model = model.ln_k(normalised=False)

    # Interpolate model predictions to match data eta points
    lnk_model_interp = np.interp(eta_data, eta_model, lnk_model)

    # Compute sum of squared errors
    sse = np.sum((lnk_data - lnk_model_interp) ** 2)
    return sse
