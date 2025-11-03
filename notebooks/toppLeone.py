import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous

# Clase personalizada
class topp_leone_gen(rv_continuous):
    def _argcheck(self, nu, sigma):
        return (nu > 0) & (sigma > 0)

    def _pdf(self, x, nu, sigma):
        z = x / sigma
        cond = (x >= 0) & (x <= sigma)
        pdf = np.zeros_like(x)
        pdf[cond] = (2 * nu / sigma) * z[cond]**(nu - 1) * (1 - z[cond]) * (2 - z[cond])**(nu - 1)
        return pdf

    def _cdf(self, x, nu, sigma):
        z = x / sigma
        cdf = np.zeros_like(x)
        cdf[x >= sigma] = 1.0
        cond = (x >= 0) & (x < sigma)
        cdf[cond] = (z[cond] * (2 - z[cond]))**nu
        return cdf

    def _ppf(self, u, nu, sigma):
        z = 1 - np.sqrt(1 - u**(1 / nu))
        return sigma * z

# Instanciar la distribución
topp_leone = topp_leone_gen(name='topp_leone', a=0)

# Parámetros
nu = 2.5
sigma = 1.5

# Rango y muestra
x_vals = np.linspace(0, sigma, 500)
sample = topp_leone.rvs(nu, sigma, size=1000, random_state=42)

# Estilizar el gráfico para publicación
plt.figure(figsize=(7, 5))
plt.rcParams.update({"font.size": 12, "text.usetex": False})

# Graficar PDF y CDF
plt.plot(x_vals, topp_leone.pdf(x_vals, nu, sigma), label=r'PDF $f(x)$', lw=2, color='C0')
plt.plot(x_vals, topp_leone.cdf(x_vals, nu, sigma), label=r'CDF $F(x)$', lw=2, color='C2')

# Histograma de simulaciones
plt.hist(sample, bins=30, density=True, alpha=0.3, color='C2', label='Simulated data')

# Etiquetas
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel('Density / Probability', fontsize=14)
plt.title(r'Topp--Leone Distribution: $f(x;\nu=2.5,\ \sigma=1.5)$', fontsize=15)

# Estética
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, sigma + 0.1)
plt.ylim(0, 1.3)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
plt.tight_layout()
plt.show()

topp_leone = topp_leone_gen(name='topp_leone', a=0)

# Parámetros
nu = 0.5
sigma = 3.0

# Rango y muestra
x_vals = np.linspace(0, sigma, 500)
sample = topp_leone.rvs(nu, sigma, size=1000, random_state=42)

# Estilizar el gráfico para publicación
plt.figure(figsize=(7, 5))
plt.rcParams.update({"font.size": 12, "text.usetex": False})

# Graficar PDF y CDF
plt.plot(x_vals, topp_leone.pdf(x_vals, nu, sigma), label=r'PDF $f(x)$', lw=2, color='C0')
plt.plot(x_vals, topp_leone.cdf(x_vals, nu, sigma), label=r'CDF $F(x)$', lw=2, color='C2')

# Histograma de simulaciones
plt.hist(sample, bins=30, density=True, alpha=0.3, color='C2', label='Simulated data')

# Etiquetas
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel('Density / Probability', fontsize=14)
plt.title(r'Topp--Leone Distribution: $f(x;\nu=0.5,\ \sigma=3)$', fontsize=15)

# Estética
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, sigma + 0.1)
plt.ylim(0, 1.3)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
plt.tight_layout()
plt.show()

topp_leone = topp_leone_gen(name='topp_leone', a=0)

# Parámetros
nu = 0.1
sigma = 1

# Rango y muestra
x_vals = np.linspace(0, sigma, 500)
sample = topp_leone.rvs(nu, sigma, size=1000, random_state=42)

# Estilizar el gráfico para publicación
plt.figure(figsize=(7, 5))
plt.rcParams.update({"font.size": 12, "text.usetex": False})

# Graficar PDF y CDF
plt.plot(x_vals, topp_leone.pdf(x_vals, nu, sigma), label=r'PDF $f(x)$', lw=2, color='C0')
plt.plot(x_vals, topp_leone.cdf(x_vals, nu, sigma), label=r'CDF $F(x)$', lw=2, color='C2')

# Histograma de simulaciones
plt.hist(sample, bins=30, density=True, alpha=0.3, color='C2', label='Simulated data')

# Etiquetas
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel('Density / Probability', fontsize=14)
plt.title(r'Topp--Leone Distribution: $f(x;\nu=0.5,\ \sigma=1)$', fontsize=15)

# Estética
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, sigma + 0.1)
plt.ylim(0, 1.3)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
plt.tight_layout()
plt.show()

import numpy as np
from scipy.optimize import minimize

def log_likelihood_tl(params, data):
    nu, sigma = params
    if nu <= 0 or sigma <= 0:
        return np.inf  # penalización por parámetros inválidos
    z = data / sigma
    if np.any(z >= 1):
        return np.inf  # fuera del dominio válido
    term1 = len(data) * np.log(2 * nu / sigma)
    term2 = (nu - 1) * np.sum(np.log(z))
    term3 = np.sum(np.log(1 - z))
    term4 = (nu - 1) * np.sum(np.log(2 - z))
    loglike = term1 + term2 + term3 + term4
    return loglike  # se minimiza la negativa


def dloglik_nu(x, nu, sigma):
    """
    Compute the derivative of the log-likelihood of the Topp-Leone distribution with respect to nu.

    Parameters
    ----------
    x : array_like
        Sample data (must be in the range (0, sigma)).
    nu : float
        Shape parameter (nu > 0).
    sigma : float
        Scale parameter (sigma > 0).

    Returns
    -------
    float
        Derivative of the log-likelihood with respect to nu.
    """
    x = np.asarray(x)


    z = x / sigma


    term1 = len(x) / nu
    term2 = np.sum(np.log(z))
    term3 = np.sum(np.log(2 - z))

    return term1 + term2 + term3

import numpy as np

def dloglik_sigma(x, nu, sigma):
    """
    Compute the derivative of the log-likelihood of the Topp-Leone distribution with respect to sigma.

    Parameters
    ----------
    x : array_like
        Sample data (must be in (0, sigma)).
    nu : float
        Shape parameter (nu > 0).
    sigma : float
        Scale parameter (sigma > 0).

    Returns
    -------
    float
        Derivative of the log-likelihood with respect to sigma.
    """
    x = np.asarray(x)


    z = x / sigma


    n = len(x)

    term1 = -n * nu / sigma
    term2 = np.sum(x / (np.power(sigma,2) * (1 - z)))
    term3 = (nu - 1) * np.sum(x / ((np.power(sigma,2) * (2 - z))))

    return term1 + term2 + term3

from scipy.optimize import root

def score_equations(params, x):
    nu, sigma = params
    if nu <= 0 or sigma <= 0:
        return [np.inf, np.inf]  # penalizar fuera del dominio
    grad_nu = dloglik_nu(x, nu, sigma)
    grad_sigma = dloglik_sigma(x, nu, sigma)
    return [grad_nu, grad_sigma]

# Función para resolver el sistema y obtener los MLE
def solve_mle_topp_leone(x, tru_sigma, nu_init=1.0, sigma_init=1.0):
    x = np.asarray(x)

    true_sigma_of_sample = tru_sigma # This is a potential limitation if sigma is unknown
    filtered_x = x[(x > 0) & (x < true_sigma_of_sample)]

    if len(filtered_x) != len(x):
        print(f"Warning: Filtered out {len(x) - len(filtered_x)} data points at boundaries.")
        if len(filtered_x) == 0:
            raise ValueError("No data points remain after filtering.")
    try:
        result = root(
            score_equations,
            [nu_init, sigma_init],
            args=(filtered_x,),
            method='hybr',
            options={'maxfev': 5000}
        )
        if result.success:
            return result.x  # [nu_hat, sigma_hat]
        else:
            return [np.nan, np.nan]
    except Exception as e:
        return [np.nan, np.nan]

nu = 0.5
sigma = 2

# Tamaño de la muestra
n = 20

# Generar muestra aleatoria usando la transformada inversa
u = np.random.uniform(0, 1, size=n)
sample = topp_leone.ppf(u, nu, sigma)
solve_mle_topp_leone(sample,tru_sigma=sigma, nu_init=0.5, sigma_init=2)

nu = 0.5
sigma = 2

n = 20

def simulate_mle_statistics(nu_true, sigma_true, n, reps=1000, seed=None, nu_init=0.5, sigma_init=2.0):
    np.random.seed(seed)
    estimates = []

    for i in range(reps):
        np.random.seed(i)
        u = np.random.uniform(0, 1, size=n)
        sample = topp_leone.ppf(u, nu_true, sigma_true)
        mle = solve_mle_topp_leone(sample,tru_sigma=sigma_true, nu_init=nu_init, sigma_init=sigma_init)
        estimates.append(mle)

    estimates = np.array(estimates)
    nu_hat = estimates[:, 0]
    sigma_hat = estimates[:, 1]

    # Filtrar NaNs
    valid = ~np.isnan(nu_hat) & ~np.isnan(sigma_hat)
    nu_hat = nu_hat[valid]
    sigma_hat = sigma_hat[valid]

    stats = {
        "mean_nu": np.mean(nu_hat),
        "mean_sigma": np.mean(sigma_hat),
        "bias_nu": np.mean(nu_hat - nu_true),
        "bias_sigma": np.mean(sigma_hat - sigma_true),
        "mse_nu": np.mean((nu_hat - nu_true)**2),
        "mse_sigma": np.mean((sigma_hat - sigma_true)**2),
        "ci_nu": (np.percentile(nu_hat, 2.5), np.percentile(nu_hat, 97.5)),
        "ci_sigma": (np.percentile(sigma_hat, 2.5), np.percentile(sigma_hat, 97.5)),
        "reps_used": len(nu_hat)
    }
    return stats

nu = 0.5
sigma = 2
n = 20
result = simulate_mle_statistics(nu_true=nu, sigma_true = sigma, n=n, reps=1000, nu_init=0.5, sigma_init=1.7)
result

# prompt: con la función simulate_mle_statistics creame una tabla donde se evalúe el resultado para myestras de tamaño, 20, 50, 200, 500 y 1000

import pandas as pd

sample_sizes = [20, 50, 200, 500, 1000]
results_list = []

nu_true = 0.5
sigma_true = 2.0

for n in sample_sizes:
    print(f"Simulating for sample size n = {n}")
    stats = simulate_mle_statistics(
        nu_true=nu_true,
        sigma_true=sigma_true,
        n=n,
        reps=1000,
        nu_init=0.5,
        sigma_init=1.7  # Using a reasonable initial guess for sigma
    )
    stats['sample_size'] = n
    results_list.append(stats)

# Convert results to a pandas DataFrame for a nice table format
results_df = pd.DataFrame(results_list)

# Reorder columns for better readability
results_df = results_df[[
    'sample_size',
    'mean_nu', 'bias_nu', 'mse_nu', 'ci_nu',
    'mean_sigma', 'bias_sigma', 'mse_sigma', 'ci_sigma',
    'reps_used'
]]

# Format confidence intervals for display
results_df['ci_nu'] = results_df['ci_nu'].apply(lambda x: f"({x[0]:.4f}, {x[1]:.4f})")
results_df['ci_sigma'] = results_df['ci_sigma'].apply(lambda x: f"({x[0]:.4f}, {x[1]:.4f})")

# Display the table
print(results_df.to_markdown(index=False, floatfmt=".4f"))

# prompt: exporta results_df a latex code

print(results_df.to_latex(index=False))

# Estimación de MLE por optimización numérica
def estimate_mle_tl(data, nu_init=1.0, sigma_init=1.0):
    initial_guess = [nu_init, sigma_init]
    bounds = [(1e-6, None), (1e-6, None)]
    result = minimize(log_likelihood_tl, initial_guess, args=(data,), bounds=bounds)
    if result.success:
        return result.x
    else:
        raise RuntimeError("Optimization failed: " + result.message)

# Ejemplo de uso
if __name__ == "__main__":
    np.random.seed(42)
    # Muestra de ejemplo dentro del soporte (0, sigma)
    sample = np.random.uniform(0.01, 0.99, size=100) * 2.0

    nu_hat, sigma_hat = estimate_mle_tl(sample)
    print(f"Estimated nu: {nu_hat:.4f}, Estimated sigma: {sigma_hat:.4f}")

def observed_info_matrix(x, nu, sigma):
    n = len(x)
    z = x / sigma

    # Second derivatives
    d2_nu2 = -n / nu**2

    # d2_sigma2
    term1 = n * nu / sigma**2
    term2 = np.sum(x * (2 * sigma - x) / (sigma**2 - x * sigma)**2)
    term3 = (nu - 1) * np.sum(x * (2 * sigma - 2 * x) / (sigma**2 - 2 * x * sigma)**2)
    d2_sigma2 = term1 - term2 - term3

    # d2_nu_sigma
    cross_terms = -np.sum(1 / sigma - x / (sigma**2 * (2 - z)))

    # Fisher information matrix (negative Hessian)
    I_obs = -np.array([
        [d2_nu2,        cross_terms],
        [cross_terms,   d2_sigma2]
    ])

    return I_obs

def compute_standard_errors(x, nu_hat, sigma_hat):
    I_obs = observed_info_matrix(x, nu_hat, sigma_hat)
    cov_matrix = np.linalg.inv(I_obs)
    se_nu, se_sigma = np.sqrt(np.diag(cov_matrix))
    return se_nu, se_sigma

data1 =[5.5, 5, 4.9, 6.4, 5.1, 5.2, 5.2, 5, 4.7, 4, 4.5, 4.2, 4.1, 4.56, 5.01, 4.7, 3.13, 3.12, 2.68, 2.77, 2.7, 2.36, 4.38, 5.73, 4.35, 6.81, 1.91, 2.66, 2.61, 1.68, 2.04, 2.08, 2.13, 3.8, 3.73, 3.71, 3.28, 3.9, 4, 3.8, 4.1, 3.9, 4.05, 4, 3.95, 4, 4.5, 4.5, 4.2, 4.55, 4.65, 4.1, 4.25, 4.3, 4.5, 4.7, 5.15, 4.3, 4.5, 4.9, 5, 5.35, 5.15, 5.25, 5.8, 5.85, 5.9, 5.75, 6.25, 6.05, 5.9, 3.6, 4.1, 4.5, 5.3, 4.85, 5.3, 5.45, 5.1, 5.3, 5.2, 5.3, 5.25, 4.75, 4.5, 4.2, 4, 4.15, 4.25, 4.3, 3.75, 3.95, 3.51, 4.13, 5.4, 5, 2.1, 4.6, 3.2, 2.5, 4.1, 3.5, 3.2, 3.3, 4.6, 4.3, 4.3, 4.5, 5.5, 4.6, 4.9, 4.3, 3, 3.4, 3.7, 4.4, 4.9, 4.9, 5]
sigma_true = 10.0

result = solve_mle_topp_leone(data1,tru_sigma=sigma_true, nu_init=2.0, sigma_init=2.5)
result

# prompt: haz el histograma de data1

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.hist(data1, bins=10, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Valor')
plt.ylabel('Densidad')
plt.title('Histograma de data1')
plt.grid(axis='y', alpha=0.75)
plt.show()