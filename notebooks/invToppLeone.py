import numpy as np

def pdf_inverted_topp_leone(y, v, xi):
    """
    Calcula la densidad de probabilidad de la distribución Inversa Topp-Leone ITL(v, xi)
    para uno o varios valores de 'y'.

    Parámetros
    ----------
    y  : float o array-like de floats
         Valores donde se evaluará la densidad. Deben ser y>0 (fuera de eso, la densidad es 0).
    v  : float
         Parámetro de forma (v>0).
    xi : float
         Parámetro de escala (xi>0).

    Retorna
    -------
    pdf : float o np.ndarray
          Valor(es) de la densidad de ITL en cada punto de 'y'.

    Ejemplo de uso
    --------------
    >>> import numpy as np
    >>> y_vals = np.array([0.5, 1.0, 2.0])
    >>> pdf_vals = pdf_inverted_topp_leone(y_vals, v=2.0, xi=1.0)
    >>> pdf_vals
    array([0.24..., 0.14..., 0.05...])
    """
    # Convertir 'y' a arreglo NumPy
    y = np.asarray(y, dtype=float)

    # Crear arreglo resultante (de la misma forma que y), inicializado en 0
    pdf = np.zeros_like(y)

    # Máscara de valores válidos (soporte)
    mask = (y > 0)

    # Cálculo dentro del soporte
    ratio = y[mask] / xi
    pdf[mask] = (
        2 * v
        * np.power(ratio,v - 1)
        * np.power(1 + ratio, -2*v - 1)
        * np.power(2 + ratio,v - 1)
    )
    return pdf

"""# **Funcion de Distribucion Acumulada de la inversa topp leone**"""

import numpy as np

def cdf_inverted_topp_leone(y, v, xi):
    """
    CDF de la distribución Inversa Topp-Leone ITL(v, xi).

    Parámetros
    ----------
    y  : float o array-like de floats
         Valores en los que se evaluará la CDF. Deben ser y >= 0 (para y<0, la CDF es 0).
    v  : float
         Parámetro de forma (v > 0).
    xi : float
         Parámetro de escala (xi > 0).

    Retorna
    -------
    F : float o np.ndarray
        Valor(es) de la CDF en cada punto de 'y'.

    Ejemplo
    -------
    >>> x_vals = np.array([0.0, 0.5, 1.0, 2.0])
    >>> F_vals = cdf_inverted_topp_leone(x_vals, v=2.0, xi=1.0)
    >>> F_vals
    array([0.        , 0.75..., 0.88..., 0.97...])
    """
    # Convertimos 'y' a arreglo NumPy
    y = np.asarray(y, dtype=float)

    # CDF inicializada en cero
    F = np.zeros_like(y)

    # Máscara para valores y >= 0
    mask = (y >= 0)

    # Cálculo de beta = xi / (xi + y) en el soporte
    beta = xi / (xi + y[mask])

    # Se aplica F = (1 - beta^2)^v para y >= 0
    F[mask] = np.power(np.power(1 - beta, 2), v)

    return F

"""# **Funcion Quantile de la inversa topp leone**"""

import numpy as np

def q_inverted_topp_leone(p, nu, xi):
    """
    Calcula la función de cuantiles (quantile function, QF) de la
    distribución Inversa Topp-Leone ITL(v, xi).  ξ[(1-p^(1/ν))^(-1/2) - 1]

    Parámetros
    ----------
    p  : float o array-like de floats
         Probabilidades en el intervalo (0, 1).
    v  : float
         Parámetro de forma (v > 0).
    xi : float
         Parámetro de escala (xi > 0).

    Retorna
    -------
    Q : float o np.ndarray
        Valor(es) del cuantil correspondiente a p.

    Ejemplo
    -------
    >>> import numpy as np
    >>> p_vals = np.array([0.1, 0.5, 0.9])
    >>> q_vals = q_inverted_topp_leone(p_vals, v=2.0, xi=1.0)
    >>> q_vals
    array([0.05..., 0.30..., 1.54...])
    """
    # Convertir p a array NumPy
    p = np.asarray(p, dtype=float)

    # Validar 0 < p < 1
    # (Si deseas forzar un manejo especial de p=0 o p=1, se podría agregar)
    if np.any(p <= 0) or np.any(p >= 1):
        raise ValueError("Los valores de p deben estar en el intervalo (0,1).")
    # Calcular el cuantil con la fórmula Q(p) = xi * ( (1-p^1/ν)^(-1/2) - 1 )
    Q = xi * (np.power(1 - np.power(p, 1/nu), -0.5) - 1)
    return Q

"""# **Estimador de NU**"""

def estimador_nu(y, nu, xi):
    """
    Función score: ∂ℓ/∂ξ para la distribución Inversa Topp–Leone.

    Parámetros
    ----------
    xi : float
        Valor candidato para ξ (escala), debe ser > 0.
    nu : float
        Parámetro de forma ν (asumido conocido o previo).
    y : array_like
        Vector de datos de la muestra (todos y_i > 0).

    Retorna
    -------
    float
        Evaluación de la derivada de la log-verosimilitud respecto a ξ.
    """
    n = y.size
    y_xi = y / xi

    term1 =  n  / nu
    term2 = -n * np.log(xi)
    term3 = np.sum(np.log(y))
    term4 = -2 * np.sum(np.log(1 + y/xi))
    term5 = np.sum(np.log(2 + y/xi))

    return term1 + term2 + term3 + term4 + term5

"""# **Estimador de XI**"""

import numpy as np
from scipy.optimize import root_scalar, minimize, root
def estimador_xi(xi, y, nu):
    """
    Derivada de la log-verosimilitud con respecto a xi para la distribución
    Inversa Topp-Leone, asumiendo que el parámetro de forma es nu.

    La expresión utilizada es:

       d/dxi = - n *nu/xi + ( (2 nu + 1)/xi ) * sum( (y/xi)/(1+y/xi) ) - ((nu-1)/xi) * sum( (y/xi)/(2+y/xi) )

    Parámetros:
    -----------
    xi : float
         Valor del parámetro de escala.
    y  : array-like
         Datos (observaciones) con y > 0.
    nu : float
         Valor del parámetro de forma (calculado previamente).

    Retorna:
    --------
    derivada : float
         Valor de la derivada de la log-verosimilitud respecto a xi.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    term1 = - (n*nu) / xi
    term2 = ((2 * nu + 1) / xi) * np.sum( (y/xi) / (1 + y/xi) )
    term3 = - ((nu - 1) / xi) * np.sum( (y/xi) / (2 + y/xi) )
    return term1 + term2 + term3


def gradient(params, y):
    """
    Gradiente de la log-verosimilitud para la Inversa Topp–Leone,
    usando las funciones estimador_nu y estimador_xi.

    Parámetros
    ----------
    params : array-like, shape (2,)
        [nu, xi]
    y : array-like
        Muestra de datos (y_i > 0).

    Retorna
    -------
    np.ndarray, shape (2,)
        [∂ℓ/∂ν, ∂ℓ/∂ξ]
    """
    nu, xi = params
    d_nu = estimador_nu(y, nu, xi)
    d_xi = estimador_xi(xi, y, nu)
    return np.array([d_nu, d_xi])

import numpy as np

def log_likelihood(nu, xi, y):
    """
    Log-verosimilitud de la distribución Inversa Topp–Leone para datos y,
    según la expresión:

      l(ν,ξ) = n [ln2 + lnν − ν lnξ]
             + (ν−1) ∑ ln yi
             − (2ν+1) ∑ ln(1 + yi/ξ)
             + (ν−1) ∑ ln(2 + yi/ξ)

    Parámetros
    ----------
    nu : float
        Parámetro de forma ν (> 0).
    xi : float
        Parámetro de escala ξ (> 0).
    y : array_like
        Datos de la muestra (todos yi > 0).

    Retorna
    -------
    float
        Valor de la log-verosimilitud l(ν,ξ).
    """
    y = np.asarray(y, dtype=float)
    n = y.size

    term1 = n * (np.log(2) + np.log(nu) - nu * np.log(xi))
    term2 = (nu - 1) * np.sum(np.log(y))
    term3 = -(2 * nu + 1) * np.sum(np.log(1 + y/xi))
    term4 = (nu - 1) * np.sum(np.log(2 + y/xi))

    return term1 + term2 + term3 + term4



def estimate_itl_params(y, nu_init=1.0, xi_init=1.0,
                             tol=1e-8, maxfev=10000):
    """
    Estima (nu, xi) de la ITL(v, xi) resolviendo ∂ℓ/∂ν = 0 y ∂ℓ/∂ξ = 0
    con scipy.optimize.root (método hybr).

    Parámetros
    ----------
    y : array-like, y > 0
        Muestra de datos.
    nu_init : float
        Valor inicial para nu (> 0).
    xi_init : float
        Valor inicial para xi (> 0).
    tol : float
        Tolerancia de convergencia en root.
    maxfev : int
        Máximo de evaluaciones de función.

    Retorna
    -------
    (nu_hat, xi_hat) : tuple de floats
    """
    y = np.asarray(y, dtype=float)

    # Función vectorial: [∂ℓ/∂ν, ∂ℓ/∂ξ]
    def fun_grad(params):
        return gradient(params, y)

    # Punto inicial
    x0 = np.array([nu_init, xi_init])

    sol = root(fun_grad,
               x0,
               method='hybr',
               tol=tol,
               options={'maxfev': maxfev})

    if not sol.success:
        raise RuntimeError(f"Root-finding no convergió: {sol.message}")

    nu_hat, xi_hat = sol.x
    return nu_hat, xi_hat

import numpy as np
import matplotlib.pyplot as plt



def simulate_itl_inverse_transform(n, nu, xi, seed=None):
    """
    Simula n muestras de la distribución Inversa Topp-Leone (ITL) usando el método
    de la transformación inversa.

    Parámetros:
      n       : int
                Número de muestras a simular.
      nu      : float
                Parámetro de forma de la ITL (v > 0).
      xi      : float
                Parámetro de escala de la ITL (xi > 0).
      seed    : int o None, opcional
                Semilla para reproducibilidad.

    Retorna:
      muestras : np.ndarray
                Array de muestras simuladas de ITL.

    Método:
      1. Generar n números U uniformemente distribuidos en el intervalo (0,1).
      2. Aplicar la función cuantil q_inverted_topp_leone a estos números.
         Esto convierte las muestras uniformes en muestras con la distribución ITL.
    """
    # Para reproducibilidad de los resultados
    if seed is not None:
        np.random.seed(seed)

    # Paso 1: Generar n valores uniformes U ~ Uniform(0,1)
    U = np.random.uniform(0.0, 1.0, n)

    # Paso 2: Aplicar la función cuantil para obtener las muestras de ITL
    muestras = q_inverted_topp_leone(U, nu, xi)

    return muestras




samples = simulate_itl_inverse_transform(100, 2, 1.1, seed=None)

def iterations_estimate(iter, sample_size, nu_init, xi_init, nu_real, xi_real):
    list_nu_hat = []
    list_xi_hat = []

    for i in range(iter):
        samples = simulate_itl_inverse_transform(sample_size, nu_real, xi_real, seed=i)
        mles = estimate_itl_params(samples, nu_init=nu_init, xi_init=xi_init)
        list_nu_hat.append(mles[0])
        list_xi_hat.append(mles[1])

    list_nu_hat = np.array(list_nu_hat)
    list_xi_hat = np.array(list_xi_hat)

    # Estimaciones promedio
    nu_hat = np.mean(list_nu_hat)
    xi_hat = np.mean(list_xi_hat)

    # Cálculo del sesgo
    bias_nu = nu_hat - nu_real
    bias_xi = xi_hat - xi_real

    # Error cuadrático medio
    mse_nu = np.mean(np.power(list_nu_hat - nu_real, 2))
    mse_xi = np.mean(np.power(list_xi_hat - xi_real, 2))

    # Varianza empírica
    var_nu = np.var(list_nu_hat, ddof=1)  # muestral
    var_xi = np.var(list_xi_hat, ddof=1)

    # Intervalos de confianza empíricos al 95%
    ci_nu = np.percentile(list_nu_hat, [2.5, 97.5])
    ci_xi = np.percentile(list_xi_hat, [2.5, 97.5])

    return {
        "nu_hat": nu_hat,
        "xi_hat": xi_hat,
        "bias_nu": bias_nu,
        "bias_xi": bias_xi,
        "mse_nu": mse_nu,
        "mse_xi": mse_xi,
        "var_nu": var_nu,
        "var_xi": var_xi,
        "ci_nu_95": ci_nu,
        "ci_xi_95": ci_xi
    }

nu_hat, xi_hat = iterations_estimate(10000,sample_size = 1000, nu_init=1.0, xi_init=1,nu_real=2.0,xi_real=1.0)
f'nu = {str(nu_hat)}' + f'xi = {str(xi_hat)}'

# prompt: quiero un codigo en python que me aplique esta función para sample_size de tamaño 20,50,200,500, 1000 y me retorne una tabla con el resultado de nu_hat, xi_hat, los bias, mse e intervalos de confianza

import pandas as pd
sample_sizes = [100,200, 500, 1000]
true_nu = 2.0
true_xi = 1.0
xi_init = 1.0
nu_init = 0.7
num_iterations = 10000

results = []

for size in sample_sizes:
    print(f"Running simulations for sample size: {size}")
    stats = iterations_estimate(num_iterations, size, nu_init=nu_init, xi_init=xi_init, nu_real=true_nu, xi_real=true_xi)
    results.append({
        'Sample Size': size,
        'nu_hat': stats['nu_hat'],
        'xi_hat': stats['xi_hat'],
        'Bias_nu': stats['bias_nu'],
        'Bias_xi': stats['bias_xi'],
        'MSE_nu': stats['mse_nu'],
        'MSE_xi': stats['mse_xi'],
        'CI_nu_lower': stats['ci_nu_95'][0],
        'CI_nu_upper': stats['ci_nu_95'][1],
        'CI_xi_lower': stats['ci_xi_95'][0],
        'CI_xi_upper': stats['ci_xi_95'][1]
    })

results_df = pd.DataFrame(results)
print("\nSimulation Results:")
results_df

# prompt: export results_df like latex table

print(results_df.to_latex(index=False))
# You can also save it to a file:
# with open('results_table.tex', 'w') as f:
#    f.write(results_df.to_latex(index=False))

n_muestras = 100    # Número de muestras a simular
nu_param = 1.5       # Ejemplo: parámetro de forma
xi_param = 0.5    # Ejemplo: parámetro de escala

# Simular la muestra
muestras = simulate_itl_inverse_transform(n_muestras, nu_param, xi_param)
estimate_itl_params(y=muestras)

valores_flotantes = [
    1.01, 1.11, 1.13, 1.15, 1.16, 1.17, 1.17, 1.20, 1.52, 1.54, 1.54, 1.57, 1.64,
    1.73, 1.79, 2.09, 2.09, 2.57, 2.75, 2.93, 3.19, 3.54, 3.57, 5.11, 5.62
]

estimate_itl_params(valores_flotantes,0.5)

"""# **Simulacion usando metodo de la transformada invertida**"""

import numpy as np
import matplotlib.pyplot as plt



# =====================================================
# Ejemplo de uso del método de transformación inversa
# =====================================================
if __name__ == '__main__':
    # Parámetros de la distribución ITL
    n = 50     # Número de muestras a simular
    nu = 2.0     # Parámetro de forma
    xi = 1.5     # Parámetro de escala
    seed = 42    # Semilla para reproducibilidad

    # Simular las muestras
    muestras = simulate_itl_inverse_transform(n, nu, xi, seed)

    # Visualización: Histograma de las muestras simuladas
    plt.figure(figsize=(8, 5))
    plt.hist(muestras, bins=30, density=True, alpha=0.7, edgecolor='k')
    plt.xlabel('Valor')
    plt.ylabel('Densidad')
    plt.title('Histograma de muestras simuladas de ITL (v = {}, xi = {})'.format(nu, xi))
    plt.show()

    # Mostrar las primeras 10 muestras simuladas
    print("Primeras 10 muestras simuladas:")
    print(muestras[:10])

estimate_itl_params(muestras,1,max_iter=1000)

import pandas as pd
def simulate_estimation(n, true_nu, true_xi, xi_init= 1.0, nu_init = 1.0, reps=1000):
    """
    Realiza reps (por defecto 1000) simulaciones, cada una con muestra de tamaño n,
    usando los parámetros verdaderos (true_nu, true_xi) para generar la muestra ITL.
    Estima los parámetros a partir de cada muestra y retorna los resultados.

    Parámetros
    ----------
    n : int
        Tamaño de cada muestra.
    true_nu : float
        Valor verdadero de nu.
    true_xi : float
        Valor verdadero de xi.
    reps : int, opcional
        Número de simulaciones (por defecto 1000).
    seed : int o None, opcional
        Semilla para reproducibilidad.

    Retorna
    -------
    nu_estimates : np.ndarray, tamaño (reps,)
        Estimaciones de nu en cada réplica.
    xi_estimates : np.ndarray, tamaño (reps,)
        Estimaciones de xi en cada réplica.
    """
    rng = np.random.default_rng(seed)
    nu_estimates = []
    xi_estimates = []
    for i in range(reps):
        # Usamos una semilla interna distinta en cada simulación
        sample = simulate_itl_inverse_transform(n, true_nu, true_xi, seed=i)
        nu_hat, xi_hat = estimate_itl_params(sample, xi_init=xi_init, nu_init=nu_init)
        nu_estimates.append(nu_hat)
        xi_estimates.append(xi_hat)
    return np.array(nu_estimates), np.array(xi_estimates)

# ======================================================
# Función para generar una tabla resumen con IC y ECM
# ======================================================
def generate_summary_table(n, true_nu, true_xi, reps=1000, nu_init = 1.0, xi_init=1.0):
    """
    Realiza reps simulaciones con muestras de tamaño n para la distribución ITL
    (con parámetros verdaderos true_nu, true_xi) y estima nu y xi en cada réplica.
    Calcula para cada parámetro: el promedio, el intervalo de confianza del 95%, y el
    error cuadrático medio (ECM). Retorna estos resultados en un DataFrame.

    Parámetros
    ----------
    n : int
        Tamaño de cada muestra.
    true_nu : float
        Valor verdadero de nu.
    true_xi : float
        Valor verdadero de xi.
    reps : int, opcional
        Número de simulaciones (por defecto 1000).
    seed : int o None, opcional
        Semilla para reproducibilidad.

    Retorna
    -------
    summary_df : pd.DataFrame
        Tabla con los resultados: parámetro, promedio, IC inferior, IC superior y ECM.
    """
    nu_estimates, xi_estimates = simulate_estimation(n, true_nu, true_xi, xi_init, nu_init, reps)

    # Calcular estadísticas para nu
    nu_mean = np.mean(nu_estimates)
    nu_ic = np.percentile(nu_estimates, [2.5, 97.5])
    nu_mse = np.mean((nu_estimates - true_nu)**2)

    # Calcular estadísticas para xi
    xi_mean = np.mean(xi_estimates)
    xi_ic = np.percentile(xi_estimates, [2.5, 97.5])
    xi_mse = np.mean((xi_estimates - true_xi)**2)

    # Crear DataFrame resumen
    summary_df = pd.DataFrame({
        "Parámetro": ["nu", "xi"],
        "Media": [nu_mean, xi_mean],
        "IC 2.5%": [nu_ic[0], xi_ic[0]],
        "IC 97.5%": [nu_ic[1], xi_ic[1]],
        "ECM": [nu_mse, xi_mse]
    })
    return summary_df

# ======================================================
# Ejemplo de uso final
# ======================================================
if __name__ == '__main__':
    # Parámetros verdaderos de la ITL
    true_nu = 2.0
    true_xi = 1.0
    xi_init = 1.0
    nu_init = 0.7
    # Tamaño de muestra a usar en cada simulación
    n = 100  # Puedes ajustarlo según convenga

    # Número de réplicas
    reps = 10000

    # Semilla para reproducibilidad

    # Realiza las simulaciones y obtiene la tabla resumen
    summary_table = generate_summary_table(n, true_nu, true_xi, reps, nu_init = nu_init, xi_init=xi_init )
    print("Tabla resumen de estimaciones (con {} réplicas y muestra de tamaño {}):".format(reps, n))
    print(summary_table)

def nu_estimation(Y, xi_0):
    """
    Calcula el estimador de ν según la fórmula:

    ν̂ = -n [Σ ln((2+Y_i/ξ₀)Y_i/ξ₀) / (1+Y_i/ξ₀)²]^(-1)

    Parameters:
    -----------
    Y : numpy.ndarray
        Vector de datos observados Y_i
    xi_0 : float
        Valor del parámetro ξ₀

    Returns:
    --------
    float
        Estimador de ν
    """
    # Obtenemos el tamaño de la muestra
    n = len(Y)

    # Calculamos Y_i/ξ₀ una sola vez para reutilizarlo
    Y_sobre_xi = Y / xi_0

    # Calculamos el término dentro del logaritmo: (2+Y_i/ξ₀)Y_i/ξ₀
    numerador = (2 + Y_sobre_xi) * Y_sobre_xi

    # Calculamos el denominador: (1+Y_i/ξ₀)²
    denominador = np.square(1 + Y_sobre_xi)

    # Calculamos ln[(2+Y_i/ξ₀)Y_i/ξ₀ / (1+Y_i/ξ₀)²]
    terminos_log = np.log(numerador / denominador)

    # Calculamos la suma de los términos logarítmicos
    suma_log = np.sum(terminos_log)

    # Calculamos el estimador de ν
    nu_estimado = -n / suma_log

    return nu_estimado

Y = simulate_itl_inverse_transform(100, 2.0, 0.993807)
nu = nu_estimation(Y,0.993807)
nu

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous, probplot

# Función de densidad de la distribución Inversa Topp-Leone


# Crear la clase personalizada usando rv_continuous
class inv_topp_leone_gen(rv_continuous):
    """
    Distribución Inversa Topp-Leone.
    """
    # Se definen las funciones básicas de la distribución
    def _pdf(self, y, v, xi):
        return pdf_inverted_topp_leone(y, v, xi)

    def _cdf(self, y, v, xi):
        return cdf_inverted_topp_leone(y, v, xi)

    def _ppf(self, p, v, xi):
        return q_inverted_topp_leone(p, v, xi)

# Instanciar la distribución "congelada".
# El argumento 'shapes' especifica los nombres de los parámetros extras (v y xi) que la distribución acepta.
inv_topp_leone = inv_topp_leone_gen(name="inv_topp_leone", shapes="v, xi")

# Ejemplo de uso: generar datos aleatorios usando el método de la transformación inversa
# y graficar la probabilidad (QQ-Plot) con scipy.stats.probplot.
v = 2.0   # parámetro de forma
xi = 1.0  # parámetro de escala

# Generar una muestra de la distribución utilizando la función cuantil (inversa de la CDF)
u = np.random.uniform(low=0.0, high=1.0, size=1000)
sample = inv_topp_leone.ppf(u, v, xi)

# Generar el probability plot
plt.figure(figsize=(8, 6))
probplot(sample, dist=inv_topp_leone, sparams=(v, xi), plot=plt)
plt.title("Probability Plot para la distribución Inversa Topp-Leone")
plt.xlabel("Cuantiles Teóricos")
plt.ylabel("Datos Ordenados")
plt.show()

valores_flotantes = [
    1.01, 1.11, 1.13, 1.15, 1.16, 1.17, 1.17, 1.20, 1.52, 1.54, 1.54, 1.57, 1.64,
    1.73, 1.79, 2.09, 2.09, 2.57, 2.75, 2.93, 3.19, 3.54, 3.57, 5.11, 5.62
]

"""# Dataset 1
**represent 40 patients suffering from blood cancer (Leukemia) from one ministry of health hospital in Saudi Arabia. The ordered life time (in years) are given as follows:**
"""

data = [
    0.315, 0.496, 0.616, 1.145, 1.208, 1.263, 1.414, 2.025, 2.036, 2.162,
    2.211, 2.370, 2.532, 2.693, 2.805, 2.910, 2.912, 3.192, 3.263, 3.348,
    3.348, 3.427, 3.499, 3.534, 3.767, 3.751, 3.858, 3.986, 4.049, 4.244,
    4.323, 4.381, 4.392, 4.397, 4.647, 4.753, 4.929, 4.973, 5.074, 4.381
]

v,xi = estimate_itl_params(data, nu_init= 0.5, xi_init=0.5)

probplot(data, dist=inv_topp_leone, sparams=(v, xi), plot=plt)
plt.title("Probability Plot for the Inverted Topp-Leone Distribution")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Ordered Data")
plt.show()

"""# Dataset 2

**consists of the number of successive failures for the air conditioning system of each member in a fleet of 13 Boeing 720 jet airplanes, see [30]. The actual data are:**
"""

data = [
    194, 413, 90, 74, 55, 23, 97, 50, 359, 50, 130, 487, 57, 102, 15, 14, 10, 57,
    320, 261, 51, 44, 9, 254, 493, 33, 18, 209, 41, 58, 60, 48, 56, 87, 11, 102,
    12, 5, 14, 14, 29, 37, 186, 29, 104, 7, 4, 72, 270, 283, 7, 61, 100, 61, 502,
    220, 120, 141, 22, 603, 35, 98, 54, 100, 11, 181, 65, 49, 12, 239, 14, 18,
    39, 3, 12, 5, 32, 9, 438, 43, 134, 184, 20, 386, 182, 71, 80, 188, 230, 152,
    5, 36, 79, 59, 33, 246, 1, 79, 3, 27, 201, 84, 27, 156, 21, 16, 88, 130, 14,
    118, 44, 15, 42, 106, 46, 230, 26, 59, 153, 104, 20, 206, 5, 66, 34, 29, 26,
    35, 5, 82, 31, 118, 326, 12, 54, 36, 34, 18, 25, 120, 31, 22, 18, 216, 139,
    67, 310, 3, 46, 210, 57, 76, 14, 111, 97, 62, 39, 30, 7, 44, 11, 63, 23, 22,
    23, 14, 18, 13, 34, 16, 18, 130, 90, 163, 208, 1, 24, 70, 16, 101, 52, 208,
    95, 62, 11, 191, 14, 7
]
v,xi = estimate_itl_params(data,nu_init= 0.5, xi_init=0.5)

probplot(data, dist=inv_topp_leone, sparams=(v, xi), plot=plt)
plt.title("Probability Plot for the Inverted Topp-Leone Distribution")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Ordered Data")
plt.show()

"""# Dataet 3


"""

data = [
    83, 51, 87, 60, 28, 95, 8, 27, 15, 10, 18, 16, 29, 54, 91, 8,
    17, 55, 10, 35, 47, 77, 36, 17, 21, 36, 18, 40, 10, 7, 34, 27,
    28, 56, 8, 25, 68, 146, 89, 18, 73, 69, 9, 37, 10, 82, 29, 8,
    60, 61, 61, 18, 169, 25, 8, 26, 11, 83, 11, 42, 17, 14, 9, 12
]

v,xi = estimate_itl_params(data,nu_init= 0.5, xi_init=0.5)

probplot(data, dist=inv_topp_leone, sparams=(v, xi), plot=plt)
plt.title("Probability Plot for the Inverted Topp-Leone Distribution")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Ordered Data")
plt.show()



plt.figure(figsize=(8, 6))
v = 1.316395904583922
xi = 4.837765004098943
probplot(valores_flotantes, dist=inv_topp_leone, sparams=(v, xi), plot=plt)
plt.title("Probability Plot for the Inverted Topp-Leone Distribution")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Ordered Data")
plt.show()

data = [
    94, 413, 90, 74, 55, 23, 97, 50, 359, 50, 130, 487, 57, 102, 15, 14, 10, 57, 320, 261, 51, 44, 9, 254, 493, 33,
    18, 209, 41, 58, 60, 48, 56, 87, 11, 102, 12, 5, 14, 14, 29, 37, 186, 29, 104, 7, 4, 72, 270, 283, 7, 61, 100,
    61, 502, 220, 120, 141, 22, 603, 35, 98, 54, 100, 11, 181, 65, 49, 12, 239, 14, 18, 39, 3, 12, 5, 32, 9, 438,
    43, 134, 184, 20, 386, 182, 71, 80, 188, 230, 152, 5, 36, 79, 59, 33, 246, 1, 79, 3, 27, 201, 84, 27, 156, 21,
    16, 88, 130, 14, 118, 44, 15, 42, 106, 46, 230, 26, 59, 153, 104, 20, 206, 5, 66, 34, 29, 26, 35, 5, 82, 31,
    118, 326, 12, 54, 36, 34, 18, 25, 120, 31, 22, 18, 216, 139, 67, 310, 3, 46, 210, 57, 76, 14, 111, 97, 62, 39,
    30, 7, 44, 11, 63, 23, 22, 23, 14, 18, 13, 34, 16, 18, 130, 90, 163, 208, 1, 24, 70, 16, 101, 52, 208, 95, 62,
    11, 191, 14, 7
]
v,xi = estimate_itl_params(data,nu_init= 0.5, xi_init=0.5)

probplot(data, dist=inv_topp_leone, sparams=(v, xi), plot=plt)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous, probplot, invgamma, invweibull
from scipy.optimize import brentq

# -----------------------------
# Tus datos
# -----------------------------
data1 = np.array([
    5.1, 1.2, 1.3, 0.6, 0.5, 2.4, 0.5, 1.1, 8.0, 0.8, 0.4, 0.6, 0.9, 0.4, 2.0,
    0.5, 5.3, 3.2, 2.7, 2.9, 2.5, 2.3, 1.0, 0.2, 0.1, 0.1, 1.8, 0.9, 2.0, 4.0,
    6.8, 1.2, 0.4, 0.2
])

inv_topp_leone = inv_topp_leone_gen(a=0, name="inv_topp_leone")
v, xi = estimate_itl_params(data1)
probplot(data1, dist=inv_topp_leone, sparams=(v, xi), plot=plt)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous, probplot, invgamma, invweibull
from scipy.optimize import brentq

# -----------------------------
# Tus datos
# -----------------------------
data1 = np.array([
    5.1, 1.2, 1.3, 0.6, 0.5, 2.4, 0.5, 1.1, 8.0, 0.8, 0.4, 0.6, 0.9, 0.4, 2.0,
    0.5, 5.3, 3.2, 2.7, 2.9, 2.5, 2.3, 1.0, 0.2, 0.1, 0.1, 1.8, 0.9, 2.0, 4.0,
    6.8, 1.2, 0.4, 0.2
])





inv_topp_leone = inv_topp_leone_gen(a=0, name="inv_topp_leone")

# -----------------------------
# Estimar parámetros
# -----------------------------
v, xi = estimate_itl_params(data1)
params_ig = invgamma.fit(data1, floc=0)
params_iw = invweibull.fit(data1, floc=0)

# -----------------------------
# PP-Plot data (no graficar aún)
# -----------------------------
osm_itl, osr = probplot(data1, dist=inv_topp_leone, sparams=(v, xi), fit=False)
osm_ig, _ = probplot(data1, dist=invgamma, sparams=params_ig, fit=False)
osm_iw, _ = probplot(data1, dist=invweibull, sparams=params_iw, fit=False)

# -----------------------------
# Graficar todo junto
# -----------------------------
plt.figure(figsize=(10, 6))
plt.plot(osm_itl, osr, 'o', label="Inverted Topp-Leone", markersize=6, alpha=0.8)
plt.plot(osm_ig, osr, 's', label="Inverse Gamma", markersize=6, alpha=0.8)
plt.plot(osm_iw, osr, '^', label="Inverse Weibull", markersize=6, alpha=0.8)

# Línea de referencia
min_val, max_val = min(osr.min(), osm_itl.min()), max(osr.max(), osm_itl.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="45° Reference")

# Estética
plt.xlabel("Theoretical Quantiles", fontsize=12)
plt.ylabel("Ordered Data", fontsize=12)
plt.title("PP-Plot Comparison of Distributions", fontsize=14)
plt.legend(loc='upper left', fontsize=10, frameon=True)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


"https://www.sciencedirect.com/science/article/pii/S1687850723001218#:~:text=(2020)%20recently%20suggested%20the%20inversed,0%20%2C%20where%20is%20the%20shape"

from scipy.stats import kstest, weibull_min, invweibull, invgamma, expon, lomax
from scipy.special import gamma
# --- Information Criteria ---
def compute_ics(logL, n, k):
    AIC = -2 * logL + 2 * k
    BIC = -2 * logL + k * np.log(n)
    CAIC = -2 * logL + k * (np.log(n) + 1)
    HQIC = -2 * logL + 2 * k * np.log(np.log(n))
    return AIC, BIC, CAIC, HQIC

# --- Placeholder for GIE, EL, EE, IE distributions ---
def gie_pdf(x, alpha, lam):
    return alpha * lam**alpha * x**(-alpha - 1) * np.exp(- (lam / x)**alpha)

def gie_cdf(x, alpha, lam):
    return 1 - np.exp(- (lam / x)**alpha)

# --- Comparison function ---
def compare_distributions(y):
    n = len(y)
    results = []

    # ITL
    nu_hat, xi_hat = estimate_itl_params(y)
    logL = log_likelihood(nu_hat, xi_hat, y)
    AIC, BIC, CAIC, HQIC = compute_ics(logL, n, 2)
    results.append(('ITL', logL, AIC, BIC, CAIC, HQIC))


    # Inverse Weibull
    c, loc, scale = invweibull.fit(y, floc=0)
    logL = np.sum(invweibull.logpdf(y, c, loc=loc, scale=scale))
    AIC, BIC, CAIC, HQIC = compute_ics(logL, n, 2)
    results.append(('Inverse Weibull', logL, AIC, BIC, CAIC, HQIC))

    # Inverse Gamma
    a, loc, scale = invgamma.fit(y, floc=0)
    logL = np.sum(invgamma.logpdf(y, a, loc=loc, scale=scale))
    AIC, BIC, CAIC, HQIC = compute_ics(logL, n, 2)
    results.append(('Inverse Gamma', logL, AIC, BIC, CAIC, HQIC))

    # Exponentiated Lomax (EL), Generalized Inverted Exp (GIE), EE, IE to be implemented manually
    # Placeholder: will fill in in next step if needed

    df = pd.DataFrame(results, columns=[
        'Model', 'Log-Likelihood', 'AIC', 'BIC', 'CAIC', 'HQIC'
    ])
    return df

compare_data1 = compare_distributions(data1)

# prompt: compare_data1 exportar como  imagen esta tabla

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))  # Adjust figure size as needed
plt.axis('off')  # Hide axes
plt.table(cellText=compare_data1.values, colLabels=compare_data1.columns, loc='center', cellLoc='center')
plt.title("Comparison of Distributions")
plt.savefig('compare_data1.png', bbox_inches='tight', pad_inches=0.5) # Save as image
plt.show()

data2 = [
1.43, 0.11, 0.71, 0.77, 2.63, 1.49, 3.46, 2.46, 0.59, 0.74, 1.23, 0.94, 4.36, 0.40, 1.74, 4.73, 2.23, 0.45, 0.70, 1.06, 1.46, 0.30, 1.82, 2.37, 0.63, 1.23, 1.24, 1.97, 1.86, 1.17
         ]
v,xi = estimate_itl_params(data2,nu_init= 0.5, xi_init=0.5)

inv_topp_leone = inv_topp_leone_gen(a=0, name="inv_topp_leone")

# -----------------------------
# Estimar parámetros
# -----------------------------
params_ig = invgamma.fit(data2, floc=0)
params_iw = invweibull.fit(data2, floc=0)

# -----------------------------
# PP-Plot data (no graficar aún)
# -----------------------------
osm_itl, osr = probplot(data2, dist=inv_topp_leone, sparams=(v, xi), fit=False)
osm_ig, _ = probplot(data2, dist=invgamma, sparams=params_ig, fit=False)
osm_iw, _ = probplot(data2, dist=invweibull, sparams=params_iw, fit=False)

# -----------------------------
# Graficar todo junto
# Cálculo de regresión lineal de ITL
slope, intercept = np.polyfit(osm_itl, osr, 1)
regression_line = slope * osm_itl + intercept

plt.figure(figsize=(10, 6))
plt.plot(osm_itl, osr, 'o', label="Inverted Topp-Leone", markersize=6, alpha=0.8)
plt.plot(osm_ig, osr, 's', label="Inverse Gamma", markersize=6, alpha=0.8)
plt.plot(osm_iw, osr, '^', label="Inverse Weibull", markersize=6, alpha=0.8)
plt.plot(osm_itl, regression_line, 'k--', lw=2, label=f"Theorical quantile")

# Línea de referencia

# Estética
plt.xlabel("Theoretical Quantiles", fontsize=12)
plt.ylabel("Ordered Data", fontsize=12)
plt.title("PP-Plot Comparison of Distributions", fontsize=14)
plt.legend(loc='upper left', fontsize=10, frameon=True)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

compare_data2 = compare_distributions(data)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))  # Adjust figure size as needed
plt.axis('off')  # Hide axes
plt.table(cellText=compare_data2.values, colLabels=compare_data2.columns, loc='center', cellLoc='center')
plt.title("Comparison of Distributions")
plt.savefig('compare_data1.png', bbox_inches='tight', pad_inches=0.5) # Save as image
plt.show()

import numpy as np
import pandas as pd

def mortality_rate_by_country_year(country, year):
    """
    Calculates and returns a DataFrame with the mortality rate for a given country and year.

    Args:
        country: The name of the country.
        year: The year.

    Returns:
        A pandas DataFrame with the mortality rate, or None if data is not found.
    """
    try:
        df = pd.read_csv('/content/drive/MyDrive/WHO-COVID-19-global-daily-data (2).csv', low_memory=False)
        # Convert Date_reported to datetime objects
        df['Date_reported'] = pd.to_datetime(df['Date_reported'])
        df['Year'] = df['Date_reported'].dt.year

        # Filter data for the given country and year
        df_filtered = df[(df['Country'] == country) & (df['Year'] == year)]

        # Handle cases where new_cases or new_deaths might be zero to avoid division by zero
        df_filtered['rate_mortality'] = np.where(df_filtered['New_cases'] == 0, 0, df_filtered['New_deaths'] / df_filtered['New_cases'])


        # Sort by mortality rate
        df_filtered = df_filtered.sort_values(by='rate_mortality')

        return df_filtered[['Date_reported', 'Country', 'New_cases', 'New_deaths', 'rate_mortality']]

    except FileNotFoundError:
        print(f"Error: File not found at /content/drive/MyDrive/WHO-COVID-19-global-daily-data (2).csv")
        return None
    except KeyError as e:
        print(f"Error: Column {e} not found in the CSV file.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None



# prompt: eliminar 0 y negativos de data

import numpy as np

df = mortality_rate_by_country_year("Colombia",2023)
data = df['rate_mortality'].sort_values()
data = list(data)
data = [x for x in data if x > 0]

v,xi = estimate_itl_params(data,0.1)

probplot(data, dist=inv_topp_leone, sparams=(v, xi), plot=plt)
plt.title("Probability Plot for the Inverted Topp-Leone Distribution")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Ordered Data")
plt.show()

v,xi = estimate_itl_params(data,0.1)

probplot(data, dist=inv_topp_leone, sparams=(v, xi), plot=plt)
plt.title("Probability Plot for the Inverted Topp-Leone Distribution")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Ordered Data")
plt.show()