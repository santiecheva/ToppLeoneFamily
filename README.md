# ToppLeoneFamily

Librería Python que implementa la familia de distribuciones Topp–Leone, su versión invertida y el modelo bivariado invertido. Ofrece clases orientadas a objetos, estimación por máxima verosimilitud, simulación Monte Carlo y utilidades estadísticas de apoyo.

## Estructura del proyecto

```
ToppLeoneFamily/
├── __init__.py                # Punto de entrada del paquete
├── distributions/             # Clases de distribución
│   ├── base.py                # Clase abstracta común y utilidades básicas
│   ├── topp_leone.py          # Distribución Topp–Leone
│   ├── inverted_topp_leone.py # Distribución Topp–Leone invertida
│   └── biv_inverted_topp_leone.py # Distribución Topp–Leone invertida bivariada
├── estimation/                # Contenedores y helpers para estimación
│   └── __init__.py
├── simulation/                # Resúmenes de simulaciones Monte Carlo
│   └── __init__.py
└── statistics/                # Funciones estadísticas auxiliares
    └── __init__.py
```

## Requisitos

Instala las dependencias mínimas con:

```bash
pip install -r requirements.txt
```

## Uso básico

```python
from ToppLeoneFamily import ToppLeoneDistribution

# Instanciar la distribución
nu, sigma = 2.5, 1.5
model = ToppLeoneDistribution(nu, sigma)

# Densidad, distribución y cuantiles
x = [0.1, 0.5, 1.0]
pdf_vals = model.pdf(x)
cdf_vals = model.cdf(x)
quantiles = model.ppf([0.1, 0.5, 0.9])

# Estadísticos
media = model.mean()
varianza = model.variance()

# Simulación
samples = model.sample(size=1000)
```

## Estimación por máxima verosimilitud

```python
import numpy as np
from ToppLeoneFamily import ToppLeoneDistribution

rng = np.random.default_rng(42)
true_nu, true_sigma = 1.0, 2.0
model = ToppLeoneDistribution(true_nu, true_sigma, rng=rng)

# Generar muestra artificial
sample = model.sample(size=200, random_state=rng)

# Ajustar MLE
fit = ToppLeoneDistribution.fit_mle(sample)
print("Parametros estimados:", fit.params)
```

## Simulaciones Monte Carlo

```python
from ToppLeoneFamily import ToppLeoneDistribution

result = ToppLeoneDistribution.simulate_mle_statistics(
    nu_true=1.0,
    sigma_true=2.0,
    sample_size=50,
    reps=500,
)
print(result)
```

## Distribución Topp–Leone invertida

```python
from ToppLeoneFamily import InvertedToppLeoneDistribution

model = InvertedToppLeoneDistribution(nu=2.0, xi=1.0)
values = model.sample(size=1000)
fit = InvertedToppLeoneDistribution.fit_mle(values)
print("Estimacion (nu, xi):", fit.params)
```

## Distribución Topp–Leone invertida bivariada

```python
from ToppLeoneFamily import BivariateInvertedToppLeoneDistribution

biv = BivariateInvertedToppLeoneDistribution(nu1=1.0, nu2=2.0, xi=1.0)

# Evaluar densidad conjunta en grids
import numpy as np
x = np.linspace(0.1, 2.0, 10)
y = np.linspace(0.1, 2.0, 10)
pdf_grid = biv.pdf(x[:, None], y[None, :])

# Tabla de correlaciones para distintos parámetros
corr_df = BivariateInvertedToppLeoneDistribution.correlation_grid(
    nu1_values=[0.5, 1.0, 2.0],
    nu2_values=[1.0, 2.0, 3.0],
    xi=1.0,
    as_dataframe=True,
)
print(corr_df)
```

## Apoyo estadístico

```python
from ToppLeoneFamily import empirical_ci

interval = empirical_ci([1.0, 1.2, 1.3, 0.9])
print("IC 95%:", interval)
```

## Notebooks originales

Los notebooks usados como referencia permanecen en `notebooks/` para consulta y comparación.

## Licencia

Pendiente de definir.
