# Quantitative GBM Simulator

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Required-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A production-grade, fully vectorized Monte Carlo engine for simulating asset price paths under **Geometric Brownian Motion (GBM)** and computing terminal risk statistics. 

Designed for performance and robust risk analytics, it provides out-of-the-box support for calculating critical financial metrics such as Value at Risk (VaR) and Expected Shortfall (ES / CVaR).

---

## Table of Contents

- [Features](#features)
- [Mathematical Formulation](#mathematical-formulation)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Risk Metric Definitions](#risk-metric-definitions)
- [Error Handling](#error-handling)
- [Performance Notes](#performance-notes)
- [Running the Demo](#running-the-demo)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- 🚀 **Zero Python Loops**: Hot paths are fully vectorized using NumPy's BLAS-backed array operations.
- 📉 **Advanced Risk Metrics**: Computes terminal mean, standard deviation, VaR (95%, 99%), Expected Shortfall, and full path bounds.
- 🛡️ **Robust Validation**: Enforces strict type checking and constraint validation for financial inputs to ensure stability.
- 🔢 **Reproducible Scenarios**: Uses the PCG64 (`default_rng`) generator for high-quality, repeatable Monte Carlo simulations.
- 💻 **Minimal Dependencies**: Requires only Python and NumPy.

---

## Mathematical Formulation

The asset price $S_t$ evolves according to the stochastic differential equation for Geometric Brownian Motion:

$$S_t = S_0 \exp\!\left(\left(\mu - \tfrac{1}{2}\sigma^2\right)t + \sigma\, W_t\right)$$

| Symbol   | Description                        |
| -------- | ---------------------------------- |
| $S_0$    | Initial asset price                |
| $\mu$    | Annualized drift (expected return) |
| $\sigma$ | Annualized volatility              |
| $W_t$    | Standard Wiener process            |

Time is discretized into daily steps with $\Delta t = 1 / \text{trading\_days\_per\_year}$.

---

## Installation

```bash
pip install numpy
```

No other dependencies are required. Clone the repository and you are ready to go:
```bash
git clone https://github.com/yourusername/gbm-simulator.git
cd gbm-simulator
```

---

## Quick Start

```python
from gbm_simulator import GBMSimulator

sim = GBMSimulator(
    s0=100.0,
    mu=0.08,
    sigma=0.20,
    days=252,
    num_paths=100_000,
    seed=42,
)

paths = sim.simulate()          # shape (100_000, 253)
stats = sim.risk_statistics()

print(f"Mean terminal price : {stats.mean:.4f}")
print(f"95% VaR             : {stats.var_95:.4f}")
print(f"99% ES              : {stats.es_99:.4f}")
```

---

## API Reference

### `GBMSimulator`

#### Constructor

```python
GBMSimulator(
    s0: float,               # Initial price (> 0)
    mu: float,               # Annualized drift
    sigma: float,            # Annualized volatility (>= 0)
    days: int,               # Horizon in trading days (>= 1)
    num_paths: int,          # Monte Carlo paths (>= 1)
    trading_days_per_year: int = 252,
    seed: int | None = None, # Reproducibility
)
```

#### Methods

| Method              | Returns                                        | Description                                  |
| ------------------- | ---------------------------------------------- | -------------------------------------------- |
| `simulate()`        | `NDArray[float64]` shape `(num_paths, days+1)` | Generate all price paths. Column 0 is $S_0$. |
| `risk_statistics()` | `RiskStatistics`                               | Terminal-horizon risk summary (see below).   |

#### Properties

| Property          | Type              | Description                            |
| ----------------- | ----------------- | -------------------------------------- |
| `paths`           | `NDArray \| None` | Cached path matrix after `simulate()`. |
| `terminal_prices` | `NDArray`         | Copy of the last column of `paths`.    |

---

### `RiskStatistics`

Frozen dataclass returned by `risk_statistics()`.

| Field            | Type    | Description                                  |
| ---------------- | ------- | -------------------------------------------- |
| `mean`           | `float` | Mean terminal price                          |
| `std`            | `float` | Sample standard deviation of terminal prices |
| `var_95`         | `float` | 95% Value at Risk (loss relative to $S_0$)   |
| `var_99`         | `float` | 99% Value at Risk                            |
| `es_95`          | `float` | 95% Expected Shortfall (CVaR)                |
| `es_99`          | `float` | 99% Expected Shortfall                       |
| `median`         | `float` | Median terminal price                        |
| `min`            | `float` | Minimum terminal price observed              |
| `max`            | `float` | Maximum terminal price observed              |
| `paths_below_s0` | `float` | Fraction of paths ending below $S_0$         |

---

## Risk Metric Definitions

### Value at Risk (VaR)

The $\alpha$-level VaR is the loss threshold such that only $1 - \alpha$ percent of scenarios produce a worse outcome:

$$\text{VaR}_\alpha = -Q_{1-\alpha}(\text{PnL})$$

where $Q_p$ denotes the $p$-th quantile.

### Expected Shortfall (ES / CVaR)

The conditional expectation of losses exceeding VaR:

$$\text{ES}_\alpha = -\frac{1}{(1-\alpha) \cdot N}\sum_{i=1}^{\lfloor (1-\alpha) N \rceil} \text{PnL}_{(i)}$$

where $\text{PnL}_{(i)}$ are order statistics sorted ascending.

---

## Error Handling

The constructor enforces strict validation:

| Condition                               | Exception      |
| --------------------------------------- | -------------- |
| Wrong type for any parameter            | `TypeError`    |
| `s0 <= 0`                               | `ValueError`   |
| `sigma < 0`                             | `ValueError`   |
| `days < 1`                              | `ValueError`   |
| `num_paths < 1`                         | `ValueError`   |
| `NaN` / `Inf` in `s0`, `mu`, `sigma`    | `ValueError`   |
| `risk_statistics()` before `simulate()` | `RuntimeError` |

---

## Performance Notes

- **Zero Python loops** in the simulation hot path — all computation is delegated to NumPy's BLAS-backed array operations.
- Uses `np.cumsum` over log-increments followed by a single `np.exp`, avoiding repeated multiplications and reducing floating-point error accumulation.
- The `default_rng` (PCG64) generator is used for high-quality, reproducible random numbers.

---

## Running the Demo

```bash
python gbm_simulator.py
```

Produces a formatted table of risk statistics for a 100,000-path, 252-day simulation:

```text
====================================================
  GBM Simulation — GBMSimulator(s0=100.0, mu=0.08, sigma=0.2, days=252, num_paths=100000, seed=42)
====================================================
  Mean terminal price      :     108.3287
  Std  terminal price      :      22.1466
  Median                   :     106.1836
  Min / Max                :  36.5681 / 327.9174
  VaR  95 %                :      24.6067
  VaR  99 %                :      34.9080
  ES   95 %                :      29.8396
  ES   99 %                :      38.9959
  Paths below S0           :      38.08%
====================================================
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
