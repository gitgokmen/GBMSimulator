from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray

@dataclass(frozen=True)
class RiskStatistics:

    mean: float
    std: float
    var_95: float
    var_99: float
    es_95: float
    es_99: float
    median: float
    min: float
    max: float
    paths_below_s0: float


class GBMSimulator:
   
    _SUPPORTED_CONFIDENCE_LEVELS: tuple[float, ...] = (0.95, 0.99)

    def __init__(
        self,
        s0: float,
        mu: float,
        sigma: float,
        days: int,
        num_paths: int,
        trading_days_per_year: int = 252,
        seed: Optional[int] = None,
    ) -> None:
        self._validate_inputs(
            s0, mu, sigma, days, num_paths, trading_days_per_year, seed,
        )

        self._s0: float = float(s0)
        self._mu: float = float(mu)
        self._sigma: float = float(sigma)
        self._days: int = int(days)
        self._num_paths: int = int(num_paths)
        self._trading_days_per_year: int = int(trading_days_per_year)
        self._seed: Optional[int] = seed

        self._dt: float = 1.0 / self._trading_days_per_year
        self._drift_per_step: float = (
            self._mu - 0.5 * self._sigma ** 2
        ) * self._dt
        self._diffusion_per_step: float = self._sigma * np.sqrt(self._dt)

        self._paths: Optional[NDArray[np.float64]] = None



    @staticmethod
    def _validate_inputs(
        s0: float,
        mu: float,
        sigma: float,
        days: int,
        num_paths: int,
        trading_days_per_year: int,
        seed: Optional[int],
    ) -> None:
        
        if not isinstance(s0, (int, float)):
            raise TypeError(
                f"s0 must be int or float, got {type(s0).__name__}"
            )
        if not isinstance(mu, (int, float)):
            raise TypeError(
                f"mu must be int or float, got {type(mu).__name__}"
            )
        if not isinstance(sigma, (int, float)):
            raise TypeError(
                f"sigma must be int or float, got {type(sigma).__name__}"
            )
        if not isinstance(days, int):
            raise TypeError(
                f"days must be int, got {type(days).__name__}"
            )
        if not isinstance(num_paths, int):
            raise TypeError(
                f"num_paths must be int, got {type(num_paths).__name__}"
            )
        if not isinstance(trading_days_per_year, int):
            raise TypeError(
                f"trading_days_per_year must be int, "
                f"got {type(trading_days_per_year).__name__}"
            )
        if seed is not None and not isinstance(seed, int):
            raise TypeError(
                f"seed must be int or None, got {type(seed).__name__}"
            )

        if s0 <= 0:
            raise ValueError(f"s0 must be positive, got {s0}")
        if sigma < 0:
            raise ValueError(
                f"sigma (volatility) must be non-negative, got {sigma}"
            )
        if days < 1:
            raise ValueError(f"days must be >= 1, got {days}")
        if num_paths < 1:
            raise ValueError(f"num_paths must be >= 1, got {num_paths}")
        if trading_days_per_year < 1:
            raise ValueError(
                f"trading_days_per_year must be >= 1, "
                f"got {trading_days_per_year}"
            )

        if np.isinf(s0) or np.isnan(s0):
            raise ValueError("s0 must be finite")
        if np.isinf(mu) or np.isnan(mu):
            raise ValueError("mu must be finite")
        if np.isinf(sigma) or np.isnan(sigma):
            raise ValueError("sigma must be finite")

    

    def simulate(self) -> NDArray[np.float64]:
        rng = np.random.default_rng(self._seed)

        increments: NDArray[np.float64] = (
            self._drift_per_step
            + self._diffusion_per_step
            * rng.standard_normal(
                size=(self._num_paths, self._days),
                dtype=np.float64,
            )
        )

        log_paths: NDArray[np.float64] = np.cumsum(increments, axis=1)

        paths = np.empty(
            (self._num_paths, self._days + 1), dtype=np.float64,
        )
        paths[:, 0] = self._s0
        np.exp(log_paths, out=paths[:, 1:])
        paths[:, 1:] *= self._s0

        self._paths = paths
        return paths


    def risk_statistics(self) -> RiskStatistics:
        if self._paths is None:
            raise RuntimeError(
                "No simulation data available. Call simulate() first."
            )

        terminal: NDArray[np.float64] = self._paths[:, -1]

        pnl: NDArray[np.float64] = terminal - self._s0

        sorted_pnl: NDArray[np.float64] = np.sort(pnl)

        var_95 = -float(np.percentile(sorted_pnl, 5))
        var_99 = -float(np.percentile(sorted_pnl, 1))

        cutoff_95 = int(np.ceil(len(sorted_pnl) * 0.05))
        cutoff_99 = int(np.ceil(len(sorted_pnl) * 0.01))

        es_95 = -float(np.mean(sorted_pnl[:cutoff_95])) if cutoff_95 > 0 else var_95
        es_99 = -float(np.mean(sorted_pnl[:cutoff_99])) if cutoff_99 > 0 else var_99

        return RiskStatistics(
            mean=float(np.mean(terminal)),
            std=float(np.std(terminal, ddof=1)),
            var_95=var_95,
            var_99=var_99,
            es_95=es_95,
            es_99=es_99,
            median=float(np.median(terminal)),
            min=float(np.min(terminal)),
            max=float(np.max(terminal)),
            paths_below_s0=float(np.mean(terminal < self._s0)),
        )


    @property
    def paths(self) -> Optional[NDArray[np.float64]]:
        return self._paths

    @property
    def terminal_prices(self) -> NDArray[np.float64]:
       
        if self._paths is None:
            raise RuntimeError(
                "No simulation data available. Call simulate() first."
            )
        return self._paths[:, -1].copy()

    def __repr__(self) -> str:
        return (
            f"GBMSimulator(s0={self._s0}, mu={self._mu}, "
            f"sigma={self._sigma}, days={self._days}, "
            f"num_paths={self._num_paths}, "
            f"seed={self._seed})"
        )



def main() -> None:
       
    sim = GBMSimulator(
        s0=100.0,
        mu=0.08,
        sigma=0.20,
        days=252,
        num_paths=100_000,
        seed=42,
    )

    sim.simulate()
    stats = sim.risk_statistics()

    print(f"{'=' * 52}")
    print(f"  GBM Simulation — {sim}")
    print(f"{'=' * 52}")
    print(f"  Mean terminal price      : {stats.mean:>12.4f}")
    print(f"  Std  terminal price      : {stats.std:>12.4f}")
    print(f"  Median                   : {stats.median:>12.4f}")
    print(f"  Min / Max                : {stats.min:>8.4f} / {stats.max:.4f}")
    print(f"  VaR  95 %                : {stats.var_95:>12.4f}")
    print(f"  VaR  99 %                : {stats.var_99:>12.4f}")
    print(f"  ES   95 %                : {stats.es_95:>12.4f}")
    print(f"  ES   99 %                : {stats.es_99:>12.4f}")
    print(f"  Paths below S0           : {stats.paths_below_s0:>11.2%}")
    print(f"{'=' * 52}")


if __name__ == "__main__":
    main()
