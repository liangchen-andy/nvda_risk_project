"""Market risk calculations for the NVDA risk pipeline."""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.stats import t as student_t


def compute_historical_var_es(
    returns: pd.Series,
    confidence_level: float,
) -> tuple[float, float]:
    """Compute historical VaR and ES from return series."""
    if not 0 < confidence_level < 1:
        msg = "confidence_level must be between 0 and 1."
        raise ValueError(msg)

    clean_returns = pd.to_numeric(returns, errors="coerce").dropna()
    if clean_returns.empty:
        msg = "returns must contain at least one valid observation."
        raise ValueError(msg)

    tail_probability = 1 - confidence_level
    var_cutoff = float(clean_returns.quantile(tail_probability))
    es = float(clean_returns[clean_returns <= var_cutoff].mean())
    return var_cutoff, es


def compute_parametric_var_es(
    returns: pd.Series,
    confidence_level: float,
) -> tuple[float, float]:
    """Compute Gaussian parametric VaR and ES from a return series."""
    if not 0 < confidence_level < 1:
        msg = "confidence_level must be between 0 and 1."
        raise ValueError(msg)

    clean_returns = pd.to_numeric(returns, errors="coerce").dropna()
    if clean_returns.empty:
        msg = "returns must contain at least one valid observation."
        raise ValueError(msg)

    mu = float(clean_returns.mean())
    sigma = float(clean_returns.std(ddof=1)) if clean_returns.shape[0] > 1 else 0.0
    tail_probability = 1 - confidence_level
    standard_normal = NormalDist(mu=0.0, sigma=1.0)
    z = float(standard_normal.inv_cdf(tail_probability))
    var_cutoff = mu + sigma * z
    es = mu - sigma * standard_normal.pdf(z) / tail_probability
    return float(var_cutoff), float(es)


def compute_garch_t_var_es(
    returns: pd.Series,
    confidence_level: float,
    *,
    min_sample_size: int = 120,
) -> dict[str, float | int | str]:
    """Compute one-step-ahead VaR/ES using GARCH(1,1)-t with QMLE volatility fit."""
    if not 0 < confidence_level < 1:
        msg = "confidence_level must be between 0 and 1."
        raise ValueError(msg)
    if min_sample_size <= 1:
        msg = "min_sample_size must be greater than 1."
        raise ValueError(msg)

    clean_returns = pd.to_numeric(returns, errors="coerce").dropna()
    sample_size = int(clean_returns.shape[0])
    if sample_size < min_sample_size:
        return {
            "var": float("nan"),
            "es": float("nan"),
            "status": "fallback",
            "fallback_reason": "short_sample",
            "sample_size": sample_size,
            "nu": float("nan"),
            "sigma_next": float("nan"),
            "converged": False,
        }

    def _garch_qmle_objective(params: np.ndarray, eps: np.ndarray, initial_var: float) -> float:
        """Return Gaussian QMLE objective with parameter constraints."""
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e12

        conditional_var = np.empty_like(eps)
        conditional_var[0] = initial_var
        for idx in range(1, len(eps)):
            conditional_var[idx] = (
                omega
                + alpha * (eps[idx - 1] ** 2)
                + beta * conditional_var[idx - 1]
            )

        if np.any(~np.isfinite(conditional_var)) or np.any(conditional_var <= 0):
            return 1e12

        return float(
            0.5 * np.sum(np.log(conditional_var) + (eps**2) / conditional_var),
        )

    try:
        mu = float(clean_returns.mean())
        eps = clean_returns.to_numpy(dtype=float) - mu
        sample_var = float(np.var(eps, ddof=1))
        if not np.isfinite(sample_var) or sample_var <= 0:
            msg = "returns variance must be positive."
            raise ValueError(msg)

        initial_guess = np.array(
            [max(sample_var * 0.02, 1e-10), 0.05, 0.90],
            dtype=float,
        )
        bounds = (
            (1e-12, max(sample_var * 10.0, 1e-4)),
            (1e-6, 0.35),
            (1e-6, 0.999),
        )
        fit = minimize(
            _garch_qmle_objective,
            x0=initial_guess,
            args=(eps, sample_var),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if fit.success and float(fit.x[1] + fit.x[2]) < 0.999:
            omega, alpha, beta = (float(fit.x[0]), float(fit.x[1]), float(fit.x[2]))
            converged = True
        else:
            # Deterministic fallback parameters preserve offline robustness.
            alpha = 0.05
            beta = 0.90
            omega = sample_var * (1 - alpha - beta)
            converged = False

        conditional_var = np.empty_like(eps)
        conditional_var[0] = sample_var
        for idx in range(1, sample_size):
            conditional_var[idx] = omega + alpha * (eps[idx - 1] ** 2) + beta * conditional_var[idx - 1]

        next_var = omega + alpha * (eps[-1] ** 2) + beta * conditional_var[-1]
        sigma_next = float(np.sqrt(max(next_var, 1e-12)))

        standardized_residuals = eps / np.sqrt(np.maximum(conditional_var, 1e-12))
        if np.any(~np.isfinite(standardized_residuals)):
            msg = "non-finite standardized residuals."
            raise ValueError(msg)

        def _nu_objective(nu: float) -> float:
            """Negative log-likelihood for Student-t degrees of freedom."""
            scale = math.sqrt((nu - 2.0) / nu)
            scaled = standardized_residuals / scale
            logpdf = student_t.logpdf(scaled, df=nu) - np.log(scale)
            return float(-np.sum(logpdf))

        nu_fit = minimize_scalar(
            _nu_objective,
            bounds=(4.1, 120.0),
            method="bounded",
        )
        nu = float(nu_fit.x if nu_fit.success else 30.0)
        nu = float(np.clip(nu, 4.1, 120.0))

        tail_probability = 1 - confidence_level
        q = float(student_t.ppf(tail_probability, df=nu))
        pdf_q = float(student_t.pdf(q, df=nu))
        std_scale = math.sqrt((nu - 2.0) / nu)
        q_standardized = q * std_scale
        var_cutoff = float(mu + sigma_next * q_standardized)
        es_standardized = -std_scale * ((nu + q**2) / (nu - 1.0)) * (
            pdf_q / tail_probability
        )
        es = float(mu + sigma_next * es_standardized)

        if not np.isfinite(var_cutoff) or not np.isfinite(es):
            msg = "non-finite risk estimate."
            raise ValueError(msg)

        return {
            "var": var_cutoff,
            "es": es,
            "status": "ok",
            "fallback_reason": "none",
            "sample_size": sample_size,
            "nu": nu,
            "sigma_next": sigma_next,
            "converged": converged,
        }
    except ValueError:
        return {
            "var": float("nan"),
            "es": float("nan"),
            "status": "fallback",
            "fallback_reason": "invalid_input",
            "sample_size": sample_size,
            "nu": float("nan"),
            "sigma_next": float("nan"),
            "converged": False,
        }
    except Exception:
        return {
            "var": float("nan"),
            "es": float("nan"),
            "status": "fallback",
            "fallback_reason": "exception",
            "sample_size": sample_size,
            "nu": float("nan"),
            "sigma_next": float("nan"),
            "converged": False,
        }


def compute_rolling_volatility(
    returns: pd.Series,
    window: int,
    periods_per_year: int = 252,
) -> pd.Series:
    """Compute annualized rolling volatility from return series."""
    if window <= 0:
        msg = "window must be a positive integer."
        raise ValueError(msg)
    if periods_per_year <= 0:
        msg = "periods_per_year must be a positive integer."
        raise ValueError(msg)
    if returns.empty:
        return pd.Series(index=returns.index, dtype=float, name="rolling_vol")

    rolling_std = returns.astype(float).rolling(window=window, min_periods=window).std(ddof=1)
    rolling_vol = rolling_std * np.sqrt(periods_per_year)
    rolling_vol.name = "rolling_vol"
    return rolling_vol


def kupiec_pof_test(exceedance_count: int, sample_size: int, alpha: float) -> dict[str, float | bool]:
    """Run Kupiec POF test and return LR statistic, p-value, and reject flag."""
    if sample_size <= 0:
        msg = "sample_size must be a positive integer."
        raise ValueError(msg)
    if exceedance_count < 0:
        msg = "exceedance_count must be non-negative."
        raise ValueError(msg)
    if exceedance_count > sample_size:
        msg = "exceedance_count cannot exceed sample_size."
        raise ValueError(msg)
    if not 0 < alpha < 1:
        msg = "alpha must be between 0 and 1."
        raise ValueError(msg)

    expected_rate = 1 - alpha
    observed_rate = exceedance_count / sample_size

    def _log_likelihood(probability: float) -> float:
        """Compute Bernoulli log-likelihood with stable boundary handling."""
        if exceedance_count == 0:
            return sample_size * float(np.log1p(-probability))
        if exceedance_count == sample_size:
            return sample_size * float(np.log(probability))
        return (
            (sample_size - exceedance_count) * float(np.log1p(-probability))
            + exceedance_count * float(np.log(probability))
        )

    ll_null = _log_likelihood(expected_rate)
    ll_alt = _log_likelihood(observed_rate)
    lr_stat = max(0.0, float(-2.0 * (ll_null - ll_alt)))
    p_value = float(math.erfc(math.sqrt(lr_stat / 2.0)))
    return {
        "lr_stat": lr_stat,
        "p_value": p_value,
        "reject_5pct": bool(p_value < 0.05),
        "expected_exceedance_rate": float(expected_rate),
        "observed_exceedance_rate": float(observed_rate),
    }


def christoffersen_independence_test(exceedances: pd.Series) -> dict[str, float | bool | int]:
    """Run Christoffersen independence test on a binary exceedance sequence."""
    clean = pd.to_numeric(exceedances, errors="coerce").dropna()
    if clean.shape[0] < 2:
        msg = "exceedances must contain at least two valid observations."
        raise ValueError(msg)
    if not clean.isin([0, 1]).all():
        msg = "exceedances must be binary values (0 or 1)."
        raise ValueError(msg)

    sequence = clean.astype(int).to_numpy()
    previous = sequence[:-1]
    current = sequence[1:]

    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())

    transitions_total = n00 + n01 + n10 + n11
    exceedance_transitions = n01 + n11
    pi = exceedance_transitions / transitions_total

    def _binomial_log_likelihood(successes: int, trials: int, probability: float) -> float:
        """Compute stable binomial log-likelihood with edge-case handling."""
        if trials == 0:
            return 0.0
        if successes == 0:
            return trials * float(np.log1p(-probability))
        if successes == trials:
            return trials * float(np.log(probability))
        return (
            successes * float(np.log(probability))
            + (trials - successes) * float(np.log1p(-probability))
        )

    ll_null = _binomial_log_likelihood(successes=exceedance_transitions, trials=transitions_total, probability=pi)

    trials_0 = n00 + n01
    trials_1 = n10 + n11
    pi01 = n01 / trials_0 if trials_0 else 0.0
    pi11 = n11 / trials_1 if trials_1 else 0.0

    ll_alt = _binomial_log_likelihood(successes=n01, trials=trials_0, probability=pi01) + _binomial_log_likelihood(
        successes=n11,
        trials=trials_1,
        probability=pi11,
    )

    lr_stat = max(0.0, float(-2.0 * (ll_null - ll_alt)))
    p_value = float(math.erfc(math.sqrt(lr_stat / 2.0)))
    return {
        "lr_stat": lr_stat,
        "p_value": p_value,
        "reject_5pct": bool(p_value < 0.05),
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
    }


def build_rolling_volatility_frame(
    panel_daily: pd.DataFrame,
    window: int,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Build a tidy rolling-volatility frame from daily panel data."""
    out = panel_daily[["date", "ret"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    out["rolling_vol"] = compute_rolling_volatility(
        returns=out["ret"],
        window=window,
        periods_per_year=periods_per_year,
    )
    return out[["date", "rolling_vol"]]


def estimate_market_risk(panel: pd.DataFrame, var_level: float) -> pd.DataFrame:
    """Return annualized volatility, historical VaR, and ES from monthly returns."""
    returns = panel["nvda_return"].astype(float)
    var_cutoff = returns.quantile(var_level)
    es = returns[returns <= var_cutoff].mean()
    metrics = pd.DataFrame(
        {
            "metric": ["volatility_annualized", f"var_{int((1 - var_level) * 100)}", "es"],
            "value": [returns.std(ddof=1) * np.sqrt(12), var_cutoff, es],
        },
    )
    return metrics
