"""
Random Matrix Theory — Correlation Matrix Cleaning

Separates true signal from statistical noise in sample correlation matrices
using the Wigner -> Marchenko-Pastur -> Ledoit-Peche chain.

The core problem: with N assets and T observations, the sample correlation
matrix contains (N - k) noise eigenvalues that are statistically
indistinguishable from random. Only k eigenvalues carry real structure.

The Marchenko-Pastur law (1967) gives the exact noise boundary:
    lambda_pm = (1 +/- sqrt(q))^2   where q = N/T

Signal eigenvalues above lambda_+ are cleaned using the BBP phase
transition inversion (Baik-Ben Arous-Peche 2005, Ledoit-Peche 2011):
    tau = (lam + 1 - q + sqrt((lam + 1 - q)^2 - 4*lam)) / 2

Noise eigenvalues inside the MP band are replaced with their average
to preserve the matrix trace (= N for a correlation matrix).

References:
    Wigner (1955): Random matrix universality
    Marchenko & Pastur (1967): Distribution of eigenvalues of sample covariance
    Baik, Ben Arous, Peche (2005): Phase transitions for largest eigenvalues
    Ledoit & Peche (2011): arXiv:0911.3010
    Bun, Bouchaud, Potters (2017): Cleaning large correlation matrices
"""

import numpy as np
from typing import Dict, Tuple

from logs.logger import get_logger

logger = get_logger("rmt_cleaner")


# ---------------------------------------------------------------------------
# Marchenko-Pastur distribution
# ---------------------------------------------------------------------------

def mp_bounds(q: float) -> Tuple[float, float]:
    """
    Marchenko-Pastur noise eigenvalue bounds.

    Args:
        q: ratio N/T (assets / observations). Must be in (0, 1].

    Returns:
        (lambda_minus, lambda_plus)
    """
    sq = np.sqrt(q)
    return (1.0 - sq) ** 2, (1.0 + sq) ** 2


def mp_density(x: np.ndarray, q: float) -> np.ndarray:
    """
    Marchenko-Pastur probability density function.

    f_MP(lam) = sqrt((lam_+ - lam)(lam - lam_-)) / (2*pi*q*lam)
    for lam in [lam_-, lam_+]
    """
    lm, lp = mp_bounds(q)
    out = np.zeros_like(x, dtype=float)
    mask = (x >= lm) & (x <= lp) & (x > 0)
    out[mask] = np.sqrt((lp - x[mask]) * (x[mask] - lm)) / (2.0 * np.pi * q * x[mask])
    return out


# ---------------------------------------------------------------------------
# BBP spike cleaning (Baik-Ben Arous-Peche / Ledoit-Peche)
# ---------------------------------------------------------------------------

def _clean_spike(lam: float, q: float) -> float:
    """
    Clean a signal eigenvalue using the BBP phase transition inversion.

    The sample eigenvalue lam relates to the true population eigenvalue tau by:
        lam = tau * (1 + q / (tau - 1))    when tau > 1 + sqrt(q)

    Inverting gives:
        tau = (lam + 1 - q + sqrt((lam + 1 - q)^2 - 4*lam)) / 2

    This formula optimally debiases signal eigenvalues that stick out
    above the Marchenko-Pastur upper bound.
    """
    disc = (lam + 1.0 - q) ** 2 - 4.0 * lam
    if disc < 0:
        # Eigenvalue too close to bulk edge — treat as noise
        return 1.0
    return (lam + 1.0 - q + np.sqrt(disc)) / 2.0


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------

def clean_correlation_matrix(
    corr: np.ndarray,
    n_observations: int,
) -> Tuple[np.ndarray, Dict]:
    """
    Clean a sample correlation matrix using the full RMT chain.

    Steps:
        1. Eigendecompose C = V Lambda V^T
        2. Compute Marchenko-Pastur bounds  lam_pm = (1 +/- sqrt(q))^2
        3. Classify eigenvalues as signal (lam > lam_+) or noise (lam <= lam_+)
        4. Apply BBP/Ledoit-Peche shrinkage to signal eigenvalues
        5. Replace noise eigenvalues to preserve trace (trace = N)
        6. Reconstruct  C_clean = V Lambda_clean V^T  and renormalize diagonal to 1

    Args:
        corr: N x N sample correlation matrix (numpy array).
        n_observations: T, number of return observations used to estimate it.

    Returns:
        (cleaned_corr, diagnostics) where diagnostics is a dict with:
            q, N, T, mp_lower, mp_upper, n_signal, n_noise,
            signal_eigenvalues (raw), cleaned_signal_eigenvalues,
            noise_eigenvalue_avg, shrinkage_ratios
    """
    N = corr.shape[0]
    T = n_observations
    q = N / T

    # 1. Eigendecompose (symmetric, so use eigh for stability)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx].copy()
    eigenvectors = eigenvectors[:, idx].copy()

    # 2. Marchenko-Pastur bounds
    lm, lp = mp_bounds(q)

    # 3. Classify signal vs noise
    signal_mask = eigenvalues > lp
    n_signal = int(signal_mask.sum())
    n_noise = N - n_signal

    # 4. Clean signal eigenvalues via BBP inversion
    cleaned_eigenvalues = np.zeros(N)
    shrinkage_ratios = []

    for i in range(n_signal):
        raw = eigenvalues[i]
        cleaned = _clean_spike(raw, q)
        cleaned_eigenvalues[i] = cleaned
        shrinkage_ratios.append(round(cleaned / raw, 4))

    # 5. Set noise eigenvalues to preserve trace
    # Correlation matrix trace = N
    signal_sum = cleaned_eigenvalues[:n_signal].sum()
    remaining_trace = max(N - signal_sum, 0.0)

    if n_noise > 0:
        noise_fill = remaining_trace / n_noise
        cleaned_eigenvalues[n_signal:] = noise_fill
    else:
        # All eigenvalues are signal — scale to preserve trace
        cleaned_eigenvalues *= N / cleaned_eigenvalues.sum()

    # 6. Reconstruct
    cleaned_corr = eigenvectors @ np.diag(cleaned_eigenvalues) @ eigenvectors.T

    # Force unit diagonal (correlation matrix invariant)
    d = np.sqrt(np.diag(cleaned_corr))
    d[d < 1e-10] = 1.0
    cleaned_corr = cleaned_corr / np.outer(d, d)

    # Symmetry and clipping
    cleaned_corr = (cleaned_corr + cleaned_corr.T) / 2.0
    np.fill_diagonal(cleaned_corr, 1.0)
    np.clip(cleaned_corr, -1.0, 1.0, out=cleaned_corr)

    diagnostics = {
        "q": round(q, 4),
        "N": N,
        "T": T,
        "mp_lower": round(float(lm), 4),
        "mp_upper": round(float(lp), 4),
        "n_signal": n_signal,
        "n_noise": n_noise,
        "signal_eigenvalues": [round(float(e), 4) for e in eigenvalues[:n_signal]],
        "cleaned_signal_eigenvalues": [round(float(e), 4) for e in cleaned_eigenvalues[:n_signal]],
        "shrinkage_ratios": shrinkage_ratios,
        "noise_eigenvalue_avg": round(float(eigenvalues[n_signal:].mean()), 4) if n_noise > 0 else 0.0,
        "noise_fill_value": round(float(noise_fill), 4) if n_noise > 0 else 0.0,
        "total_variance_preserved": round(float(cleaned_eigenvalues.sum()), 4),
    }

    logger.info(
        "rmt_cleaning_complete",
        N=N, T=T, q=round(q, 3),
        mp_upper=round(float(lp), 3),
        n_signal=n_signal, n_noise=n_noise,
        signal_eigenvalues=diagnostics["signal_eigenvalues"],
        shrinkage=diagnostics["shrinkage_ratios"],
    )

    return cleaned_corr, diagnostics


# ---------------------------------------------------------------------------
# Convenience: clean a pandas DataFrame correlation matrix
# ---------------------------------------------------------------------------

def clean_correlation_dataframe(corr_df, n_observations: int):
    """
    Clean a pandas DataFrame correlation matrix using RMT.

    Returns (cleaned_df, diagnostics).
    """
    import pandas as pd

    raw = corr_df.values.copy()
    cleaned_arr, diag = clean_correlation_matrix(raw, n_observations)
    cleaned_df = pd.DataFrame(cleaned_arr, index=corr_df.index, columns=corr_df.columns)
    return cleaned_df, diag
