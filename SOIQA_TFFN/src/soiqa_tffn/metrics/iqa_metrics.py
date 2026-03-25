from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, pearsonr, spearmanr


def _safe_corr(fn, targets: np.ndarray, preds: np.ndarray) -> float:
    if len(targets) < 2:
        return float("nan")
    if np.allclose(targets, targets[0]) or np.allclose(preds, preds[0]):
        return float("nan")
    try:
        return float(fn(targets, preds)[0])
    except Exception:
        return float("nan")


def _safe_kendall(targets: np.ndarray, preds: np.ndarray) -> float:
    if len(targets) < 2:
        return float("nan")
    if np.allclose(targets, targets[0]) or np.allclose(preds, preds[0]):
        return float("nan")
    try:
        return float(kendalltau(targets, preds)[0])
    except Exception:
        return float("nan")


def _logistic4(x: np.ndarray, beta1: float, beta2: float, beta3: float, beta4: float) -> np.ndarray:
    return beta2 + (beta1 - beta2) / (1.0 + np.exp(-(x - beta3) / np.maximum(np.abs(beta4), 1e-8)))


def fit_logistic_mapping(targets: np.ndarray, preds: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    if len(targets) < 4:
        return preds.copy()
    beta0 = [float(targets.max()), float(targets.min()), float(preds.mean()), float(np.std(preds) + 1e-6)]
    try:
        popt, _ = curve_fit(_logistic4, preds, targets, p0=beta0, maxfev=20000)
        fitted = _logistic4(preds, *popt)
        if np.any(np.isnan(fitted)) or np.any(np.isinf(fitted)):
            return preds.copy()
        return fitted.astype(np.float64)
    except Exception:
        return preds.copy()


def compute_iqa_metrics(
    targets: list[float] | np.ndarray,
    preds: list[float] | np.ndarray,
    apply_logistic_fit: bool = False,
) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    if len(targets) < 2:
        rmse = float(np.sqrt(np.mean((targets - preds) ** 2))) if len(targets) > 0 else float("nan")
        return {
            "PLCC": float("nan"),
            "SRCC": float("nan"),
            "KRCC": float("nan"),
            "RMSE": rmse,
            "PLCC_raw": float("nan"),
            "RMSE_raw": rmse,
        }

    preds_for_plcc = fit_logistic_mapping(targets, preds) if apply_logistic_fit else preds
    plcc = _safe_corr(pearsonr, targets, preds_for_plcc)
    srcc = _safe_corr(spearmanr, targets, preds)
    krcc = _safe_kendall(targets, preds)
    rmse = float(np.sqrt(np.mean((targets - preds_for_plcc) ** 2)))
    plcc_raw = _safe_corr(pearsonr, targets, preds)
    rmse_raw = float(np.sqrt(np.mean((targets - preds) ** 2)))
    return {
        "PLCC": plcc,
        "SRCC": srcc,
        "KRCC": krcc,
        "RMSE": rmse,
        "PLCC_raw": plcc_raw,
        "RMSE_raw": rmse_raw,
    }
