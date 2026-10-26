"""
metrics.py - accuracy AND peak-risk AND cost metrics (research gap G6).

Why MAPE/RMSE alone are not enough
---------------------------------
Two models with the same 2.0 % MAPE can be very different for the grid:
model A is +2 % (over) on every heatwave day, model B is -2 % (under) on
every heatwave day. Symmetric metrics call them equal; the grid operator
would fire B. We therefore report, per regime (all / ordinary / heatwave /
critical-peak days):

  Accuracy     MAE (MW), RMSE (MW), MAPE (%), R^2, bias (MW, + = over)
  Peak risk    under-forecast rate (% of days short)
               mean shortfall (MW, averaged over short days)
               P95 shortfall and max shortfall (worst-case planning)
               reserve-breach rate: % of days short by > 3 % of the peak.
                 Rationale: India holds roughly 3-5 % of demand as
                 secondary/tertiary reserves (CERC Ancillary Services
                 Regulations 2022). A shortfall beyond ~3 % exhausts them
                 -> exchange purchases at the cap or load-shedding.
  Cost         stylised Rs crore: shortfall MWh x Rs 10/kWh + surplus MWh x
               Rs 2.5/kWh, over a 4-hour peak window (config.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

RESERVE_BREACH_PCT = 3.0


def forecast_cost_crore(y, y_hat, hours=C.PEAK_WINDOW_HOURS,
                        c_under=C.COST_UNDER_RS_PER_KWH, c_over=C.COST_OVER_RS_PER_KWH):
    """Stylised daily cost of the forecast error in Rs crore (1 crore = 1e7).
    Example: 2,000 MW short for 4 h = 8,000 MWh = 8e6 kWh x Rs 10 = Rs 8 crore."""
    err = np.asarray(y_hat) - np.asarray(y)                  # + = over-forecast
    mwh_short = np.clip(-err, 0, None) * hours
    mwh_over = np.clip(err, 0, None) * hours
    rupees = (mwh_short * 1000 * c_under) + (mwh_over * 1000 * c_over)
    return rupees / 1e7


def regime_metrics(y, y_hat) -> dict:
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    n = len(y)
    if n == 0:
        return {}
    err = y_hat - y
    short = -err[err < 0]
    pct = 100 * err / y
    ss_res, ss_tot = np.sum(err ** 2), np.sum((y - y.mean()) ** 2)
    return {
        "n_days": n,
        "MAE_MW": float(np.mean(np.abs(err))),
        "RMSE_MW": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_pct": float(np.mean(np.abs(pct))),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "bias_MW": float(np.mean(err)),
        "under_rate_pct": float(100 * np.mean(err < 0)),
        "mean_shortfall_MW": float(short.mean()) if len(short) else 0.0,
        "p95_shortfall_MW": float(np.percentile(np.clip(-err, 0, None), 95)),
        "max_shortfall_MW": float(short.max()) if len(short) else 0.0,
        "reserve_breach_pct": float(100 * np.mean(pct < -RESERVE_BREACH_PCT)),
        "cost_crore_per_day": float(np.mean(forecast_cost_crore(y, y_hat))),
    }


def evaluate(df: pd.DataFrame, pred_col: str, y_col: str = "peak_mw") -> pd.DataFrame:
    """Metrics for all days and for each regime. `df` must contain the
    columns national_hw_regime and is_critical_peak."""
    masks = {
        "all": np.ones(len(df), bool),
        "ordinary": (df["national_hw_regime"] == 0).to_numpy(),
        "heatwave": (df["national_hw_regime"] == 1).to_numpy(),
        "critical_peak": (df["is_critical_peak"] == 1).to_numpy(),
        "heatwave_or_critical": ((df["national_hw_regime"] == 1) | (df["is_critical_peak"] == 1)).to_numpy(),
    }
    rows = {k: regime_metrics(df.loc[m, y_col], df.loc[m, pred_col]) for k, m in masks.items()}
    return pd.DataFrame(rows).T


def compare_models(df: pd.DataFrame, pred_cols: dict[str, str]) -> pd.DataFrame:
    """Long table: (model, regime) x metrics."""
    out = []
    for name, col in pred_cols.items():
        m = evaluate(df, col)
        m.insert(0, "model", name)
        out.append(m.rename_axis("regime").reset_index())
    return pd.concat(out, ignore_index=True)
