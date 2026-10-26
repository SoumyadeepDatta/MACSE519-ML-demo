"""
models.py - every model in the comparison, trained on IDENTICAL features,
target, splits and tree hyper-parameters, so that the ONLY difference
between the LightGBM variants is the loss function (a controlled experiment).

Models
------
  naive_lag7       peak of the same weekday last week (what a spreadsheet
                   planner does) - any ML model must beat this
  lgbm_l2          LightGBM, symmetric MSE                   (main baseline)
  lgbm_quantile80  LightGBM, pinball loss at tau = 0.80. Under LINEAR costs
                   c_under = 10, c_over = 2.5, the cost-optimal forecast is
                   the c_u/(c_u+c_o) = 0.8 quantile -> the "textbook"
                   asymmetric approach (gap G4 baseline)
  lgbm_prototype   the team's earlier piecewise asymmetric-L2 objective
                   (binary heatwave flag, constant Hessian) - ablation
  lgbm_linex       HALO with a_heat = a_scarcity = 0, i.e. plain LINEX
                   (ablation: is the HEAT-CONDITIONING what helps?)
  lgbm_halo        the proposed Heat-Adaptive, Load-aware LINEX Objective
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C
from .losses import HALOObjective, PrototypeAsymmetricL2

# Tree hyper-parameters shared by ALL LightGBM variants. Chosen for a small
# (~1,700-row), noisy, high-dimensional daily problem:
#   num_leaves 15 / max_depth 5  -> shallow trees, interactions of <= 5
#                                   features (e.g. heat x weekday x month)
#   min_data_in_leaf 25          -> a leaf must hold ~1 month of days, so
#                                   it cannot memorise a single heatwave
#   feature_fraction 0.6         -> decorrelates trees across the many
#                                   correlated zone-weather columns
#   bagging 0.8 every iteration  -> stochastic boosting, lowers variance
#   lambda_l2 2                  -> shrinks leaf values (ridge on leaves)
#   learning_rate 0.02 + early stopping on the 2024 validation year
BASE_PARAMS = dict(
    learning_rate=0.02, num_leaves=15, max_depth=5, min_data_in_leaf=25,
    feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=2.0, min_gain_to_split=0.0, verbosity=-1, seed=C.RANDOM_SEED,
    deterministic=True, force_row_wise=True, num_threads=4,
)
MAX_ROUNDS = 4000
EARLY_STOP = 300


@dataclass
class SplitData:
    """Arrays for one split, all aligned by row."""
    X: pd.DataFrame
    z: np.ndarray          # target: % above weekly base
    base: np.ndarray       # MW baseline -> MW = base * (1 + z/100)
    weight: np.ndarray     # training weight (0 = excluded row)
    severity: np.ndarray   # s_i in [0, 1]
    scarcity: np.ndarray   # g_i in [0, 1]
    hw_flag: np.ndarray    # national heatwave regime (0/1)
    frame: pd.DataFrame    # original rows (for evaluation)


def add_loss_side_info(tab: pd.DataFrame, severity_ref: float | None = None) -> tuple[pd.DataFrame, float]:
    """
    Add the two conditioning signals HALO needs.

    severity  = heat_severity_idx / (its training 99th percentile), clipped
                to [0, 1]. Normalising by the 99th percentile maps the
                ~top-1 % of heat days (e.g. late May 2024) to 1.0.
    scarcity  = clip((peak / record_365 - 0.95) / 0.05, 0, 1) where
                record_365 is the max peak of the PREVIOUS 365 days.
                Example: record 237 GW -> a 230 GW day (97 %) gets g = 0.4,
                a new record 246 GW day gets g = 1.
    """
    tab = tab.copy()
    if severity_ref is None:
        tr = tab[(tab["split"] == "train") & (tab["train_weight"] > 0)]
        severity_ref = float(np.nanpercentile(tr["heat_severity_idx"], 99))
    tab["loss_severity"] = (tab["heat_severity_idx"] / severity_ref).clip(0, 1).fillna(0)
    record = tab["peak_mw"].shift(1).rolling(365, min_periods=90).max()
    tab["loss_scarcity"] = ((tab["peak_mw"] / record - 0.95) / 0.05).clip(0, 1).fillna(0)
    return tab, severity_ref


def make_split(tab: pd.DataFrame, features: list[str], mask) -> SplitData:
    d = tab.loc[mask]
    d = d[d["base_mw"].notna() & d["target_z"].notna()]
    return SplitData(
        X=d[features], z=d["target_z"].to_numpy(), base=d["base_mw"].to_numpy(),
        weight=d["train_weight"].to_numpy(), severity=d["loss_severity"].to_numpy(),
        scarcity=d["loss_scarcity"].to_numpy(), hw_flag=d["national_hw_regime"].to_numpy(),
        frame=d,
    )


def to_mw(base: np.ndarray, z_hat: np.ndarray) -> np.ndarray:
    return base * (1 + z_hat / 100.0)


class Trained:
    """A fitted LightGBM booster plus the target offset it was trained on."""

    def __init__(self, name, booster, offset, best_iter):
        self.name, self.booster, self.offset, self.best_iter = name, booster, offset, best_iter

    def predict_z(self, X):
        return self.booster.predict(X, num_iteration=self.best_iter) + self.offset

    def predict_mw(self, s: SplitData):
        return to_mw(s.base, self.predict_z(s.X))


def _offset(tr: SplitData) -> float:
    """Weighted mean target. Custom objectives start boosting from 0, so we
    centre the target ourselves (built-in objectives do this internally via
    boost_from_average); applying it to every model keeps them comparable."""
    return float(np.average(tr.z, weights=np.maximum(tr.weight, 1e-12)))


def _scale(s: SplitData, ref: float) -> np.ndarray:
    return s.base / ref


def fit_builtin(name: str, objective: str, tr: SplitData, va: SplitData | None,
                extra: dict | None = None, rounds: int | None = None) -> Trained:
    off = _offset(tr)
    params = dict(BASE_PARAMS, objective=objective, **(extra or {}))
    scale_ref = tr.base[tr.weight > 0].mean()
    w_tr = tr.weight * _scale(tr, scale_ref)
    dtr = lgb.Dataset(tr.X, tr.z - off, weight=w_tr, free_raw_data=False)
    if rounds is None and va is not None:
        dva = lgb.Dataset(va.X, va.z - off, weight=va.weight * _scale(va, scale_ref), reference=dtr)
        b = lgb.train(params, dtr, MAX_ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        return Trained(name, b, off, b.best_iteration)
    b = lgb.train(params, dtr, rounds or 1000)
    return Trained(name, b, off, rounds or 1000)


def fit_halo(name: str, halo: HALOObjective, tr: SplitData, va: SplitData | None,
             rounds: int | None = None) -> Trained:
    off = _offset(tr)
    params = dict(BASE_PARAMS, objective=halo.objective, metric="None")
    scale_ref = tr.base[tr.weight > 0].mean()
    dtr = lgb.Dataset(tr.X, tr.z - off, free_raw_data=False)
    halo.register(dtr, tr.severity, tr.scarcity, _scale(tr, scale_ref), tr.weight)
    if rounds is None and va is not None:
        dva = lgb.Dataset(va.X, va.z - off, reference=dtr, free_raw_data=False)
        halo.register(dva, va.severity, va.scarcity, _scale(va, scale_ref), va.weight)
        b = lgb.train(params, dtr, MAX_ROUNDS, valid_sets=[dva], feval=halo.feval,
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        return Trained(name, b, off, b.best_iteration)
    b = lgb.train(params, dtr, rounds or 1000)
    return Trained(name, b, off, rounds or 1000)


def fit_prototype(name: str, proto: PrototypeAsymmetricL2, tr: SplitData, rounds: int) -> Trained:
    """No validation-based early stopping (the prototype has no matching
    metric); uses the same number of rounds as the L2 baseline."""
    off = _offset(tr)
    params = dict(BASE_PARAMS, objective=proto.objective, metric="None")
    scale_ref = tr.base[tr.weight > 0].mean()
    dtr = lgb.Dataset(tr.X, tr.z - off, free_raw_data=False)
    proto.register(dtr, tr.hw_flag, tr.weight * _scale(tr, scale_ref))
    b = lgb.train(params, dtr, rounds)
    return Trained(name, b, off, rounds)
