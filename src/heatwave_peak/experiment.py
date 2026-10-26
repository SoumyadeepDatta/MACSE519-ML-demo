"""
experiment.py - the full, controlled experiment:

  1. Fit all baselines on TRAIN (2019-2023), early-stop on VALID (2024).
  2. Tune HALO's three asymmetry parameters on VALID by grid search.
     Selection rule:
        minimise the stylised grid cost (Rs crore/day) on the 2024
        HEATWAVE-OR-CRITICAL-PEAK days
        subject to ordinary-day MAE <= 105 % of the L2 baseline's.
     i.e. "spend at most 5 % accuracy on normal days to buy safety on hot
     / record days" - the trade-off a grid operator would sign off on, and
     the prototype's two success criteria turned into a selection rule.
     (Revision note: a first run minimised cost over ALL days. Because the
     stylised cost is 4:1 on every day, that rule rewarded shifting EVERY
     forecast upwards - the opposite of heatwave conditioning - and it
     selected the largest base asymmetry a0. The rule was changed to the
     one above; both runs are reported in the README.)
  3. Refit every model on TRAIN+VALID (2019-2024) with the number of
     boosting rounds found in step 1, and forecast the untouched 2025 test
     year ONCE.  (Refitting matters: 2024 holds the record heatwave; a
     model that never saw it would be needlessly handicapped in 2025.)
  4. Walk-forward backtest (2022, 2023, 2024, 2025 test years, expanding
     training window) for robustness of the heatwave-day metrics, which
     rest on few days in any single year.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import metrics as M
from . import models as MD
from .losses import HALOObjective, PrototypeAsymmetricL2

MODEL_ORDER = ["naive_lag7", "lgbm_l2", "lgbm_quantile60", "lgbm_quantile80", "lgbm_prototype",
               "lgbm_linex", "lgbm_halo"]

# Search ranges follow from the LINEX margin mu + a*sigma^2/2: the model's
# residual sigma is ~2 % on ordinary days, so a ~ 0.1-0.5 adds a 0.2-1 %
# margin (gentle), while a ~ 3-8 on severe-heat / record days adds several %.
# A first coarse pass with a <= 1.5 changed almost nothing - too timid.
HALO_GRID = dict(a0=[0.05, 0.1, 0.2, 0.3], a_heat=[0.0, 3.0, 5.0, 8.0, 12.0],
                 a_scarcity=[0.0, 2.0, 4.0, 6.0])
# Ordinary-day accuracy may degrade by at most 5 % versus the L2 baseline
# (the same tolerance as the team's prototype success criterion).
MAX_ORDINARY_MAE_RATIO = 1.05


def load_table() -> tuple[pd.DataFrame, list[str], float]:
    tab = pd.read_parquet(C.PROCESSED_DIR / "model_table.parquet")
    tab, sev_ref = MD.add_loss_side_info(tab)
    feats = F.feature_columns(tab)
    feats = [f for f in feats if f not in ("loss_severity", "loss_scarcity")]
    return tab, feats, sev_ref


def _mask(tab, start=None, end=None):
    m = np.ones(len(tab), bool)
    if start:
        m &= tab.index >= pd.Timestamp(start)
    if end:
        m &= tab.index <= pd.Timestamp(end)
    return m


def fit_all(tr, va, halo_params, rounds=None):
    """Fit every LightGBM variant. With `rounds` (dict) -> no early stopping."""
    r = rounds or {}
    out = {}
    out["lgbm_l2"] = MD.fit_builtin("lgbm_l2", "regression", tr, va, rounds=r.get("lgbm_l2"))
    out["lgbm_quantile60"] = MD.fit_builtin("lgbm_quantile60", "quantile", tr, va,
                                            extra=dict(alpha=0.6), rounds=r.get("lgbm_quantile60"))
    out["lgbm_quantile80"] = MD.fit_builtin("lgbm_quantile80", "quantile", tr, va,
                                            extra=dict(alpha=0.8), rounds=r.get("lgbm_quantile80"))
    out["lgbm_prototype"] = MD.fit_prototype("lgbm_prototype", PrototypeAsymmetricL2(), tr,
                                             rounds=r.get("lgbm_prototype", out["lgbm_l2"].best_iter))
    lin = dict(halo_params, a_heat=0.0, a_scarcity=0.0)
    out["lgbm_linex"] = MD.fit_halo("lgbm_linex", HALOObjective(**lin), tr, va, rounds=r.get("lgbm_linex"))
    out["lgbm_halo"] = MD.fit_halo("lgbm_halo", HALOObjective(**halo_params), tr, va, rounds=r.get("lgbm_halo"))
    return out


def predict_frame(models: dict, s: MD.SplitData) -> pd.DataFrame:
    df = s.frame[["peak_mw", "base_mw", "national_hw_regime", "is_critical_peak",
                  "heat_exposure", "heat_severity_idx", "loss_severity", "loss_scarcity",
                  "train_weight", "target_imputed"]].copy()
    df["pred_naive_lag7"] = s.frame["base_mw"] * (1 + s.frame["peak_lag7_rel"] / 100)
    for name, m in models.items():
        df[f"pred_{name}"] = m.predict_mw(s)
    return df


def _eval_frame(df):
    """Evaluate only on measured (non-imputed) targets."""
    return df[df["target_imputed"] == 0]


def tune_halo(tr, va, verbose=False) -> tuple[pd.DataFrame, dict]:
    l2 = MD.fit_builtin("lgbm_l2", "regression", tr, va)
    base_pred = predict_frame({"lgbm_l2": l2}, va)
    base_m = M.evaluate(_eval_frame(base_pred), "pred_lgbm_l2")
    ord_mae_l2 = base_m.loc["ordinary", "MAE_MW"]
    rows = []
    for a0, ah, asc in itertools.product(*HALO_GRID.values()):
        p = dict(a0=a0, a_heat=ah, a_scarcity=asc)
        m = MD.fit_halo("halo", HALOObjective(**p), tr, va)
        df = _eval_frame(predict_frame({"halo": m}, va))
        ev = M.evaluate(df, "pred_halo")
        rows.append({**p, "best_iter": m.best_iter,
                     "cost_all": ev.loc["all", "cost_crore_per_day"],
                     "cost_hw_or_critical": ev.loc["heatwave_or_critical", "cost_crore_per_day"],
                     "MAPE_all": ev.loc["all", "MAPE_pct"],
                     "MAE_ordinary": ev.loc["ordinary", "MAE_MW"],
                     "MAE_hw": ev.loc["heatwave", "MAE_MW"],
                     "under_rate_hw": ev.loc["heatwave", "under_rate_pct"],
                     "under_rate_critical": ev.loc["critical_peak", "under_rate_pct"],
                     "reserve_breach_all": ev.loc["all", "reserve_breach_pct"]})
        if verbose:
            print(rows[-1])
    res = pd.DataFrame(rows)
    res["ordinary_mae_ratio_vs_l2"] = res["MAE_ordinary"] / ord_mae_l2
    feasible = res[res["ordinary_mae_ratio_vs_l2"] <= MAX_ORDINARY_MAE_RATIO]
    best = (feasible if len(feasible) else res).sort_values("cost_hw_or_critical").iloc[0]
    params = {k: float(best[k]) for k in HALO_GRID}
    return res, params


def run_full_experiment(verbose=False) -> dict:
    tab, feats, sev_ref = load_table()
    train_m = _mask(tab, C.STUDY_START, C.TRAIN_END)
    valid_m = _mask(tab, C.VALID_START, C.VALID_END)
    test_m = _mask(tab, C.TEST_START, C.TEST_END)
    tr, va, te = (MD.make_split(tab, feats, m) for m in (train_m, valid_m, test_m))

    # ---- 2. tune HALO on the validation year -----------------------------
    tuning, halo_params = tune_halo(tr, va, verbose=verbose)
    tuning.to_csv(C.METRICS_DIR / "halo_tuning_grid_valid2024.csv", index=False)

    # ---- 1. validation-stage models (early stopping) ---------------------
    val_models = fit_all(tr, va, halo_params)
    rounds = {k: max(int(m.best_iter), 50) for k, m in val_models.items()}
    val_pred = predict_frame(val_models, va)
    val_pred.to_parquet(C.PROCESSED_DIR / "predictions_valid_2024.parquet")

    # ---- 3. refit on 2019-2024, forecast 2025 ----------------------------
    trva = MD.make_split(tab, feats, train_m | valid_m)
    final = fit_all(trva, None, halo_params, rounds=rounds)
    test_pred = predict_frame(final, te)
    test_pred.to_parquet(C.PROCESSED_DIR / "predictions_test_2025.parquet")
    for name, m in final.items():
        m.booster.save_model(str(C.MODELS_DIR / f"{name}.txt"), num_iteration=m.best_iter)

    cols = {n: f"pred_{n}" for n in MODEL_ORDER}
    val_metrics = M.compare_models(_eval_frame(val_pred), cols)
    test_metrics = M.compare_models(_eval_frame(test_pred), cols)
    val_metrics.to_csv(C.METRICS_DIR / "metrics_valid_2024.csv", index=False)
    test_metrics.to_csv(C.METRICS_DIR / "metrics_test_2025.csv", index=False)

    # ---- 4. walk-forward backtest ----------------------------------------
    bt = []
    for year in (2022, 2023, 2024, 2025):
        trm = _mask(tab, C.STUDY_START, f"{year - 1}-12-31")
        tem = _mask(tab, f"{year}-01-01", f"{year}-12-31")
        s_tr, s_te = MD.make_split(tab, feats, trm), MD.make_split(tab, feats, tem)
        ms = fit_all(s_tr, None, halo_params, rounds=rounds)
        p = predict_frame(ms, s_te)
        p["fold"] = year
        bt.append(p)
    bt = pd.concat(bt)
    bt.to_parquet(C.PROCESSED_DIR / "predictions_backtest_2022_2025.parquet")
    bt_metrics = M.compare_models(_eval_frame(bt), cols)
    bt_metrics.to_csv(C.METRICS_DIR / "metrics_backtest_2022_2025.csv", index=False)

    summary = {"halo_params": halo_params, "boosting_rounds": rounds,
               "severity_reference_p99": sev_ref, "n_features": len(feats),
               "n_train_rows": int((tr.weight > 0).sum()), "n_valid_rows": len(va.z),
               "n_test_rows": len(te.z)}
    (C.METRICS_DIR / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
    if verbose:
        pd.set_option("display.width", 250)
        show = ["model", "regime", "n_days", "MAE_MW", "MAPE_pct", "R2", "bias_MW", "under_rate_pct",
                "mean_shortfall_MW", "max_shortfall_MW", "reserve_breach_pct", "cost_crore_per_day"]
        print(json.dumps(summary, indent=2))
        print("\nTEST 2025\n", test_metrics[show].round(3).to_string())
        print("\nBACKTEST 2022-2025\n", bt_metrics[show].round(3).to_string())
    return dict(tab=tab, features=feats, tuning=tuning, halo_params=halo_params,
                val_pred=val_pred, test_pred=test_pred, backtest=bt,
                test_metrics=test_metrics, val_metrics=val_metrics, bt_metrics=bt_metrics,
                final_models=final, rounds=rounds)
