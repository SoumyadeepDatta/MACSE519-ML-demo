"""
evidence.py - statistical tests and ablations that turn "HALO looks better"
into claims that can be defended against a reviewer.

A novelty claim needs three things:
  1. The effect is REAL, not noise  -> paired significance tests
     (moving-block bootstrap, Diebold-Mariano, McNemar).
  2. The effect comes from the COMPONENT we claim is new -> ablations
     (remove one ingredient, keep everything else identical).
  3. The comparison is FAIR -> compare at equal accuracy (iso-accuracy
     frontier), not "our biased model vs an unbiased baseline".

All experiments here use the walk-forward backtest (test years 2022-2025,
expanding training window) with the boosting rounds and HALO parameters
already fixed by the main experiment. No parameter is re-tuned on the
backtest. Folds 2022, 2023 and 2025 are fully out-of-sample for HALO's
parameters (tuned on 2024); results are also reported on those folds only.
"""

from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

from . import clustering as CL
from . import config as C
from . import features as F
from . import heatwave as HW
from . import load_processing as LP
from . import metrics as M
from . import models as MD
from .losses import HALOObjective

FOLDS = (2022, 2023, 2024, 2025)
OOS_FOLDS = (2022, 2023, 2025)          # not used to tune HALO


# ============================================================================
# 1. Statistics evaluated on a prediction frame
# ============================================================================
def _hc(df):
    """Heat-or-critical-peak mask: the days the research question is about."""
    return ((df["national_hw_regime"] == 1) | (df["is_critical_peak"] == 1)).to_numpy()


def stat_under_rate_hc(df, col):
    m = _hc(df)
    return 100 * np.mean(df.loc[m, col].to_numpy() < df.loc[m, "peak_mw"].to_numpy())


def stat_breach_hc(df, col):
    m = _hc(df)
    pct = 100 * (df.loc[m, col] - df.loc[m, "peak_mw"]) / df.loc[m, "peak_mw"]
    return 100 * np.mean(pct < -M.RESERVE_BREACH_PCT)


def stat_expected_shortfall_hc(df, col):
    """Average MW short per heat/critical day (0 on days with no shortfall).
    Unlike 'mean shortfall given short', this cannot improve just by being
    short on fewer days by tiny amounts."""
    m = _hc(df)
    return float(np.clip(df.loc[m, "peak_mw"] - df.loc[m, col], 0, None).mean())


def stat_expected_shortfall_hw(df, col):
    """Expected shortfall on national heatwave-regime days only - where the
    HEAT term of HALO (as opposed to the scarcity term) should act."""
    m = (df["national_hw_regime"] == 1).to_numpy()
    return float(np.clip(df.loc[m, "peak_mw"] - df.loc[m, col], 0, None).mean())


def stat_cost_hc(df, col):
    m = _hc(df)
    return float(M.forecast_cost_crore(df.loc[m, "peak_mw"], df.loc[m, col]).mean())


def stat_cost_all(df, col):
    return float(M.forecast_cost_crore(df["peak_mw"], df[col]).mean())


def stat_mae_ordinary(df, col):
    m = ~_hc(df)
    return float(np.abs(df.loc[m, col] - df.loc[m, "peak_mw"]).mean())


def stat_mape_all(df, col):
    return float(100 * np.mean(np.abs(df[col] - df["peak_mw"]) / df["peak_mw"]))


STATS = {
    "under_rate_hc_pct": stat_under_rate_hc,
    "reserve_breach_hc_pct": stat_breach_hc,
    "expected_shortfall_hc_MW": stat_expected_shortfall_hc,
    "expected_shortfall_heatwave_MW": stat_expected_shortfall_hw,
    "cost_hc_crore": stat_cost_hc,
    "cost_all_crore": stat_cost_all,
    "MAE_ordinary_MW": stat_mae_ordinary,
    "MAPE_all_pct": stat_mape_all,
}


# ============================================================================
# 2. Significance tests
# ============================================================================
def block_bootstrap_diff(df, col_a, col_b, stat_fn, block=7, n_boot=2000, seed=C.RANDOM_SEED):
    """
    Paired MOVING-BLOCK bootstrap of  stat(b) - stat(a).

    Why blocks? Forecast errors are autocorrelated: a model that is short on
    Monday of a heatwave is likely short on Tuesday too. Resampling single
    days would pretend we have more independent evidence than we do and
    give over-optimistic p-values. Resampling whole weeks (block = 7)
    keeps that dependence (Kunsch 1989).

    Paired = both models are evaluated on the SAME resampled days, so
    day-to-day difficulty cancels out.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    point = stat_fn(df, col_b) - stat_fn(df, col_a)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        sub = df.iloc[idx]
        diffs[i] = stat_fn(sub, col_b) - stat_fn(sub, col_a)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return {"diff": point, "ci_low": lo, "ci_high": hi, "p_value": min(p, 1.0)}


def diebold_mariano(loss_a, loss_b, max_lag=7):
    """
    Diebold-Mariano (1995) test of equal predictive accuracy under a given
    loss, with a Newey-West (HAC) variance using `max_lag` days, plus the
    Harvey-Leybourne-Newbold (1997) small-sample correction.
    Positive statistic = model b has LOWER loss than a.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    dc = d - d.mean()
    gamma0 = np.dot(dc, dc) / n
    var = gamma0 + 2 * sum((1 - k / (max_lag + 1)) * np.dot(dc[k:], dc[:-k]) / n
                           for k in range(1, max_lag + 1))
    dm = d.mean() / np.sqrt(var / n)
    h = 1
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return {"DM_stat": float(stat), "p_value": float(p), "mean_loss_diff": float(d.mean())}


def mcnemar_under(df, col_a, col_b, mask=None):
    """
    Exact McNemar test on paired under-forecast indicators: on how many
    heat/critical days was A short while B was not, and vice versa?
    Under H0 the two discordant counts are equally likely (binomial 0.5).
    """
    m = _hc(df) if mask is None else mask
    ua = df.loc[m, col_a].to_numpy() < df.loc[m, "peak_mw"].to_numpy()
    ub = df.loc[m, col_b].to_numpy() < df.loc[m, "peak_mw"].to_numpy()
    a_only, b_only = int(np.sum(ua & ~ub)), int(np.sum(~ua & ub))
    p = stats.binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return {"A_short_only": a_only, "B_short_only": b_only, "p_value": float(p)}


def significance_table(df, col_a, col_b, label_a, label_b, n_boot=2000):
    rows = []
    for name, fn in STATS.items():
        r = block_bootstrap_diff(df, col_a, col_b, fn, n_boot=n_boot)
        rows.append({"comparison": f"{label_b} - {label_a}", "statistic": name,
                     label_a: fn(df, col_a), label_b: fn(df, col_b), **r})
    out = pd.DataFrame(rows)
    out["significant_5pct"] = out["p_value"] < 0.05
    return out


# ============================================================================
# 3. Re-usable backtest runner for ablations
# ============================================================================
def load_rounds_and_params():
    s = json.loads((C.METRICS_DIR / "experiment_summary.json").read_text())
    return s["boosting_rounds"], s["halo_params"]


def _frame(s: MD.SplitData) -> pd.DataFrame:
    return s.frame[["peak_mw", "base_mw", "national_hw_regime", "is_critical_peak",
                    "target_imputed"]].copy()


def backtest(tab: pd.DataFrame, feats: list[str], specs: dict, folds=FOLDS) -> pd.DataFrame:
    """
    specs: {name: callable(train_split) -> object with .predict_mw(split)}
    Returns concatenated fold predictions with columns pred_<name>.
    """
    out = []
    for y in folds:
        trm = (tab.index <= pd.Timestamp(f"{y - 1}-12-31"))
        tem = (tab.index >= pd.Timestamp(f"{y}-01-01")) & (tab.index <= pd.Timestamp(f"{y}-12-31"))
        tr, te = MD.make_split(tab, feats, trm), MD.make_split(tab, feats, tem)
        fr = _frame(te)
        for name, fit in specs.items():
            fr[f"pred_{name}"] = fit(tr).predict_mw(te)
        fr["fold"] = y
        out.append(fr)
    bt = pd.concat(out)
    return bt[bt["target_imputed"] == 0]


def spec_l2(rounds):
    return lambda tr: MD.fit_builtin("l2", "regression", tr, None, rounds=rounds)


def spec_quantile(alpha, rounds):
    return lambda tr: MD.fit_builtin(f"q{alpha}", "quantile", tr, None, extra=dict(alpha=alpha), rounds=rounds)


def spec_halo(params, rounds):
    return lambda tr: MD.fit_halo("halo", HALOObjective(**params), tr, None, rounds=rounds)


def summarise(bt: pd.DataFrame, names, folds=None) -> pd.DataFrame:
    d = bt if folds is None else bt[bt["fold"].isin(folds)]
    return pd.DataFrame({n: {k: fn(d, f"pred_{n}") for k, fn in STATS.items()} for n in names}).T


# ============================================================================
# 4. Feature-set and spatial-representation ablations
# ============================================================================
def _match(feats, patterns):
    return [f for f in feats if any(p in f for p in patterns)]


def feature_ablation_sets(feats: list[str]) -> dict[str, list[str]]:
    """
    G3  'explicit heatwave regime' = IMD/EHF heatwave indicators and the
        demand-weighted exposure/severity indices. Removing them leaves the
        model with weather as plain inputs only (temperature, humidity ...),
        which is what gap G3 says most studies do.
    G7a 'heat accumulation / thermal lag' = run lengths, EWM heat load,
        3-day cooling degree-days, EHF (3-day), warm nights.
    G7b 'humidity coupling' = heat index, apparent temperature, dew point, RH.
    """
    regime = _match(feats, ["heat_exposure", "imd_hw_exposure", "severe_exposure", "heat_severity",
                            "warm_night_exposure", "hot40_exposure", "hw_area_frac", "__ehf"])
    accum = _match(feats, ["run_length", "heat_accum_ewm", "cdd24_3d", "__ehf", "warm_night"])
    humid = _match(feats, ["heat_index", "apparent", "dewpoint", "rh_"])
    return {
        "full": feats,
        "no_regime_features (G3)": [f for f in feats if f not in set(regime)],
        "no_accumulation_features (G7a)": [f for f in feats if f not in set(accum)],
        "no_humidity_coupling (G7b)": [f for f in feats if f not in set(humid)],
    }


def table_with_spatial_mode(mode: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Rebuild the model table with a different spatial representation:
      'zones'    - our 8 data-driven thermal-demand zones (the proposal)
      'grid'     - the 5 official Grid-India regions (NR/WR/SR/ER/NER)
      'national' - only the demand-weighted national average (no regions)
    Everything else (load history, calendar, national heat indices) is the
    same, so the comparison isolates the value of the clustering step.
    """
    sw = pd.read_parquet(C.INTERIM_DIR / "state_daily_weather_1991_2025.parquet")
    sw = sw[(sw["date"] >= "2018-12-01") & (sw["date"] <= C.STUDY_END)]
    w = pd.read_csv(C.PROCESSED_DIR / "state_demand_weights.csv", index_col=0)["demand_weight"]
    zones = pd.read_csv(C.PROCESSED_DIR / "state_signatures_and_zones.csv", index_col=0)
    national_heat = HW.national_heat_exposure(sw, w)
    nat_w = CL.zone_daily_features(sw, pd.Series(0, index=zones.index), w, {0: "IN"})
    nat_w = nat_w.drop(columns=[c for c in nat_w.columns if c.endswith("heat_exposure")])
    if mode == "zones":
        zone_of = zones["zone_id"]
        names = zones.drop_duplicates("zone_id").set_index("zone_id")["zone"].to_dict()
        zone_wide = CL.zone_daily_features(sw, zone_of, w, names)
    elif mode == "grid":
        codes = {r: i for i, r in enumerate(sorted(zones["grid_region"].unique()))}
        zone_of = zones["grid_region"].map(codes)
        zone_wide = CL.zone_daily_features(sw, zone_of, w, {i: f"G_{r}" for r, i in codes.items()})
    elif mode == "national":
        zone_wide = pd.DataFrame(index=nat_w.index)
    else:
        raise ValueError(mode)
    daily = pd.read_parquet(C.PROCESSED_DIR / "daily_peak_national.parquet")
    tab = F.build_model_table(daily, national_heat, zone_wide, nat_w, LP.read_psp_daily())
    tab, _ = MD.add_loss_side_info(tab)
    feats = [f for f in F.feature_columns(tab) if f not in ("loss_severity", "loss_scarcity")]
    return tab, feats


# ============================================================================
# 5. No-target-transform ablation (distribution shift, G11)
# ============================================================================
class _RawMW:
    """LightGBM-L2 trained directly on peak MW (no '% above last week'
    transform). base_mw is given as a feature so the model at least knows
    the current level."""

    def __init__(self, booster, feats):
        self.booster, self.feats = booster, feats

    def predict_mw(self, s):
        return self.booster.predict(s.X.assign(base_mw=s.frame["base_mw"].to_numpy()))


def spec_raw_mw(rounds):
    def fit(tr):
        X = tr.X.assign(base_mw=tr.frame["base_mw"].to_numpy())
        y = tr.frame["peak_mw"].to_numpy()
        params = dict(MD.BASE_PARAMS, objective="regression")
        b = lgb.train(params, lgb.Dataset(X, y, weight=tr.weight), rounds)
        return _RawMW(b, list(X.columns))
    return fit


def beyond_training_mask(bt: pd.DataFrame, tab: pd.DataFrame, frac: float = 0.98) -> np.ndarray:
    """True for test days whose peak is >= `frac` x the highest peak in that
    fold's TRAINING years - demand levels at or beyond anything the model
    has seen (e.g. 2024's 247 GW versus a 2019-2023 maximum of 237 GW)."""
    m = np.zeros(len(bt), bool)
    for y in bt["fold"].unique():
        train_max = tab.loc[:f"{y - 1}-12-31", "peak_mw"].max()
        sel = (bt["fold"] == y).to_numpy()
        m[sel] = bt.loc[sel, "peak_mw"].to_numpy() >= frac * train_max
    return m


def level_shift_table(bt: pd.DataFrame, tab: pd.DataFrame, names) -> pd.DataFrame:
    """Bias / under-forecast rate / MAPE on beyond-training-level days."""
    m = beyond_training_mask(bt, tab)
    d = bt[m]
    rows = {}
    for n in names:
        e = d[f"pred_{n}"] - d["peak_mw"]
        rows[n] = {"n_days": len(d), "bias_MW": e.mean(), "under_rate_pct": 100 * (e < 0).mean(),
                   "MAPE_pct": 100 * (e.abs() / d["peak_mw"]).mean(),
                   "worst_shortfall_MW": float((-e).clip(lower=0).max())}
    return pd.DataFrame(rows).T


def timed(fn, *a, **k):
    t = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t


# ============================================================================
# 6. The experiments (each saves a CSV in reports/metrics/evidence_*.csv)
# ============================================================================
def _save(df, name):
    df.to_csv(C.METRICS_DIR / f"evidence_{name}.csv")
    return df


def _main_table():
    from . import experiment as EX
    return EX.load_table()


def exp_significance(n_boot=2000) -> pd.DataFrame:
    """E1 - is HALO's improvement over L2 (and over pinball-0.6) real?"""
    bt = pd.read_parquet(C.PROCESSED_DIR / "predictions_backtest_2022_2025.parquet")
    te = pd.read_parquet(C.PROCESSED_DIR / "predictions_test_2025.parquet")
    bt, te = bt[bt["target_imputed"] == 0], te[te["target_imputed"] == 0]
    sets = {"backtest 2022-2025": bt, "backtest OOS folds (2022, 2023, 2025)": bt[bt["fold"].isin(OOS_FOLDS)],
            "test 2025": te}
    rows = []
    for set_name, d in sets.items():
        d = d.sort_index()
        for base in ("lgbm_l2", "lgbm_quantile60"):
            t = significance_table(d, f"pred_{base}", "pred_lgbm_halo", base, "lgbm_halo", n_boot=n_boot)
            t = t.rename(columns={base: "baseline_value", "lgbm_halo": "halo_value"})
            t.insert(0, "baseline", base)
            t.insert(0, "eval_set", set_name)
            dm = diebold_mariano(M.forecast_cost_crore(d["peak_mw"], d[f"pred_{base}"]),
                                 M.forecast_cost_crore(d["peak_mw"], d["pred_lgbm_halo"]))
            mc = mcnemar_under(d, f"pred_{base}", "pred_lgbm_halo")
            t["DM_cost_stat"], t["DM_cost_p"] = dm["DM_stat"], dm["p_value"]
            t["McNemar_base_short_only"] = mc["A_short_only"]
            t["McNemar_halo_short_only"] = mc["B_short_only"]
            t["McNemar_p"] = mc["p_value"]
            rows.append(t)
    return _save(pd.concat(rows, ignore_index=True), "significance")


def exp_frontier() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    E2 - the CORE claim (G3 + G4 + G5): at the SAME ordinary-day accuracy,
    does heat/scarcity-CONDITIONED asymmetry give more safety on hot and
    record days than UNIFORM asymmetry (plain LINEX with one constant a,
    or pinball/quantile loss)?  We trace each family's accuracy-safety
    curve by sweeping its single asymmetry knob, then ask whether HALO
    lies below (safer than) the uniform curves.
    """
    tab, feats, _ = _main_table()
    rounds, hp = load_rounds_and_params()
    r_halo, r_lin, r_q, r_l2 = rounds["lgbm_halo"], rounds["lgbm_linex"], 1000, rounds["lgbm_l2"]
    specs, family = {"L2": spec_l2(r_l2)}, {"L2": "symmetric"}
    for a in (0.2, 0.4, 0.6, 0.8, 1.0, 1.5):
        specs[f"LINEX a={a}"] = spec_halo(dict(a0=a, a_heat=0, a_scarcity=0), r_lin)
        family[f"LINEX a={a}"] = "uniform LINEX"
    for q in (0.55, 0.6, 0.65, 0.7, 0.75):
        specs[f"Quantile {q}"] = spec_quantile(q, r_q)
        family[f"Quantile {q}"] = "uniform pinball"
    halo_grid = {"HALO (selected)": hp,
                 "HALO 0.05/12/6": dict(a0=0.05, a_heat=12, a_scarcity=6),
                 "HALO 0.1/8/4": dict(a0=0.1, a_heat=8, a_scarcity=4),
                 "HALO 0.3/12/6": dict(a0=0.3, a_heat=12, a_scarcity=6),
                 "HALO heat-only 0.2/12/0": dict(hp, a_scarcity=0.0),
                 "HALO scarcity-only 0.2/0/6": dict(hp, a_heat=0.0)}
    for k, p in halo_grid.items():
        specs[k] = spec_halo(p, r_halo)
        family[k] = "HALO ablation" if "only" in k else "HALO (conditioned)"
    bt = backtest(tab, feats, specs)
    bt.to_parquet(C.PROCESSED_DIR / "evidence_frontier_backtest.parquet")
    res = summarise(bt, specs.keys())
    res.insert(0, "family", pd.Series(family))
    res_oos = summarise(bt, specs.keys(), folds=OOS_FOLDS)
    res_oos.insert(0, "family", pd.Series(family))
    _save(res_oos, "frontier_oos_folds")
    return _save(res, "frontier"), bt


def iso_accuracy_gain(res: pd.DataFrame, x="MAE_ordinary_MW",
                      ys=("cost_hc_crore", "expected_shortfall_hc_MW", "under_rate_hc_pct",
                          "reserve_breach_hc_pct"), tag="") -> pd.DataFrame:
    """
    For each HALO point, linearly interpolate each UNIFORM family's curve at
    HALO's ordinary-day MAE and report how much safer HALO is there.
    Negative 'gain' = HALO has lower risk than uniform asymmetry at equal
    ordinary-day accuracy.
    """
    out = []
    for name, r in res[res["family"].str.startswith("HALO")].iterrows():
        for fam in ("uniform LINEX", "uniform pinball"):
            curve = res[res["family"] == fam].sort_values(x)
            if not (curve[x].min() <= r[x] <= curve[x].max()):
                continue
            row = {"halo_variant": name, "vs": fam, x: r[x]}
            for y in ys:
                uni = float(np.interp(r[x], curve[x], curve[y]))
                row[f"{y} (HALO)"], row[f"{y} (uniform @ same MAE)"] = r[y], uni
                row[f"{y} gain"] = r[y] - uni
            out.append(row)
    return _save(pd.DataFrame(out), f"iso_accuracy{tag}")


def exp_feature_ablation() -> pd.DataFrame:
    """E3 - value of explicit heatwave-regime features (G3) and of heat
    accumulation / humidity features (G7), for both L2 and HALO."""
    tab, feats, _ = _main_table()
    rounds, hp = load_rounds_and_params()
    rows, preds = [], None
    for set_name, fs in feature_ablation_sets(feats).items():
        bt = backtest(tab, fs, {"L2": spec_l2(rounds["lgbm_l2"]), "HALO": spec_halo(hp, rounds["lgbm_halo"])})
        s = summarise(bt, ["L2", "HALO"])
        s.insert(0, "n_features", len(fs))
        s.insert(0, "feature_set", set_name)
        rows.append(s.rename_axis("model").reset_index())
        tag = set_name.split(" ")[0]
        bt = bt.rename(columns={"pred_L2": f"pred_L2|{tag}", "pred_HALO": f"pred_HALO|{tag}"})
        preds = bt if preds is None else preds.join(bt[[f"pred_L2|{tag}", f"pred_HALO|{tag}"]])
    preds.to_parquet(C.PROCESSED_DIR / "evidence_feature_ablation_backtest.parquet")
    return _save(pd.concat(rows, ignore_index=True), "feature_ablation")


def ablation_bootstrap(n_boot=2000) -> pd.DataFrame:
    """Is the drop from removing accumulation / regime / humidity features
    significant? Paired block bootstrap of  stat(ablated) - stat(full)."""
    bt = pd.read_parquet(C.PROCESSED_DIR / "evidence_feature_ablation_backtest.parquet").sort_index()
    rows = []
    for model in ("L2", "HALO"):
        for tag in ("no_regime_features", "no_accumulation_features", "no_humidity_coupling"):
            for st in ("expected_shortfall_hc_MW", "cost_hc_crore", "MAE_ordinary_MW"):
                r = block_bootstrap_diff(bt, f"pred_{model}|full", f"pred_{model}|{tag}", STATS[st], n_boot=n_boot)
                rows.append({"model": model, "ablation": tag, "statistic": st,
                             "full": STATS[st](bt, f"pred_{model}|full"),
                             "ablated": STATS[st](bt, f"pred_{model}|{tag}"), **r})
    return _save(pd.DataFrame(rows), "feature_ablation_bootstrap")


def frontier_bootstrap(halo="HALO (selected)", n_boot=1000, block=7, seed=C.RANDOM_SEED,
                       folds=None) -> pd.DataFrame:
    """
    Confidence interval for the ISO-ACCURACY gain: in every bootstrap
    resample we recompute each uniform family's whole accuracy-safety curve
    AND the HALO point on the same resampled days, interpolate the curve at
    HALO's ordinary-day MAE, and record HALO minus uniform. If the 95 % CI
    is entirely below 0, conditioning beats uniform asymmetry at equal
    accuracy - the central novelty claim.
    """
    bt = pd.read_parquet(C.PROCESSED_DIR / "evidence_frontier_backtest.parquet").sort_index()
    if folds is not None:
        bt = bt[bt["fold"].isin(folds)]
    fam = {"uniform LINEX": [c for c in bt if c.startswith("pred_LINEX")],
           "uniform pinball": [c for c in bt if c.startswith("pred_Quantile")]}
    ys = ("cost_hc_crore", "expected_shortfall_hc_MW", "under_rate_hc_pct")
    x = "MAE_ordinary_MW"

    def gains(d):
        hx = STATS[x](d, f"pred_{halo}")
        out = {}
        for fname, cols in fam.items():
            xs = np.array([STATS[x](d, c) for c in cols])
            order = np.argsort(xs)
            for y in ys:
                if not (xs.min() <= hx <= xs.max()):      # never extrapolate a curve
                    out[(fname, y)] = np.nan
                    continue
                yv = np.array([STATS[y](d, c) for c in cols])[order]
                out[(fname, y)] = STATS[y](d, f"pred_{halo}") - np.interp(hx, xs[order], yv)
        return out

    point = gains(bt)
    rng = np.random.default_rng(seed)
    n = len(bt)
    nb = int(np.ceil(n / block))
    draws = {k: [] for k in point}
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        g = gains(bt.iloc[idx])
        for k, v in g.items():
            draws[k].append(v)
    rows = []
    for (fname, y), v in point.items():
        if np.isnan(v):
            continue
        d = np.array(draws[(fname, y)])
        d = d[~np.isnan(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append({"halo": halo, "vs": fname, "statistic": y, "gain_at_equal_accuracy": v,
                     "ci_low": lo, "ci_high": hi, "p_value": min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))})
    return pd.DataFrame(rows)


def exp_component_contrast(n_boot=2000) -> pd.DataFrame:
    """
    Does the HEAT term add anything on top of the SCARCITY term?
    Paired bootstrap of  full HALO - scarcity-only HALO  (same a0, same
    scarcity strength; the only difference is a_heat = 12 vs 0).
    """
    bt = pd.read_parquet(C.PROCESSED_DIR / "evidence_frontier_backtest.parquet").sort_index()
    rows = []
    for label, d in (("all folds 2022-2025", bt), ("OOS folds 2022/2023/2025", bt[bt["fold"].isin(OOS_FOLDS)])):
        for st in ("expected_shortfall_heatwave_MW", "expected_shortfall_hc_MW", "cost_hc_crore",
                   "under_rate_hc_pct", "MAE_ordinary_MW"):
            r = block_bootstrap_diff(d, "pred_HALO scarcity-only 0.2/0/6", "pred_HALO (selected)",
                                     STATS[st], n_boot=n_boot)
            rows.append({"eval_set": label, "statistic": st,
                         "scarcity_only": STATS[st](d, "pred_HALO scarcity-only 0.2/0/6"),
                         "full_HALO": STATS[st](d, "pred_HALO (selected)"), **r})
    return _save(pd.DataFrame(rows), "component_contrast")


def exp_frontier_bootstrap(n_boot=1000) -> pd.DataFrame:
    """Iso-accuracy CIs for the selected HALO and its two one-component
    ablations, on all backtest folds and on the out-of-sample folds only."""
    out = []
    for halo in ("HALO (selected)", "HALO heat-only 0.2/12/0", "HALO scarcity-only 0.2/0/6"):
        for label, folds in (("all folds 2022-2025", None), ("OOS folds 2022/2023/2025", OOS_FOLDS)):
            r = frontier_bootstrap(halo, n_boot=n_boot, folds=folds)
            r.insert(0, "eval_set", label)
            out.append(r)
    return _save(pd.concat(out, ignore_index=True), "frontier_bootstrap")


def exp_spatial_ablation() -> pd.DataFrame:
    """E4 - data-driven zones vs official grid regions vs national-only."""
    rounds, hp = load_rounds_and_params()
    rows = []
    for mode in ("zones", "grid", "national"):
        tab, feats = table_with_spatial_mode(mode)
        bt = backtest(tab, feats, {"L2": spec_l2(rounds["lgbm_l2"]), "HALO": spec_halo(hp, rounds["lgbm_halo"])})
        s = summarise(bt, ["L2", "HALO"])
        s.insert(0, "n_features", len(feats))
        s.insert(0, "spatial_representation", mode)
        rows.append(s.rename_axis("model").reset_index())
    return _save(pd.concat(rows, ignore_index=True), "spatial_ablation")


def exp_level_shift() -> pd.DataFrame:
    """E5 - G11: what happens on days with demand beyond the training range,
    with and without the '% above last week' target transform?"""
    tab, feats, _ = _main_table()
    rounds, hp = load_rounds_and_params()
    specs = {"L2 raw-MW target": spec_raw_mw(rounds["lgbm_l2"]), "L2 (% target)": spec_l2(rounds["lgbm_l2"]),
             "HALO (% target)": spec_halo(hp, rounds["lgbm_halo"])}
    bt = backtest(tab, feats, specs)
    t = level_shift_table(bt, tab, specs.keys())
    s = summarise(bt, specs.keys())[["MAPE_all_pct", "under_rate_hc_pct"]].add_prefix("all days: ")
    return _save(t.join(s), "level_shift")


def exp_metric_disagreement() -> pd.DataFrame:
    """E6 - G6: do standard accuracy metrics rank models the same way as the
    peak-risk metrics? (If not, MAPE/RMSE alone would pick the wrong model.)"""
    bt = pd.read_parquet(C.PROCESSED_DIR / "predictions_backtest_2022_2025.parquet")
    bt = bt[bt["target_imputed"] == 0]
    names = ["lgbm_l2", "lgbm_quantile60", "lgbm_quantile80", "lgbm_prototype", "lgbm_linex", "lgbm_halo"]
    s = summarise(bt, names)
    s["RMSE_all_MW"] = [float(np.sqrt(((bt[f"pred_{n}"] - bt["peak_mw"]) ** 2).mean())) for n in names]
    tau = {c: stats.kendalltau(s["MAPE_all_pct"], s[c]).statistic for c in s.columns}
    ranks = s.rank()
    ranks.columns = [f"rank: {c}" for c in ranks.columns]
    out = s.join(ranks)
    out.loc["Kendall tau vs MAPE ranking"] = pd.Series(tau)
    return _save(out, "metric_disagreement")


def exp_lightweight() -> pd.DataFrame:
    """E7 - G9: training / inference cost of the final models."""
    tab, feats, _ = _main_table()
    rounds, hp = load_rounds_and_params()
    tr = MD.make_split(tab, feats, tab.index <= pd.Timestamp(C.VALID_END))
    te = MD.make_split(tab, feats, tab.index >= pd.Timestamp(C.TEST_START))
    rows = {}
    for name, fit, file in [("LGBM-L2", spec_l2(rounds["lgbm_l2"]), "lgbm_l2.txt"),
                            ("LGBM-HALO", spec_halo(hp, rounds["lgbm_halo"]), "lgbm_halo.txt")]:
        model, t_fit = timed(fit, tr)
        _, t_pred = timed(model.predict_mw, te)
        rows[name] = {"train_rows": len(tr.z), "features": len(feats), "train_seconds": t_fit,
                      "predict_365_days_ms": 1000 * t_pred,
                      "model_file_kB": (C.MODELS_DIR / file).stat().st_size / 1e3}
    return _save(pd.DataFrame(rows).T, "lightweight")
