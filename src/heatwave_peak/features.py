"""
features.py - build the leakage-safe DAY-AHEAD modelling table.

Forecasting setting (what an Indian grid operator actually does)
---------------------------------------------------------------
Every morning NLDC/RLDCs schedule generation for TOMORROW (day D). They know:
  * all demand up to and including yesterday (D-1)
  * the weather forecast for D (IMD issues district forecasts daily)
We mirror that: load features come from D-1 and earlier only; weather
features describe day D. We use the OBSERVED weather of D as a stand-in for
a perfect forecast (the standard "ex-post" set-up in load-forecasting
research, e.g. GEFCom competitions). In deployment, IMD's NWP forecast would
be plugged in; the forecast error would then add some noise on top.

Target transform: why "% above last week" instead of MW
-------------------------------------------------------
Gradient-boosted trees cannot extrapolate: every leaf predicts an average of
TRAINING targets. Indian demand grows 6-8 %/yr, so a model trained on
2019-2023 MW values (max ~237 GW) can never predict 2025's 240 GW+ peaks -
it would systematically under-forecast, the most dangerous error.
So the model predicts
        z_D = 100 * (peak_D / base_D - 1),     base_D = mean(peak_{D-1..D-7})
i.e. "tomorrow's peak in % above the last 7 days' average". z is stationary
(about -10 .. +10 %) and the MW forecast is recovered as base_D*(1+z/100).
Real-life reading: "a heatwave arrives tomorrow -> peak +4 % over last week".
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from . import config as C


def indian_holidays(years) -> pd.DataFrame:
    """
    National and major festival holidays of India (python-holidays package,
    which follows the Government of India gazetted list: Republic Day,
    Holi, Good Friday, Id-ul-Fitr, Independence Day, Gandhi Jayanti,
    Dussehra, Diwali, Christmas ...).

    Why it matters: on Diwali the peak DROPS ~8-10 % (offices/factories
    shut) even though lighting rises at night; Holi afternoon drops ~15 %.
    Without this flag the model blames weather for a holiday dip.
    """
    h = holidays.country_holidays("IN", years=list(years))
    return pd.DataFrame({"date": pd.to_datetime(list(h.keys())), "holiday_name": list(h.values())})


MAJOR_FESTIVALS = "Holi|Diwali|Dussehra|Independence Day|Republic Day|Id-ul-Fitr|Id-ul-Zuha|Christmas|Gandhi"


def _calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    cal = pd.DataFrame(index=dates)
    cal["dow"] = dates.dayofweek
    cal["is_sunday"] = (dates.dayofweek == 6).astype(int)
    cal["is_saturday"] = (dates.dayofweek == 5).astype(int)
    cal["month"] = dates.month
    doy = dates.dayofyear
    cal["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    cal["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    hol = indian_holidays(range(dates.year.min() - 1, dates.year.max() + 2))
    hol_days = pd.DatetimeIndex(hol["date"]).normalize().unique().sort_values()
    cal["is_holiday"] = dates.isin(hol_days).astype(int)
    # Distance to nearest holiday (capped at 7): demand dips around long
    # weekends and the days between Diwali and Bhai Dooj.
    pos = np.searchsorted(hol_days.values, dates.values)
    nxt = hol_days.values[np.clip(pos, 0, len(hol_days) - 1)]
    prv = hol_days.values[np.clip(pos - 1, 0, len(hol_days) - 1)]
    d_next = (nxt - dates.values) / np.timedelta64(1, "D")
    d_prev = (dates.values - prv) / np.timedelta64(1, "D")
    cal["days_to_holiday"] = np.clip(np.where(d_next < 0, 7, d_next), 0, 7)
    cal["days_from_holiday"] = np.clip(np.where(d_prev < 0, 7, d_prev), 0, 7)
    # Not all gazetted holidays are equal. Mahavir Jayanti barely moves
    # national demand; Holi and Diwali cut the peak by 8-11 % and the day
    # AFTER rebounds by ~+11 % (validation 2024: 25->26 March, Holi). The
    # rebound is the dangerous one: yesterday's low peak drags the lag
    # features down and the model under-forecasts today.
    major = hol[hol["holiday_name"].str.contains(MAJOR_FESTIVALS, case=False, regex=True)]
    major_days = pd.DatetimeIndex(major["date"]).normalize().unique()
    cal["is_major_festival"] = dates.isin(major_days).astype(int)
    cal["major_festival_yesterday"] = dates.isin(major_days + pd.Timedelta(days=1)).astype(int)
    # Diwali week: Diwali, Govardhan Puja and Bhai Dooj span ~4 days of
    # depressed industrial load across North and West India.
    diwali = pd.DatetimeIndex(hol.loc[hol["holiday_name"].str.contains("Diwali"), "date"]).normalize()
    win = pd.DatetimeIndex(sorted({d + pd.Timedelta(days=k) for d in diwali for k in range(-1, 4)}))
    cal["diwali_window"] = dates.isin(win).astype(int)
    lock_a, lock_b = pd.Timestamp(C.COVID_LOCKDOWN[0]), pd.Timestamp(C.COVID_LOCKDOWN[1])
    cal["covid_lockdown"] = ((dates >= lock_a) & (dates <= lock_b)).astype(int)
    return cal


def _load_history(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Load features that are known on the morning of D (D-1 and earlier).
    All are expressed RELATIVE to base_D (see module docstring), in %.
    """
    p = daily["peak_mw"]
    base = p.shift(1).rolling(7, min_periods=5).mean()
    f = pd.DataFrame(index=daily.index)
    f["base_mw"] = base

    def rel(x):
        return 100 * (x / base - 1)

    for lag in (1, 2, 3, 7, 14):
        f[f"peak_lag{lag}_rel"] = rel(p.shift(lag))
    f["peak_lag364_rel"] = rel(p.shift(364))          # same weekday last year
    f["peak_max7_rel"] = rel(p.shift(1).rolling(7).max())
    f["peak_std7_pct"] = 100 * p.shift(1).rolling(7).std() / base
    f["trend_1v7_pct"] = rel(p.shift(1).rolling(3).mean())  # momentum
    # Year-on-year growth of the weekly base: how fast demand is growing.
    f["yoy_growth_pct"] = 100 * (base / base.shift(364) - 1)
    for col in ("mean_mw", "min_mw", "solar_peak_mw", "evening_peak_mw"):
        f[f"{col}_lag1_rel"] = rel(daily[col].shift(1))
    f["load_factor_lag1"] = daily["load_factor"].shift(1)
    f["peak_hour_lag1"] = daily["peak_hour"].shift(1)
    # Same-weekday seasonal ratio: last week's same weekday vs its own base
    f["weekday_effect_lag7"] = 100 * (p.shift(7) / p.shift(8).rolling(7).mean() - 1)
    return f


def _weather_deltas(weather: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Weather of D MINUS the mean weather of D-1..D-7 (the baseline week).

    Because the target is "peak relative to the last 7 days", the matching
    weather signal is "weather relative to the last 7 days". Example: Delhi
    at 44 C after a week at 44 C changes little (already in base); 44 C
    after a week at 38 C (a sudden heat surge) is what pushes z up.
    """
    out = {}
    for c in cols:
        out[f"{c}__d7"] = weather[c] - weather[c].shift(1).rolling(7, min_periods=5).mean()
    return pd.DataFrame(out, index=weather.index)


def build_model_table(daily: pd.DataFrame, national_heat: pd.DataFrame,
                      zone_wide: pd.DataFrame, national_weather: pd.DataFrame,
                      psp_daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Assemble the final table: target, MW baseline, weights, regime flags
    and all features, indexed by date (STUDY_START .. STUDY_END).
    """
    idx = daily.index
    tab = pd.DataFrame(index=idx)
    tab["peak_mw"] = daily["peak_mw"]
    lh = _load_history(daily)
    tab = tab.join(lh)
    tab["target_z"] = 100 * (tab["peak_mw"] / tab["base_mw"] - 1)

    tab = tab.join(_calendar(idx))
    tab = tab.join(national_heat.reindex(idx))
    tab = tab.join(national_weather.reindex(idx))
    tab = tab.join(zone_wide.reindex(idx))

    # Weekly deltas for the most demand-relevant signals (national + zones).
    key = [c for c in tab.columns if c.endswith(("imd_tmax", "era5_cdh24", "era5_heat_index_max",
                                                 "era5_precip_sum", "heat_severity", "era5_night_tmin"))]
    key += ["heat_exposure", "heat_severity_idx"]
    tab = tab.join(_weather_deltas(tab, [c for c in key if c in tab.columns]))

    if psp_daily is not None:
        # Grid-operations context from yesterday's NLDC PSP report.
        ops = pd.DataFrame(index=psp_daily.index)
        ops["share_res_pct_lag1"] = psp_daily["share_res_pct"]
        ops["energy_shortage_mu_lag1"] = psp_daily["energy_shortage_total_mu"]
        ops["solar_gen_mu_lag1"] = psp_daily["solar_gen_total_mu"]
        tab = tab.join(ops.shift(1, freq="D").reindex(idx))

    # ---- flags used for weighting and evaluation (NOT features) ----------
    tab["is_critical_peak"] = daily["is_critical_peak"].astype(int)
    tab["target_imputed"] = daily["target_imputed"].astype(int)
    tab["split"] = np.select(
        [idx <= pd.Timestamp(C.TRAIN_END),
         (idx >= pd.Timestamp(C.VALID_START)) & (idx <= pd.Timestamp(C.VALID_END))],
        ["train", "valid"], default="test")
    # Training weight 0 for: COVID lockdown (demand broken from weather),
    # imputed targets (not a real hourly measurement), and the first weeks
    # where lag features are undefined.
    tab["train_weight"] = 1.0
    tab.loc[tab["covid_lockdown"] == 1, "train_weight"] = 0.0
    tab.loc[tab["target_imputed"] == 1, "train_weight"] = 0.0
    tab.loc[tab["base_mw"].isna() | tab["target_z"].isna(), "train_weight"] = 0.0
    return tab


# Columns that are NOT model inputs.
NON_FEATURES = {
    "peak_mw", "base_mw", "target_z", "is_critical_peak", "target_imputed",
    "split", "train_weight", "national_hw_regime", "covid_lockdown",
}


def feature_columns(tab: pd.DataFrame) -> list[str]:
    return [c for c in tab.columns if c not in NON_FEATURES]
