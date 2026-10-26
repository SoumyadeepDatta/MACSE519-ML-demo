"""
heatwave.py - decide, for every state and every day, "is this a heatwave?",
and how SEVERE / how PERSISTENT the heat is.

Design principle: use the official definition, then add what it misses.

1. OFFICIAL (IMD) - cell level, computed by imdlib from IMD gridded Tmax:
       entry gate : Tmax >= 40 C (plains) / 37 C (coast) / 30 C (hills)
       then       : departure from 1991-2020 normal >= 4.5 C  (severe > 6.4)
                    or Tmax >= 45 C                         (severe >= 47)
   -> hw_area_frac = share of the state's area meeting it on that day.

2. STATE DECLARATION (this module) - IMD declares a heatwave only when the
   criteria hold for TWO CONSECUTIVE days. We apply that literally:
       state_hw_day(d) = hw_area_frac(d) >= 25 %  AND  hw_area_frac(d-1) >= 25 %
   It only looks backwards, so it is known on the morning of day d
   (no future information leaks into the features).

3. WHAT IMD's DRY-HEAT RULE MISSES (added as extra signals, not as labels):
   * Warm nights (IMD's own definition): Tmax >= 40 C and Tmin departure
     >= 4.5 C. Buildings cannot cool down overnight -> next-day AC load
     starts from a higher base. E.g. Delhi, 18-19 June 2024: Tmin 35 C,
     the warmest night in decades, and NR demand hit an all-time high.
   * Heat persistence: the 4th consecutive hot day draws more power than
     the 1st (thermal mass of concrete buildings, people giving up on fans
     and switching to ACs). Captured by run length and an exponentially
     weighted heat accumulation.
   * Humid heat (heat index from ERA5) for coastal states - see
     weather_processing.heat_index_c.

4. SECOND, INTERNATIONAL CRITERION - Excess Heat Factor (EHF; Nairn &
   Fawcett 2015, operational at the Australian Bureau of Meteorology):
       DMT      = daily mean temperature (Tmax + Tmin) / 2
       EHI_sig  = mean DMT of last 3 days - T95   (T95 = 95th pct of DMT,
                                                    1991-2020, per state)
       EHI_accl = mean DMT of last 3 days - mean DMT of the 30 days before
       EHF      = max(EHI_sig, 0) * max(EHI_accl, 1)
   EHF > 0 -> heatwave; EHF / EHF85 >= 1 severe, >= 3 extreme.
   Why add it? IMD's rule needs Tmax >= 40 C, so it almost never fires in
   humid states (Kerala, Tamil Nadu, West Bengal, the North-East), yet
   their humid heat drives real demand records - e.g. September 2023 and
   2024 "monsoon-break" heat set national demand records with almost no
   IMD heatwave. EHF is relative to each state's own climate (T95) and to
   recent acclimatisation (30-day mean), and it includes night-time heat
   through Tmin. Measured on 2019-2025: IMD alone flags ~26 national
   regime days in 7 years (none in 2025), too rare to learn or evaluate;
   IMD OR EHF flags ~205.

FINAL STATE RULE:   state_heat_day = IMD-declared heatwave  OR  EHF > 0
FINAL NATIONAL RULE: heat regime   = share of national demand in
                     state_heat_day states >= 20 %   (config.py)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def daily_normals(df: pd.DataFrame, col: str, window: int = 15) -> pd.Series:
    """
    Climatological normal of `col` for each (state, day-of-year), using the
    1991-2020 WMO normal period and a 15-day centred circular running mean
    (the same smoothing imdlib uses for the cell-level classification).

    Why smooth? A raw 30-year mean for "17 May" is noisy (30 samples); the
    15-day window uses ~450 samples and removes day-to-day jitter while
    keeping the seasonal cycle.
    """
    ref = df[(df["date"].dt.year >= C.CLIMATE_NORMAL_START) &
             (df["date"].dt.year <= C.CLIMATE_NORMAL_END)].copy()
    ref["doy"] = ref["date"].dt.dayofyear.clip(upper=365)
    clim = ref.groupby(["state", "doy"])[col].mean().unstack("state")
    clim = clim.reindex(range(1, 366))
    half = window // 2
    padded = pd.concat([clim.iloc[-half:], clim, clim.iloc[:half]])
    smooth = padded.rolling(window, center=True, min_periods=1).mean().iloc[half:-half]
    smooth.index = pd.Index(range(1, 366), name="doy")
    smooth.columns.name = "state"
    return smooth.stack().rename(f"{col}_normal")


def _run_length(flag: pd.Series) -> pd.Series:
    """Length of the current consecutive run of True values (0 when False).
    Example: [F, T, T, T, F, T] -> [0, 1, 2, 3, 0, 1]."""
    f = flag.astype(bool)
    grp = (~f).cumsum()
    return f.groupby(grp).cumsum().astype(int)


def add_heatwave_features(state_day: pd.DataFrame) -> pd.DataFrame:
    """
    Add departures, IMD declarations, warm nights and persistence features
    to a long (date, state) table that contains imd_tmax, imd_tmin,
    hw_area_frac, severe_hw_area_frac.
    """
    df = state_day.sort_values(["state", "date"]).reset_index(drop=True)
    df["doy"] = df["date"].dt.dayofyear.clip(upper=365)
    for col in ("imd_tmax", "imd_tmin"):
        normals = daily_normals(df, col).reset_index()
        df = df.merge(normals, on=["doy", "state"], how="left")
        df[f"{col}_departure"] = df[col] - df[f"{col}_normal"]

    df = df.sort_values(["state", "date"]).reset_index(drop=True)
    g = df.groupby("state", group_keys=False)

    # --- IMD 2-consecutive-day declaration rule, state level --------------
    hot_today = df["hw_area_frac"] >= C.STATE_HW_AREA_FRACTION
    hot_yday = g["hw_area_frac"].shift(1).fillna(0) >= C.STATE_HW_AREA_FRACTION
    df["state_hw_day"] = (hot_today & hot_yday).astype(int)
    sev_today = df["severe_hw_area_frac"] >= C.STATE_HW_AREA_FRACTION
    df["state_severe_hw_day"] = (df["state_hw_day"].astype(bool) & sev_today).astype(int)

    # --- IMD warm night ------------------------------------------------------
    df["warm_night"] = ((df["imd_tmax"] >= C.IMD_PLAINS_ENTRY) &
                        (df["imd_tmin_departure"] >= C.IMD_HW_DEPARTURE)).astype(int)

    # --- Persistence -------------------------------------------------------
    df["hw_run_length"] = g["state_hw_day"].transform(_run_length)
    # "Hot day" (looser than heatwave): Tmax >= 40 C. Run length of hot days
    # captures persistent pre-monsoon heat that is normal-for-season (thus
    # not an IMD heatwave) yet still drives record demand, e.g. 40-42 C for
    # 10 straight days over the Indo-Gangetic plain in June.
    df["hot_day_run_length"] = g["imd_tmax"].transform(lambda s: _run_length(s >= 40.0))
    # Exponentially-weighted heat load: excess over 35 C, half-life ~2 days.
    # A day at 44 C after three days at 44 C scores ~2.6x a lone 44 C day
    # (6.75 vs 2.64 with alpha = 1 - 0.5**(1/2) = 0.293).
    excess = (df["imd_tmax"] - 35.0).clip(lower=0)
    df["heat_accum_ewm"] = excess.groupby(df["state"]).transform(
        lambda s: s.ewm(halflife=2.0, adjust=False).mean())
    # Cooling degree-days on the IMD daily mean, base 24 C (ASHRAE/BEE
    # practice for India), and its 3-day sum (thermal lag of buildings).
    tmean = (df["imd_tmax"] + df["imd_tmin"]) / 2
    df["cdd24"] = (tmean - 24.0).clip(lower=0)
    df["cdd24_3d"] = df.groupby("state")["cdd24"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    df["tmax_change_1d"] = g["imd_tmax"].diff()

    # --- Excess Heat Factor (see module docstring, point 4) -----------------
    df["dmt"] = tmean
    ref = df[(df["date"].dt.year >= C.CLIMATE_NORMAL_START) &
             (df["date"].dt.year <= C.CLIMATE_NORMAL_END)]
    t95 = ref.groupby("state")["dmt"].quantile(0.95)
    dmt3 = df.groupby("state")["dmt"].transform(lambda s: s.rolling(3, min_periods=3).mean())
    dmt30 = df.groupby("state")["dmt"].transform(
        lambda s: s.shift(3).rolling(30, min_periods=20).mean())
    df["ehi_sig"] = dmt3 - df["state"].map(t95)
    df["ehi_accl"] = dmt3 - dmt30
    df["ehf"] = (df["ehi_sig"].clip(lower=0) * df["ehi_accl"].clip(lower=1)).fillna(0.0)
    ref_pos = df[(df["ehf"] > 0) & (df["date"].dt.year <= C.CLIMATE_NORMAL_END)]
    ehf85 = ref_pos.groupby("state")["ehf"].quantile(0.85)
    df["ehf_severity"] = (df["ehf"] / df["state"].map(ehf85)).fillna(0.0)

    # --- Final state-level heat day (IMD official OR EHF) -------------------
    df["state_heat_day"] = ((df["state_hw_day"] == 1) | (df["ehf"] > 0)).astype(int)
    df["heat_day_run_length"] = df.groupby("state")["state_heat_day"].transform(_run_length)

    # Continuous severity in [0, 1] - the larger of
    #   IMD part : area under heatwave, severe area counting double
    #   EHF part : EHF / EHF85 / 3   (BoM "extreme" EHF maps to 1.0)
    # Used by the severity-adaptive loss (losses.py) and as a feature.
    imd_part = ((df["hw_area_frac"] + df["severe_hw_area_frac"]) / 2).clip(0, 1)
    ehf_part = (df["ehf_severity"] / 3.0).clip(0, 1)
    df["heat_severity"] = np.maximum(imd_part, ehf_part)
    return df.drop(columns=["doy"])


def national_heat_exposure(state_day: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """
    Demand-weighted national heat indicators (one row per day).

    imd_hw_exposure  = sum_s w_s * state_hw_day_s      (IMD-declared only)
    heat_exposure    = sum_s w_s * state_heat_day_s    (IMD OR EHF)
                       "what share of India's electricity demand is sitting
                        inside a heatwave today?"
    severe_exposure  = same with IMD severe heatwave days
    heat_severity_idx= sum_s w_s * heat_severity_s   (continuous, 0..1)
    warm_night_exposure = sum_s w_s * warm_night_s
    hot40_exposure   = share of demand in states with IMD Tmax >= 40 C

    Example (computed in notebook 03): on 28-May-2024 Rajasthan, Haryana,
    Delhi, Punjab, UP and MP were all under IMD-declared heatwave ->
    imd_hw_exposure = 0.37, i.e. more than a third of India's demand.
    """
    w = weights.reindex(state_day["state"].unique()).fillna(0)
    w = w / w.sum()
    df = state_day.assign(w=state_day["state"].map(w))
    agg = df.assign(
        imd_hw=df["w"] * df["state_hw_day"],
        heat=df["w"] * df["state_heat_day"],
        sev=df["w"] * df["state_severe_hw_day"],
        hsi=df["w"] * df["heat_severity"],
        wn=df["w"] * df["warm_night"],
        hot40=df["w"] * (df["imd_tmax"] >= 40.0),
    ).groupby("date")[["imd_hw", "heat", "sev", "hsi", "wn", "hot40"]].sum()
    agg.columns = ["imd_hw_exposure", "heat_exposure", "severe_exposure",
                   "heat_severity_idx", "warm_night_exposure", "hot40_exposure"]
    agg["national_hw_regime"] = (agg["heat_exposure"] >= C.NATIONAL_HW_EXPOSURE_THRESHOLD).astype(int)
    return agg
