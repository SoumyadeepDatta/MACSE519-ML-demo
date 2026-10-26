"""
load_processing.py - clean, calibrate and stitch hourly All-India demand,
then extract the DAILY PEAK (the forecasting target).

Why three sources must be stitched
----------------------------------
No single free dataset covers 2019-2025 hourly:

    2019-01-01 ............ 2024-04-30   Kaggle hourly (Grid-India)      [A]
                  2024-01-01 ............ 2025-06-24   Mendeley NERLDC   [B]
                            2024-11-04 ............ 2026-09  NLDC SCADA  [C]

May-Oct 2024 exists ONLY in [B] - and it holds the record heatwave
(All-India peak 250 GW on 30-May-2024). Dropping it would remove the most
important event in the whole study, so we stitch A -> B -> C.

Why calibration is needed
-------------------------
The sources measure the same grid with different instruments:
  [A] SCADA hourly averages, [B] PMU (phasor) measurements, [C] 15-min
  SCADA. On the Jan-Apr 2024 overlap [B] reads ~0.8 % higher than [A].
  0.8 % of 200 GW = 1.6 GW - that is the size of a large power station and
  about a third of a good model's error, so an uncorrected "step" at the
  seam would be learned as fake demand growth. We rescale [B] and [C] to
  [A]'s level with a robust (median) ratio estimated on the overlaps.

Why a daily PEAK and not daily energy
-------------------------------------
Grid adequacy fails at the peak: on 30-May-2024 the grid needed 250 GW for
roughly an hour in the afternoon. Energy (MWh/day) tells you how much coal
to burn; the peak tells you whether you have enough plants online at
3 pm. Load-shedding happens when the PEAK is under-planned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

REGION_COLS = {
    "Northen Region Hourly Demand": "NR",   # (sic) spelling in the source file
    "Western Region Hourly Demand": "WR",
    "Southern Region Hourly Demand": "SR",
    "Eastern Region Hourly Demand": "ER",
    "North-Eastern Region Hourly Demand": "NER",
}


# ----------------------------------------------------------------------------
# Readers (one per raw source) - they only parse; no cleaning decisions here
# ----------------------------------------------------------------------------
def read_kaggle_hourly() -> pd.DataFrame:
    """[A] hourly national + regional demand, MW, 2019-01 .. 2024-04."""
    df = pd.read_excel(C.RAW_KAGGLE_HOURLY_DIR / "hourlyLoadDataIndia.xlsx")
    df = df.rename(columns={"National Hourly Demand": "national_mw", **REGION_COLS})
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.set_index("datetime").sort_index()


def read_mendeley_hourly() -> pd.Series:
    """[B] hourly All-India demand (MW), 2024-01-01 .. 2025-06-24.

    The sheet ends with summary rows ("Timestamp of max", "Average", "Sum");
    those fail date parsing with errors='coerce' and are dropped.
    """
    raw = pd.read_excel(C.RAW_MENDELEY_DIR / "January 2024- June 2025.xlsx")
    ts = pd.to_datetime(raw["Timestamp"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    s = pd.Series(pd.to_numeric(raw["Demand (MW)"], errors="coerce").values, index=ts)
    s = s[s.index.notna()].sort_index()
    return s[~s.index.duplicated()].rename("national_mw")


def read_scada_hourly(min_slots_per_hour: int = 3) -> pd.Series:
    """[C] NLDC 15-minute SCADA demand -> hourly mean (MW).

    Hour H's value = mean of the 15-min blocks H:00, H:15, H:30, H:45, i.e.
    the same "hourly average" convention as [A]. An hour with fewer than 3 of
    its 4 blocks (corrupted report days such as 2025-10-02) is set to NaN
    rather than trusting one or two blocks.
    """
    df = pd.read_csv(C.RAW_PSP_DIR / "study2_scada.csv",
                     usecols=["date", "time", "demand_met_mw"])
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    s = df.drop_duplicates("datetime").set_index("datetime")["demand_met_mw"]
    s = s.where(s > 0)                                    # zeros = telemetry gaps
    g = s.resample("h")
    hourly = g.mean().where(g.count() >= min_slots_per_hour)
    return hourly.rename("national_mw")


def read_psp_daily() -> pd.DataFrame:
    """NLDC daily Power Supply Position: official daily max demand met etc."""
    df = pd.read_csv(C.RAW_PSP_DIR / "study1_daily.csv", parse_dates=["date"])
    return df.set_index("date").sort_index()


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------
def flag_spikes(s: pd.Series, neighbour_tol: float = 0.10,
                reference_tol: float = 0.15, hard_tol: float = 0.30) -> pd.Series:
    """
    Flag telemetry glitches in an hourly national-demand series.

    A value is a glitch if it is BOTH
      (a) >10 % away from the average of the hour before and after
          (all-India demand is a sum over ~300 million consumers; it cannot
          jump 10 % for one hour and come back), AND
      (b) >15 % away from the median of the SAME clock hour over a centred
          7-day window,
    OR if it is >30 % away from that 7-day same-hour median (no real event
    except the COVID lockdown does that, and the lockdown persists for weeks
    so the 7-day median follows it).

    Why two conditions? Real one-off events exist and must be KEPT:
      * 05-Apr-2020 21:00 - PM's "9 pm, 9 minutes" lights-off: -31 GW in
        minutes, deviation from neighbours ~10 %, but genuine.
      * Holi (25-Mar-2024) evening, Independence Day afternoons: genuine
        festival dips of 15-17 % versus the weekly reference.
    Requiring both (a) and (b) keeps such events and removes only isolated
    sensor spikes (e.g. 17-Dec-2021 05:00, +17 % for a single hour).
    """
    nb = (s.shift(1) + s.shift(-1)) / 2
    dev_nb = (s / nb - 1).abs()
    ref = s.groupby(s.index.hour).transform(
        lambda x: x.rolling(7, center=True, min_periods=3).median())
    dev_ref = (s / ref - 1).abs()
    return ((dev_nb > neighbour_tol) & (dev_ref > reference_tol)) | (dev_ref > hard_tol)


def overlap_ratio(reference: pd.Series, other: pd.Series) -> tuple[float, int]:
    """Median of reference/other on common, clean hours (robust to glitches)."""
    j = pd.concat([reference, other], axis=1, keys=["ref", "oth"]).dropna()
    j = j[~flag_spikes(j["ref"]) & ~flag_spikes(j["oth"])]
    return float(np.median(j["ref"] / j["oth"])), len(j)


def build_national_hourly() -> tuple[pd.DataFrame, dict]:
    """
    Stitch [A] -> [B] -> [C] into one hourly series on [A]'s measurement scale.

    Returns
    -------
    hourly : DataFrame indexed by hour with columns
        national_mw    cleaned, calibrated demand (MW)
        source         'kaggle_A' | 'mendeley_B' | 'scada_C'
        was_glitch     True where the raw value was flagged and replaced
        was_filled     True where the value is interpolated (gap <= 3 h)
    calib : dict with the calibration factors and the overlap sizes.
    """
    a = read_kaggle_hourly()["national_mw"]
    b = read_mendeley_hourly()
    c = read_scada_hourly()

    f_b, n_ab = overlap_ratio(a, b)          # A-level / B-level
    f_cb, n_bc = overlap_ratio(b, c)         # B-level / C-level
    f_c = f_b * f_cb                         # chain C -> B -> A
    calib = {"factor_mendeley_to_kaggle": f_b, "overlap_hours_A_B": n_ab,
             "factor_scada_to_kaggle": f_c, "overlap_hours_B_C": n_bc}

    b_cal, c_cal = b * f_b, c * f_c
    a_end, c_start = a.index.max(), c.index.min()
    parts = [
        a.to_frame().assign(source="kaggle_A"),
        b_cal[(b_cal.index > a_end) & (b_cal.index < c_start)].to_frame().assign(source="mendeley_B"),
        c_cal[c_cal.index >= c_start].to_frame().assign(source="scada_C"),
    ]
    h = pd.concat(parts).sort_index()
    h = h[~h.index.duplicated(keep="first")]
    full_idx = pd.date_range(C.STUDY_START, f"{C.STUDY_END} 23:00", freq="h")
    h = h.reindex(full_idx)
    h.index.name = "datetime"

    # Where SCADA [C] is missing but Mendeley [B] (calibrated) has the hour,
    # use [B] - a real measurement beats an interpolation.
    gap = h["national_mw"].isna()
    fill_b = b_cal.reindex(h.index)
    use_b = gap & fill_b.notna()
    h.loc[use_b, "national_mw"] = fill_b[use_b]
    h.loc[use_b, "source"] = "mendeley_B"

    h["was_glitch"] = flag_spikes(h["national_mw"]).fillna(False)
    h.loc[h["was_glitch"], "national_mw"] = np.nan
    before = h["national_mw"].isna()
    # Short gaps (<= 3 hours) are interpolated in time; longer gaps (whole
    # missing report days) stay NaN and are handled at the DAILY level with
    # the official PSP daily maximum (see build_daily_peaks).
    h["national_mw"] = h["national_mw"].interpolate("time", limit=3, limit_area="inside")
    h["was_filled"] = before & h["national_mw"].notna()
    h["source"] = h["source"].fillna("missing")
    return h, calib


def build_regional_hourly() -> pd.DataFrame:
    """[A]'s five regional series (2019-01 .. 2024-04), glitch-cleaned.
    Used for exploratory analysis and to compute regional demand shares."""
    a = read_kaggle_hourly()[list(REGION_COLS.values())]
    out = a.copy()
    for col in out.columns:
        bad = flag_spikes(out[col])
        out.loc[bad, col] = np.nan
        out[col] = out[col].interpolate("time", limit=3, limit_area="inside")
    return out


# ----------------------------------------------------------------------------
# Daily peak extraction (the TARGET)
# ----------------------------------------------------------------------------
def build_daily_peaks(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly demand to one row per day.

    Columns produced (all MW unless noted):
      peak_mw          max hourly demand of the day          <- TARGET
      peak_hour        clock hour of that maximum (0-23). In 2019 the
                       peak was ~19-20 h (evening lighting); by 2024 summer
                       days peak at ~15 h (AC load while rooftop solar has
                       not yet rolled off) - a structural change worth seeing.
      min_mw           night-time trough (base load)
      mean_mw          average demand = daily energy / 24 h
      load_factor      mean / peak (0-1). Heatwave days push the peak up
                       faster than the mean, so load factor drops.
      solar_peak_mw    max over 09-16 h   (day-time AC + agriculture)
      evening_peak_mw  max over 17-23 h   (lighting + residential AC)
      n_valid_hours    hours with data; a day needs >= 20 to be trusted
      target_imputed   True when the peak came from the PSP fallback
    """
    d = hourly["national_mw"]
    day = d.index.floor("D")
    g = d.groupby(day)
    daily = pd.DataFrame({
        "peak_mw": g.max(),
        "peak_hour": g.apply(lambda x: x.idxmax().hour if x.notna().any() else np.nan),
        "min_mw": g.min(),
        "mean_mw": g.mean(),
        "n_valid_hours": g.count(),
    })
    hr = d.index.hour
    daily["solar_peak_mw"] = d[(hr >= 9) & (hr <= 16)].groupby(day[(hr >= 9) & (hr <= 16)]).max()
    daily["evening_peak_mw"] = d[(hr >= 17)].groupby(day[hr >= 17]).max()
    daily.index.name = "date"

    # ---- Fallback for days with too little hourly data ---------------------
    # On a missing report day we still know the OFFICIAL daily maximum demand
    # met from NLDC's PSP report (an instantaneous maximum, which is ~1 %
    # higher than a max of hourly averages). We convert it to the hourly-max
    # scale with the median ratio observed on days where both exist.
    psp = read_psp_daily()["max_demand_met_total_mw"].where(lambda x: x > 50_000)
    ok = daily["n_valid_hours"] >= 20
    ratio = (daily.loc[ok, "peak_mw"] / psp.reindex(daily.index)[ok]).dropna()
    ratio = float(ratio.clip(0.9, 1.1).median())
    daily["target_imputed"] = ~ok
    fill = psp.reindex(daily.index) * ratio
    daily.loc[~ok, "peak_mw"] = fill[~ok]
    for col in ("min_mw", "mean_mw", "solar_peak_mw", "evening_peak_mw", "peak_hour"):
        daily.loc[~ok, col] = np.nan
    daily["load_factor"] = daily["mean_mw"] / daily["peak_mw"]
    daily.attrs["psp_to_hourly_peak_ratio"] = ratio
    return daily


def add_critical_peak_flag(daily: pd.DataFrame,
                           window: int = C.CRITICAL_PEAK_WINDOW_DAYS,
                           q: float = C.CRITICAL_PEAK_QUANTILE) -> pd.DataFrame:
    """
    Mark growth-adjusted "critical peak" days.

    Threshold_t = 90th percentile of peaks over days t-365 .. t-1 (strictly
    past, so the flag could be computed operationally each morning).

    Example: 220 GW was a record in 2023 but an ordinary June day in 2025.
    A rolling quantile adapts to such growth whereas a fixed MW cut-off
    would not.
    """
    out = daily.copy()
    thr = out["peak_mw"].shift(1).rolling(window, min_periods=180).quantile(q)
    out["critical_peak_threshold_mw"] = thr
    out["is_critical_peak"] = (out["peak_mw"] >= thr) & thr.notna()
    return out


def load_state_daily_demand() -> pd.DataFrame:
    """
    NLDC daily state-wise maximum demand met (MW), mapped to IMD state names.

    Non-geographic entities are dropped: captive industrial consumers
    (AMNSIL steel, BALCO aluminium, Essar Steel, RIL Jamnagar) and railway /
    bulk-consumer drawals. They run 24x7 regardless of temperature and
    would only dilute the weather signal.
    """
    s = pd.read_csv(C.RAW_PSP_DIR / "study3_states.csv", parse_dates=["date"])
    alias = {n: st.imd_name for st in C.STATES for n in st.nldc_names}
    s["state_imd"] = s["state"].map(alias)
    s = s.dropna(subset=["state_imd"])
    s = s[(s["max_demand_met_mw"] > 0) & (s["date"] >= "2018-12-31")]
    # DD + DNH were reported separately until May-2022 and merged afterwards;
    # Jharkhand + DVC are summed. A sum of maxima slightly overstates the
    # combined maximum but these entities peak at the same time of day.
    out = (s.groupby(["date", "state_imd"])
             .agg(max_demand_mw=("max_demand_met_mw", "sum"),
                  energy_met_mu=("energy_met_mu", "sum"))
             .reset_index().rename(columns={"state_imd": "state"}))
    return out


def state_demand_weights(state_daily: pd.DataFrame,
                         start: str = C.STUDY_START, end: str = C.TRAIN_END) -> pd.Series:
    """
    Each state's share of the national demand, from the TRAINING years only.

    Used to aggregate state weather into zone / national weather. Example:
    UP (~20 GW) gets ~10x the weight of Uttarakhand (~2 GW): a heatwave over
    Lucknow moves the national peak far more than one over Dehradun.
    Computing the shares on 2019-2023 only keeps the test year unseen.
    """
    m = state_daily[(state_daily["date"] >= start) & (state_daily["date"] <= end)]
    w = m.groupby("state")["max_demand_mw"].mean()
    return (w / w.sum()).rename("demand_weight").sort_values(ascending=False)
