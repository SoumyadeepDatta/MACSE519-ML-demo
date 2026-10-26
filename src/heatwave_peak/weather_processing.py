"""
weather_processing.py - turn raw weather into one clean STATE x DAY table.

Two complementary weather sources
---------------------------------
IMD gridded Tmax/Tmin (daily, 1 deg, whole-state area average)
    + official Government of India data, used by IMD for heatwave warnings
    + spatially complete: averages every 1-degree cell inside a state
    - only two numbers a day (max and min temperature), no humidity

ERA5 hourly at the state's main load centre (Open-Meteo)
    + hourly: tells us WHEN it is hot (a 34 C night in Delhi keeps ACs on
      at 2 am; a 44 C afternoon with a 26 C night lets buildings cool off)
    + humidity, dew point, rain, cloud, sun, wind
    - a single point (the load centre), and reanalysis has a known cool
      bias of 1-2 C in Indian summer afternoons -> we use ERA5 for
      SHAPE/humidity features and IMD for the official heat thresholds.

Every hourly ERA5 variable is collapsed to a handful of physically
meaningful daily statistics (see `aggregate_era5_daily`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# ----------------------------------------------------------------------------
# IMD gridded -> state daily series
# ----------------------------------------------------------------------------
def _open_imd(var: str, start: int = C.CLIMATE_NORMAL_START, end: int = 2025):
    import imdlib as imd
    return imd.open_data(var, start, end, "yearwise", file_dir=str(C.RAW_IMD_DIR))


def imd_state_daily() -> pd.DataFrame:
    """
    Area-weighted state means of IMD daily Tmax / Tmin plus the share of each
    state's area that meets IMD's heatwave / severe-heatwave cell criteria.

    The cell-level heatwave classification is imdlib's implementation of the
    official IMD two-gate rule (terrain-specific entry threshold, then
    departure-from-normal OR absolute threshold) with 1991-2020 normals.

    Returns a long table: date, state, imd_tmax, imd_tmin,
    hw_area_frac, severe_hw_area_frac  (1991-01-01 .. 2025-12-31).
    """
    tmax = _open_imd("tmax")
    tmin = _open_imd("tmin")
    states = C.STATE_NAMES

    t_max = tmax.region(state=states)
    t_min = tmin.region(state=states)

    cls = tmax.copy().heatwave(output="daily",
                               norm_start=C.CLIMATE_NORMAL_START,
                               norm_end=C.CLIMATE_NORMAL_END)
    nan = np.isnan(cls.data)
    any_hw, severe = cls.copy(), cls.copy()
    any_hw.data = np.where(nan, np.nan, (cls.data >= 1).astype(float))
    severe.data = np.where(nan, np.nan, (cls.data >= 2).astype(float))
    hw_frac = any_hw.region(state=states)
    sev_frac = severe.region(state=states)

    def _long(df, name):
        df = df.copy()
        df.index = pd.to_datetime(df.index).normalize()
        df.index.name = "date"
        return df.stack(future_stack=True).rename(name)

    out = pd.concat([_long(t_max, "imd_tmax"), _long(t_min, "imd_tmin"),
                     _long(hw_frac, "hw_area_frac"), _long(sev_frac, "severe_hw_area_frac")],
                    axis=1).reset_index().rename(columns={"level_1": "state"})
    # IMD encodes missing as 99.9 / -99.9; region() already drops them, but a
    # state whose only cell is missing on a day would give NaN -> fill from
    # the previous day (temperature is strongly persistent day-to-day).
    out = out.sort_values(["state", "date"])
    for col in ("imd_tmax", "imd_tmin"):
        out[col] = out.groupby("state")[col].transform(lambda s: s.ffill(limit=2))
    out[["hw_area_frac", "severe_hw_area_frac"]] = out[["hw_area_frac", "severe_hw_area_frac"]].fillna(0.0)
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------
# ERA5 hourly -> state daily features
# ----------------------------------------------------------------------------
def heat_index_c(t_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """
    NOAA / US National Weather Service heat index (Rothfusz 1990 regression
    with Steadman's simple formula below 26.7 C), returned in deg C.

    Why: the human body (and so AC usage) responds to temperature AND
    humidity. Example - Chennai 36 C at 65 % RH -> heat index ~ 50 C
    ("extreme danger"), while Jaipur 42 C at 12 % RH -> ~ 40 C. Both are
    stressful; only the heat index shows that Chennai is even worse.
    """
    t_f = t_c * 9 / 5 + 32
    simple = 0.5 * (t_f + 61.0 + (t_f - 68.0) * 1.2 + rh * 0.094)
    hi = (-42.379 + 2.04901523 * t_f + 10.14333127 * rh - 0.22475541 * t_f * rh
          - 6.83783e-3 * t_f ** 2 - 5.481717e-2 * rh ** 2 + 1.22874e-3 * t_f ** 2 * rh
          + 8.5282e-4 * t_f * rh ** 2 - 1.99e-6 * t_f ** 2 * rh ** 2)
    # NWS adjustments for very dry and very humid conditions
    adj_dry = ((13 - rh) / 4) * np.sqrt(np.clip((17 - np.abs(t_f - 95.0)) / 17, 0, None))
    hi = np.where((rh < 13) & (t_f >= 80) & (t_f <= 112), hi - adj_dry, hi)
    adj_wet = ((rh - 85) / 10) * ((87 - t_f) / 5)
    hi = np.where((rh > 85) & (t_f >= 80) & (t_f <= 87), hi + adj_wet, hi)
    hi_f = np.where((simple + t_f) / 2 < 80, simple, hi)
    return (hi_f - 32) * 5 / 9


def read_era5_state(state: str) -> pd.DataFrame:
    f = C.RAW_ERA5_DIR / f"era5_hourly_{state.replace(' ', '_')}.csv"
    df = pd.read_csv(f, parse_dates=["datetime"])
    return df


def aggregate_era5_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse one state's hourly ERA5 rows to daily features.

    Feature                 meaning / why it matters for peak demand
    ----------------------  --------------------------------------------------
    era5_tmax/tmean/tmin    daily temperature envelope at the load centre
    era5_night_tmin         min over 00-06 h: hot nights = ACs never switch off
    era5_t_15h              temperature at 15:00 IST, the summer peak hour
    era5_rh_mean            mean relative humidity
    era5_rh_15h             afternoon humidity (pairs with t_15h)
    era5_dewpoint_mean      absolute moisture - pre-monsoon "sticky" heat
    era5_apparent_max       Steadman "feels-like" maximum
    era5_heat_index_max     NOAA heat index maximum (see heat_index_c)
    era5_cdh24              cooling degree-HOURS above 24 C, i.e. how much
                            cooling the day needs: 24 h at 34 C = 240 C-h
    era5_hours_ge_40        hours >= 40 C (extreme sensible heat)
    era5_hours_hi_ge_41     hours with heat index >= 41 C (NWS "danger")
    era5_precip_sum         mm/day; monsoon rain cools AND stops irrigation
                            pumps (agricultural load ~18 % of Indian demand)
    era5_rain_hours         hours with >= 0.5 mm
    era5_cloud_mean         % cloud cover
    era5_solar_kwh_m2       daily irradiation; rooftop solar hides demand
    era5_wind_mean          ventilation / desert-cooler effectiveness
    """
    d = df.copy()
    d["date"] = d["datetime"].dt.normalize()
    d["hour"] = d["datetime"].dt.hour
    d["heat_index"] = heat_index_c(d["temperature_2m"].to_numpy(),
                                   d["relative_humidity_2m"].to_numpy())
    d["cdh24"] = (d["temperature_2m"] - 24).clip(lower=0)
    g = d.groupby("date")
    out = pd.DataFrame({
        "era5_tmax": g["temperature_2m"].max(),
        "era5_tmean": g["temperature_2m"].mean(),
        "era5_tmin": g["temperature_2m"].min(),
        "era5_rh_mean": g["relative_humidity_2m"].mean(),
        "era5_dewpoint_mean": g["dew_point_2m"].mean(),
        "era5_apparent_max": g["apparent_temperature"].max(),
        "era5_heat_index_max": g["heat_index"].max(),
        "era5_cdh24": g["cdh24"].sum(),
        "era5_hours_ge_40": g["temperature_2m"].apply(lambda x: (x >= 40).sum()),
        "era5_hours_hi_ge_41": g["heat_index"].apply(lambda x: (x >= 41).sum()),
        "era5_precip_sum": g["precipitation"].sum(),
        "era5_rain_hours": g["precipitation"].apply(lambda x: (x >= 0.5).sum()),
        "era5_cloud_mean": g["cloud_cover"].mean(),
        "era5_solar_kwh_m2": g["shortwave_radiation"].sum() / 1000.0,
        "era5_wind_mean": g["wind_speed_10m"].mean(),
    })
    night = d[d["hour"] <= 6].groupby("date")["temperature_2m"].min()
    at15 = d[d["hour"] == 15].set_index("date")
    out["era5_night_tmin"] = night
    out["era5_t_15h"] = at15["temperature_2m"]
    out["era5_rh_15h"] = at15["relative_humidity_2m"]
    out.index.name = "date"
    return out.reset_index()


def era5_state_daily() -> pd.DataFrame:
    """Daily ERA5 features for every state (long format)."""
    frames = []
    for s in C.STATE_NAMES:
        hourly = read_era5_state(s)
        daily = aggregate_era5_daily(hourly)
        daily.insert(1, "state", s)
        frames.append(daily)
    return pd.concat(frames, ignore_index=True)


def era5_hourly_long() -> pd.DataFrame:
    """All states' raw hourly ERA5 rows stacked (for the interim folder)."""
    return pd.concat([read_era5_state(s) for s in C.STATE_NAMES], ignore_index=True)
