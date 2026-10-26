"""
config.py - single source of truth for paths, dates and domain constants.

WHY a central config?
    A research pipeline is only reproducible if every "magic number" lives in
    one auditable place. If a reviewer asks "why is the test year 2025?" or
    "why 40 degC?", the answer (with the reasoning) is here, not scattered
    across seven notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ----------------------------------------------------------------------------
# 1. PATHS
# ----------------------------------------------------------------------------
# PROJECT_ROOT is resolved from this file's location (src/heatwave_peak/), so
# notebooks and scripts work no matter which folder they are launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # untouched downloads (never edited)
INTERIM_DIR = DATA_DIR / "interim"      # cleaned, per-source tables
PROCESSED_DIR = DATA_DIR / "processed"  # final model-ready tables

RAW_LOAD_DIR = RAW_DIR / "load"
RAW_KAGGLE_HOURLY_DIR = RAW_LOAD_DIR / "kaggle_hourly_load_india"
RAW_PSP_DIR = RAW_LOAD_DIR / "nldc_psp_reports"
RAW_MENDELEY_DIR = RAW_LOAD_DIR / "mendeley_nerldc_hourly"

RAW_WEATHER_DIR = RAW_DIR / "weather"
RAW_IMD_DIR = RAW_WEATHER_DIR / "imd_gridded"
RAW_ERA5_DIR = RAW_WEATHER_DIR / "era5_openmeteo_hourly"
RAW_REFERENCE_DIR = RAW_DIR / "reference"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

for _d in (INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR,
           RAW_IMD_DIR, RAW_ERA5_DIR, RAW_REFERENCE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 2. STUDY PERIOD AND CHRONOLOGICAL SPLITS
# ----------------------------------------------------------------------------
# Hourly load starts 2019-01-01. IMD's gridded archive is published up to the
# last complete year (2025 at the time of writing; 2026 returns an empty file),
# so the modelling window is 2019-01-01 .. 2025-12-31 (7 full years).
STUDY_START = "2019-01-01"
STUDY_END = "2025-12-31"

# Time-series data must be split by time, never randomly. A random split would
# put 31-May-2024 in train and 30-May-2024 in test: the model would "see the
# future" of the same heatwave and look far better than it is.
#
#   TRAIN      2019-01-01 .. 2023-12-31   5 years, includes 2019/2022/2023 heat
#   VALIDATION 2024-01-01 .. 2024-12-31   record heat year (250 GW on 30-May-24)
#                                         -> used ONLY for early stopping and
#                                            for tuning the loss parameters
#   TEST       2025-01-01 .. 2025-12-31   untouched hold-out, reported once
TRAIN_END = "2023-12-31"
VALID_START, VALID_END = "2024-01-01", "2024-12-31"
TEST_START, TEST_END = "2025-01-01", "2025-12-31"

# COVID-19 national lockdown: demand fell ~25 % for reasons unrelated to
# weather (factories shut). Training on it would teach the model that hot
# April-May days have LOW demand. We keep the rows for lag continuity but give
# them zero training weight. (Lockdown 1.0 began 25-Mar-2020; Unlock-1 began
# 1-Jun-2020; we extend to 30-Jun to cover the slow industrial restart.)
COVID_LOCKDOWN = ("2020-03-22", "2020-06-30")

# Normal period for "departure from normal" temperatures. WMO and IMD use
# 30-year normals; 1991-2020 is the current WMO standard normal.
CLIMATE_NORMAL_START, CLIMATE_NORMAL_END = 1991, 2020

# ----------------------------------------------------------------------------
# 3. IMD HEATWAVE CRITERIA  (IMD Heat/Cold-wave FAQ, mausam.imd.gov.in)
# ----------------------------------------------------------------------------
# Gate 1 (entry): Tmax >= 40 C plains, >= 37 C coastal, >= 30 C hills.
# Gate 2 (either):  departure from normal >= 4.5 C  (severe if > 6.4 C)
#                   OR absolute Tmax >= 45 C         (severe if >= 47 C)
# Declaration rule: criteria must hold for 2 consecutive days.
# These are applied per 1-degree grid cell by imdlib, then aggregated to
# states (see heatwave.py).
IMD_HW_DEPARTURE = 4.5
IMD_SEVERE_DEPARTURE = 6.5
IMD_PLAINS_ENTRY, IMD_COASTAL_ENTRY, IMD_HILL_ENTRY = 40.0, 37.0, 30.0

# A state is "under heatwave" on a day if at least this fraction of its land
# area meets the IMD cell criteria. Example: Rajasthan (~34 one-degree cells);
# a single hot cell near Barmer (~3 %) is a local hot spot, not a state-wide
# heatwave that pushes Jaipur's and Jodhpur's ACs to the limit. 25 % of the
# area is a common choice in Indian heatwave climatology studies (it mirrors
# IMD's requirement that >= 2 stations in a sub-division meet the criteria).
STATE_HW_AREA_FRACTION = 0.25

# National "heatwave regime" day (used for the regime-specific metrics):
# at least this share of India's electricity demand sits in states under a
# heatwave (IMD-declared OR Excess-Heat-Factor > 0, see heatwave.py).
# 20 % ~ the demand of UP + Rajasthan + MP together, i.e. one large heat
# dome over the North-West / Indo-Gangetic plain. Notebook 03 shows the
# sensitivity: 10 %..30 % gives 280..150 regime days; 20 % keeps ~8 % of days
# (rare, as heatwaves are) while leaving 14 days in the 2025 test year.
NATIONAL_HW_EXPOSURE_THRESHOLD = 0.20

# ----------------------------------------------------------------------------
# 4. "CRITICAL PEAK" DAYS
# ----------------------------------------------------------------------------
# Indian demand grows ~6-8 %/year, so a fixed MW threshold (e.g. 200 GW) would
# label almost no 2019 days and almost all 2025 days. Instead, a day is a
# critical peak day if its peak exceeds the 90th percentile of the PREVIOUS
# 365 days' peaks (growth-adjusted and causal - only past data is used).
CRITICAL_PEAK_WINDOW_DAYS = 365
CRITICAL_PEAK_QUANTILE = 0.90

# ----------------------------------------------------------------------------
# 5. STYLISED COST OF FORECAST ERRORS (Indian power market, 2023-25)
# ----------------------------------------------------------------------------
# Under-forecast: the shortfall must be bought in IEX's Real-Time Market,
#   which hits the CERC price cap of Rs 10/kWh on hot evenings, or load is shed.
# Over-forecast: surplus scheduled thermal is backed down; DISCOMs still pay
#   capacity charges, roughly Rs 2.5/kWh of stranded energy.
# Ratio ~4:1 -> the economic reason the loss must be asymmetric.
# An error persists for roughly the 4-hour evening-peak window.
COST_UNDER_RS_PER_KWH = 10.0
COST_OVER_RS_PER_KWH = 2.5
PEAK_WINDOW_HOURS = 4.0

RANDOM_SEED = 42


@dataclass(frozen=True)
class StateMeta:
    """
    Static metadata for one Indian state / UT that is connected to the grid.

    Attributes
    ----------
    imd_name : name as used by imdlib's bundled state boundaries.
    nldc_names : the entity names NLDC uses in its daily PSP state table.
        e.g. NLDC writes "UP", IMD writes "Uttar Pradesh". DVC (Damodar
        Valley Corporation) serves the coal/steel belt of Jharkhand (and a
        part of West Bengal) - we attach it to Jharkhand because most of
        its command area and load is there.
    grid_region : official Grid-India regional grid (NR/WR/SR/ER/NER) -
        used only to VALIDATE the data-driven clusters, never as an input.
    city, lat, lon : main load centre used for the ERA5 hourly time series.
        One point per state keeps the download within the free API quota;
        the IMD gridded data (area-averaged over the whole state) provides
        the spatially complete temperature signal.
    """
    imd_name: str
    nldc_names: tuple
    grid_region: str
    city: str
    lat: float
    lon: float


STATES: list[StateMeta] = [
    # Northern Region
    StateMeta("Delhi", ("Delhi",), "NR", "New Delhi", 28.61, 77.21),
    StateMeta("Haryana", ("Haryana",), "NR", "Rohtak", 28.89, 76.61),
    StateMeta("Punjab", ("Punjab",), "NR", "Ludhiana", 30.90, 75.85),
    StateMeta("Rajasthan", ("Rajasthan",), "NR", "Jaipur", 26.91, 75.79),
    StateMeta("Uttar Pradesh", ("UP",), "NR", "Lucknow", 26.85, 80.95),
    StateMeta("Uttarakhand", ("Uttarakhand",), "NR", "Dehradun", 30.32, 78.03),
    StateMeta("Himachal Pradesh", ("HP",), "NR", "Shimla", 31.10, 77.17),
    StateMeta("Jammu and Kashmir", ("J&K(UT) & Ladakh(UT)",), "NR", "Jammu", 32.73, 74.86),
    StateMeta("Chandigarh", ("Chandigarh",), "NR", "Chandigarh", 30.73, 76.78),
    # Western Region
    StateMeta("Gujarat", ("Gujarat",), "WR", "Ahmedabad", 23.02, 72.57),
    StateMeta("Maharashtra", ("Maharashtra",), "WR", "Pune", 18.52, 73.86),
    StateMeta("Madhya Pradesh", ("MP",), "WR", "Bhopal", 23.26, 77.41),
    StateMeta("Chhattisgarh", ("Chhattisgarh",), "WR", "Raipur", 21.25, 81.63),
    StateMeta("Goa", ("Goa",), "WR", "Panaji", 15.50, 73.83),
    StateMeta("The Dadra and Nagar Haveli and Daman and Diu",
              ("DD", "DNH", "DNHDDPDCL"), "WR", "Silvassa", 20.27, 73.01),
    # Southern Region
    StateMeta("Andhra Pradesh", ("Andhra Pradesh",), "SR", "Vijayawada", 16.51, 80.65),
    StateMeta("Telangana", ("Telangana",), "SR", "Hyderabad", 17.39, 78.49),
    StateMeta("Karnataka", ("Karnataka",), "SR", "Bengaluru", 12.97, 77.59),
    StateMeta("Tamil Nadu", ("Tamil Nadu",), "SR", "Chennai", 13.08, 80.27),
    StateMeta("Kerala", ("Kerala",), "SR", "Kochi", 9.93, 76.27),
    StateMeta("Puducherry", ("Puducherry",), "SR", "Puducherry", 11.94, 79.81),
    # Eastern Region
    StateMeta("Bihar", ("Bihar",), "ER", "Patna", 25.59, 85.14),
    StateMeta("Jharkhand", ("Jharkhand", "DVC"), "ER", "Ranchi", 23.34, 85.31),
    StateMeta("Odisha", ("Odisha",), "ER", "Bhubaneswar", 20.30, 85.82),
    StateMeta("West Bengal", ("West Bengal",), "ER", "Kolkata", 22.57, 88.36),
    StateMeta("Sikkim", ("Sikkim",), "ER", "Gangtok", 27.33, 88.61),
    # North-Eastern Region
    StateMeta("Assam", ("Assam",), "NER", "Guwahati", 26.14, 91.74),
    StateMeta("Meghalaya", ("Meghalaya",), "NER", "Shillong", 25.58, 91.89),
    StateMeta("Tripura", ("Tripura",), "NER", "Agartala", 23.83, 91.29),
    StateMeta("Manipur", ("Manipur",), "NER", "Imphal", 24.82, 93.94),
    StateMeta("Mizoram", ("Mizoram",), "NER", "Aizawl", 23.73, 92.72),
    StateMeta("Nagaland", ("Nagaland",), "NER", "Kohima", 25.67, 94.11),
    StateMeta("Arunachal Pradesh", ("Arunachal Pradesh",), "NER", "Itanagar", 27.08, 93.61),
]

STATE_NAMES = [s.imd_name for s in STATES]

# ERA5 hourly variables requested from Open-Meteo's archive API.
# Each one is there for a physical reason:
#   temperature_2m        -> sensible heat, drives AC compressor load
#   relative_humidity_2m  -> humid heat (Chennai at 36 C/70 % RH feels worse
#                            than Jaipur at 42 C/15 %), basis of heat index
#   dew_point_2m          -> absolute moisture; night-time discomfort
#   apparent_temperature  -> "feels-like" temperature (Steadman), human comfort
#   precipitation         -> monsoon rain cools and cuts irrigation pumping
#   cloud_cover           -> clouds cut both heat and rooftop-solar output
#   shortwave_radiation   -> solar irradiance -> behind-the-meter solar
#   wind_speed_10m        -> ventilation / evaporative cooling (desert coolers)
ERA5_HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "precipitation", "cloud_cover",
    "shortwave_radiation", "wind_speed_10m",
]
# One month of lead-in so lagged/rolling weather features exist on 2019-01-01.
ERA5_START, ERA5_END = "2018-12-01", STUDY_END
