"""
data_download.py - reproducible downloaders for every raw dataset.

All sources are public and free, and none needs an account:

LOAD (electricity demand)
-------------------------
1. Kaggle "Hourly Load India - Electrical Load Forecasting"
   (shubhamvashisht) - hourly All-India + 5 regional grid demand in MW,
   2019-01-01 .. 2024-04-30 (46,728 hourly rows). Compiled from Grid-India.
2. Kaggle "India Power Grid - NLDC Daily PSP Reports" (halcyonvector),
   scraped from Grid-India's daily Power Supply Position reports:
     study1_hourly.csv  46,728 rows x 151 cols (hourly load + daily grid ops)
     study1_daily.csv   ~2,760 rows x 144 cols (generation mix, outages ...)
     study2_scada.csv   15-minute NLDC SCADA demand, 2024-11 .. present
     study3_states.csv  daily state-wise max demand met, 2018-12 .. present
3. Mendeley Data "Electricity Demand, Solar and Wind Generation Data of India
   at 1-hour interval" (NERLDC / Grid-India + IIT Guwahati, CC-BY-4.0),
   file "January 2024- June 2025.xlsx": hourly All-India demand. This file
   bridges the May-Oct 2024 gap between sources 1 and 2(SCADA) - and that
   gap contains the record heatwave of May-June 2024, so it is essential.

WEATHER
-------
4. IMD gridded daily Tmax / Tmin, 1.0 x 1.0 degree, 1991-2025 (India
   Meteorological Department, Pune; Srivastava et al. 2009), via `imdlib`.
   This is the official Government of India dataset that IMD itself uses
   for heatwave climatology -> heatwave labels are "official", not ad hoc.
5. ERA5 reanalysis (ECMWF / Copernicus) HOURLY weather at each state's main
   load centre, via the free Open-Meteo archive API. IMD gives only daily
   max/min temperature; ERA5 adds humidity, night-time heat, rain, cloud
   and solar radiation at hourly resolution.

Every function is idempotent: if the file already exists it is skipped, so
re-running the notebook never re-downloads hundreds of MB.
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from . import config as C

KAGGLE_API = "https://www.kaggle.com/api/v1/datasets/download/{slug}"
MENDELEY_BRIDGE_URL = (
    "https://data.mendeley.com/public-files/datasets/y58jknpgs8/files/"
    "c67cbcfa-f33b-49c0-925d-d6e1cda3a9c1/file_downloaded"
)
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def _download_kaggle_zip(slug: str, out_dir: Path) -> list[Path]:
    """Download a public Kaggle dataset zip (no API key needed for public
    datasets) and extract it into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        print(f"[skip] {slug}: already present in {out_dir}")
        return sorted(out_dir.iterdir())
    print(f"[get ] Kaggle {slug} ...")
    r = requests.get(KAGGLE_API.format(slug=slug), timeout=600)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(out_dir)
    return sorted(out_dir.iterdir())


def download_load_datasets() -> None:
    """Fetch the three electricity-load sources (see module docstring)."""
    _download_kaggle_zip("shubhamvashisht/hourly-load-india-electrical-load-forecasting",
                         C.RAW_KAGGLE_HOURLY_DIR)
    _download_kaggle_zip("halcyonvector/india-power-grid-nldc-daily-psp-reports",
                         C.RAW_PSP_DIR)
    C.RAW_MENDELEY_DIR.mkdir(parents=True, exist_ok=True)
    target = C.RAW_MENDELEY_DIR / "January 2024- June 2025.xlsx"
    if target.exists():
        print(f"[skip] Mendeley bridge file present: {target.name}")
    else:
        print("[get ] Mendeley NERLDC hourly All-India demand Jan-2024..Jun-2025 ...")
        r = requests.get(MENDELEY_BRIDGE_URL, timeout=600)
        r.raise_for_status()
        target.write_bytes(r.content)


def download_imd_gridded(start_year: int = C.CLIMATE_NORMAL_START,
                         end_year: int = 2025) -> None:
    """
    Download IMD 1-degree daily Tmax and Tmin binary (.GRD) files.

    Why from 1991 and not just 2019? Heatwave criteria need the
    "departure from NORMAL" - the 30-year (1991-2020) average Tmax for that
    calendar day at that grid cell. 40 C in Delhi in mid-May is ordinary
    (normal ~39.5 C); 40 C in Bengaluru would be +6.5 C above normal.
    So we need 30 years of history even though we model only 2019-2025.
    """
    import imdlib as imd

    for var in ("tmax", "tmin"):
        var_dir = C.RAW_IMD_DIR / var
        missing = [y for y in range(start_year, end_year + 1)
                   if not (var_dir / f"{y}.GRD").exists()]
        for y in missing:
            for attempt in range(4):
                try:
                    imd.get_data(var, y, y, fn_format="yearwise", file_dir=str(C.RAW_IMD_DIR))
                    break
                except Exception as exc:  # network hiccups on imdpune.gov.in
                    print(f"  retry {var} {y} ({exc!r}); attempt {attempt + 1}")
                    time.sleep(10 * (attempt + 1))
        print(f"[ok  ] IMD {var}: {len(list(var_dir.glob('*.GRD')))} yearly files")


def download_era5_hourly(sleep_s: float = 20.0) -> None:
    """
    Download hourly ERA5 weather for each state's load centre.

    The free Open-Meteo tier is rate-limited (it weighs a 7-year x 8-variable
    request as ~150 "calls"); we therefore pause between states and back off
    on HTTP 429. Output: one CSV per state, ~61,000 hourly rows x 9 columns.
    Times are requested in Asia/Kolkata (IST) so that "hour 15" is 3 pm local,
    matching how Grid-India timestamps its load data.
    """
    for s in C.STATES:
        out = C.RAW_ERA5_DIR / f"era5_hourly_{s.imd_name.replace(' ', '_')}.csv"
        if out.exists():
            continue
        params = {
            "latitude": s.lat, "longitude": s.lon,
            "start_date": C.ERA5_START, "end_date": C.ERA5_END,
            "hourly": ",".join(C.ERA5_HOURLY_VARS),
            "timezone": "Asia/Kolkata",
        }
        for attempt in range(8):
            r = requests.get(OPEN_METEO_ARCHIVE, params=params, timeout=300)
            if r.status_code == 429:  # quota exceeded -> wait and retry
                wait = 90 * (attempt + 1)
                print(f"  429 for {s.imd_name}; sleeping {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            break
        else:
            raise RuntimeError(f"Open-Meteo kept refusing {s.imd_name}")
        js = r.json()
        df = pd.DataFrame(js["hourly"]).rename(columns={"time": "datetime"})
        df.insert(1, "state", s.imd_name)
        df.insert(2, "grid_lat", js["latitude"])
        df.insert(3, "grid_lon", js["longitude"])
        df.insert(4, "elevation_m", js.get("elevation"))
        df.to_csv(out, index=False)
        print(f"[ok  ] ERA5 {s.imd_name:<45} {len(df):>6} rows")
        time.sleep(sleep_s)


def write_state_reference_table() -> Path:
    """Persist the state metadata (load centre, grid region) as a CSV so the
    raw folder is self-describing."""
    rows = [dict(state=s.imd_name, nldc_names="|".join(s.nldc_names),
                 grid_region=s.grid_region, load_centre=s.city, lat=s.lat, lon=s.lon)
            for s in C.STATES]
    out = C.RAW_REFERENCE_DIR / "state_metadata.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def download_all() -> None:
    write_state_reference_table()
    download_load_datasets()
    download_imd_gridded()
    download_era5_hourly()


if __name__ == "__main__":
    download_all()
