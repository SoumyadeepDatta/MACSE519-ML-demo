"""
heatwave_peak
=============

Heatwave-Conditioned Peak Electricity Demand Forecasting Using Asymmetric
Loss Optimization (Indian national grid, 2019-2025).

Package layout (one module per pipeline stage, so every notebook can import
exactly the logic it explains):

    config.py            -> every path, date boundary and constant in ONE place
    data_download.py     -> reproducible downloaders for all raw datasets
    load_processing.py   -> clean + stitch hourly load, extract daily peaks
    weather_processing.py-> IMD gridded Tmax/Tmin + ERA5 hourly -> state-day table
    heatwave.py          -> IMD heatwave rules, warm nights, heat persistence
    clustering.py        -> group states into thermal-demand zones
    features.py          -> leakage-safe day-ahead feature matrix
    losses.py            -> the novelty: Severity-Adaptive Huberised LINEX objective
    models.py            -> baselines + proposed LightGBM training helpers
    metrics.py           -> accuracy, peak-risk and rupee-cost metrics
"""

__version__ = "1.0.0"
