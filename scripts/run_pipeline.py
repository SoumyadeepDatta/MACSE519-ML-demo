"""
run_pipeline.py - reproduce the whole study end-to-end from the command line.

    python scripts/run_pipeline.py                # all stages
    python scripts/run_pipeline.py --stages load weather

The notebooks in notebooks/ walk through the SAME functions step by step with
plots and explanations; this script is the "one button" version used to
regenerate every file in data/interim, data/processed, models/ and reports/.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from heatwave_peak import clustering as CL
from heatwave_peak import config as C
from heatwave_peak import data_download as DL
from heatwave_peak import features as F
from heatwave_peak import heatwave as HW
from heatwave_peak import load_processing as LP
from heatwave_peak import weather_processing as WP
from heatwave_peak import experiment as EX


def stage_download():
    DL.download_all()


def stage_load():
    hourly, calib = LP.build_national_hourly()
    hourly.to_parquet(C.INTERIM_DIR / "load_hourly_national_2019_2025.parquet")
    LP.build_regional_hourly().to_parquet(C.INTERIM_DIR / "load_hourly_regional_2019_2024.parquet")
    daily = LP.add_critical_peak_flag(LP.build_daily_peaks(hourly))
    calib["psp_to_hourly_peak_ratio"] = daily.attrs["psp_to_hourly_peak_ratio"]
    daily.attrs = {}
    daily.to_parquet(C.PROCESSED_DIR / "daily_peak_national.parquet")
    sd = LP.load_state_daily_demand()
    sd.to_parquet(C.INTERIM_DIR / "state_daily_demand.parquet")
    LP.state_demand_weights(sd).to_csv(C.PROCESSED_DIR / "state_demand_weights.csv")
    (C.METRICS_DIR / "load_calibration.json").write_text(json.dumps(calib, indent=2))
    print("[load] done", calib)


def stage_weather():
    imd = WP.imd_state_daily()
    hw = HW.add_heatwave_features(imd)
    era5 = WP.era5_state_daily()
    sw = hw.merge(era5, on=["date", "state"], how="left")
    sw.to_parquet(C.INTERIM_DIR / "state_daily_weather_1991_2025.parquet")
    WP.era5_hourly_long().to_parquet(C.INTERIM_DIR / "era5_hourly_states_2018_2025.parquet")
    print("[weather] state-days:", len(sw))


def stage_cluster():
    sw = pd.read_parquet(C.INTERIM_DIR / "state_daily_weather_1991_2025.parquet")
    sd = pd.read_parquet(C.INTERIM_DIR / "state_daily_demand.parquet")
    w = pd.read_csv(C.PROCESSED_DIR / "state_demand_weights.csv", index_col=0)["demand_weight"]
    sig = CL.build_state_signatures(sw, sd)
    X, cols = CL.scale_signatures(sig)
    scores = CL.compare_clusterings(X, sig)
    best = CL.select_clustering(scores)
    labels = best["labels"]
    names = CL.name_zones(sig, labels, w)
    zones = sig.assign(zone_id=labels, zone=[names[int(z)] for z in labels])
    zones.to_csv(C.PROCESSED_DIR / "state_signatures_and_zones.csv")
    scores.drop(columns="labels").to_csv(C.METRICS_DIR / "clustering_model_selection.csv", index=False)
    print("[cluster] best:", best["algorithm"], "k =", best["k"],
          "silhouette = %.3f" % best["silhouette"], "ARI vs grid = %.2f" % best["ari_vs_grid"])
    print(zones.groupby("zone").apply(lambda g: ", ".join(g.index)).to_string())


def stage_features():
    sw = pd.read_parquet(C.INTERIM_DIR / "state_daily_weather_1991_2025.parquet")
    sw = sw[(sw["date"] >= "2018-12-01") & (sw["date"] <= C.STUDY_END)]
    w = pd.read_csv(C.PROCESSED_DIR / "state_demand_weights.csv", index_col=0)["demand_weight"]
    zones = pd.read_csv(C.PROCESSED_DIR / "state_signatures_and_zones.csv", index_col=0)
    zone_of = zones["zone_id"]
    names = zones.drop_duplicates("zone_id").set_index("zone_id")["zone"].to_dict()
    national_heat = HW.national_heat_exposure(sw, w)
    zone_wide = CL.zone_daily_features(sw, zone_of, w, names)
    nat_w = CL.zone_daily_features(sw, pd.Series(0, index=zone_of.index), w, {0: "IN"})
    nat_w = nat_w.drop(columns=[c for c in nat_w.columns if c.endswith("heat_exposure")])
    daily = pd.read_parquet(C.PROCESSED_DIR / "daily_peak_national.parquet")
    psp = LP.read_psp_daily()
    tab = F.build_model_table(daily, national_heat, zone_wide, nat_w, psp)
    tab.to_parquet(C.PROCESSED_DIR / "model_table.parquet")
    national_heat.to_parquet(C.PROCESSED_DIR / "national_heat_exposure.parquet")
    print("[features] table", tab.shape, "features:", len(F.feature_columns(tab)))


def stage_train():
    EX.run_full_experiment(verbose=True)


def stage_evidence():
    """Significance tests and ablations behind the novelty claims
    (notebook 08 runs the same functions with explanations)."""
    from heatwave_peak import evidence as EV
    EV.exp_significance()
    EV.exp_frontier()
    EV.exp_frontier_bootstrap()
    EV.exp_component_contrast()
    EV.exp_feature_ablation()
    EV.ablation_bootstrap()
    EV.exp_spatial_ablation()
    EV.exp_level_shift()
    EV.exp_metric_disagreement()
    EV.exp_lightweight()
    print("[evidence] tables written to reports/metrics/evidence_*.csv")


STAGES = {"download": stage_download, "load": stage_load, "weather": stage_weather,
          "cluster": stage_cluster, "features": stage_features, "train": stage_train,
          "evidence": stage_evidence}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=[s for s in STAGES if s != "download"])
    args = ap.parse_args()
    for s in args.stages:
        t = time.time()
        STAGES[s]()
        print(f"  ({s}: {time.time() - t:.1f}s)")
