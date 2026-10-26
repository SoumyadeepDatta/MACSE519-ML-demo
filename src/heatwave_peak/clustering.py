"""
clustering.py - group the 33 grid-connected states/UTs into a few
"thermal-demand zones" and build zone-level daily weather signals.

Why cluster at all?
-------------------
The target is ONE number per day (All-India peak), but the weather comes from
33 states. Two naive options both fail:
  * 33 x ~30 weather columns = ~1,000 features for ~1,800 training days ->
    the trees would overfit noise (e.g. "Mizoram humidity" splits).
  * One national average -> a heatwave over Rajasthan and a cool, rainy
    Kerala average out to "normal", hiding exactly the regional extremes
    that drive the peak.
Clustering the states into a handful of climatically AND behaviourally
similar zones keeps the regional contrast with only K x ~15 features.

Why not just use the 5 official grid regions (NR/WR/SR/ER/NER)?
---------------------------------------------------------------
Grid regions are administrative/electrical, not climatic. Example: NR
contains both desert Rajasthan (dry heat, 47 C) and Himachal Pradesh (snow,
winter-peaking demand); WR mixes humid coastal Goa with semi-arid MP. A
heatwave signal averaged over NR is diluted by Himachal. We still compare
our zones with the grid regions (Adjusted Rand Index) as a sanity check.

What each state is described by (its "signature")
------------------------------------------------
  Climate (IMD 1991-2020): annual mean Tmax, seasonal amplitude, pre-monsoon
      Tmax, diurnal range, T95, heat-day frequency, month of hottest Tmax.
  Humidity/rain (ERA5 2019-2023): mean RH, pre-monsoon dew point, share of
      rain falling in Jun-Sep, mean afternoon heat index in Apr-Jun.
  Demand behaviour (NLDC 2019-2023): summer/winter demand ratio, demand
      sensitivity to heat (% demand change per +1 C Tmax in Mar-Jun),
      monsoon dip (Jul-Sep vs Apr-Jun demand), month of peak demand.
  Geography: latitude, longitude (so zones come out spatially coherent).
Only training-period years are used -> no information from 2024/2025 leaks
into the zone definitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from . import config as C


def _circular(month: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Encode a month (1-12) on a circle so December (12) is next to
    January (1); a plain number would put them 11 units apart."""
    ang = 2 * np.pi * (month - 1) / 12
    return np.sin(ang), np.cos(ang)


def build_state_signatures(state_weather: pd.DataFrame,
                           state_demand: pd.DataFrame) -> pd.DataFrame:
    """
    One row per state with climate, humidity and demand-behaviour features.

    Parameters
    ----------
    state_weather : long (date, state) table with IMD + ERA5 + heatwave cols.
    state_demand  : long (date, state, max_demand_mw) table from NLDC.
    """
    sw = state_weather.copy()
    sw["month"] = sw["date"].dt.month
    clim = sw[(sw["date"].dt.year >= C.CLIMATE_NORMAL_START) &
              (sw["date"].dt.year <= C.CLIMATE_NORMAL_END)]
    train = sw[(sw["date"] >= C.STUDY_START) & (sw["date"] <= C.TRAIN_END)]

    mon_tmax = clim.groupby(["state", "month"])["imd_tmax"].mean().unstack()
    sig = pd.DataFrame(index=sorted(sw["state"].unique()))
    sig["tmax_annual_mean"] = clim.groupby("state")["imd_tmax"].mean()
    sig["tmax_seasonal_amplitude"] = mon_tmax.max(axis=1) - mon_tmax.min(axis=1)
    sig["tmax_premonsoon_AMJ"] = clim[clim["month"].isin([4, 5, 6])].groupby("state")["imd_tmax"].mean()
    sig["diurnal_range"] = (clim["imd_tmax"] - clim["imd_tmin"]).groupby(clim["state"]).mean()
    sig["dmt_p95"] = clim.assign(d=(clim["imd_tmax"] + clim["imd_tmin"]) / 2).groupby("state")["d"].quantile(0.95)
    years = clim["date"].dt.year.nunique()
    sig["heat_days_per_year"] = clim.groupby("state")["state_heat_day"].sum() / years
    sig["hot_month_sin"], sig["hot_month_cos"] = _circular(mon_tmax.idxmax(axis=1))

    if "era5_rh_mean" in train.columns:
        sig["rh_mean"] = train.groupby("state")["era5_rh_mean"].mean()
        amj = train[train["month"].isin([4, 5, 6])]
        sig["dewpoint_premonsoon"] = amj.groupby("state")["era5_dewpoint_mean"].mean()
        sig["heat_index_AMJ"] = amj.groupby("state")["era5_heat_index_max"].mean()
        rain = train.groupby(["state", "month"])["era5_precip_sum"].sum().unstack()
        sig["monsoon_rain_share"] = rain[[6, 7, 8, 9]].sum(axis=1) / rain.sum(axis=1)

    # ---- demand behaviour ------------------------------------------------
    sd = state_demand[(state_demand["date"] >= C.STUDY_START) &
                      (state_demand["date"] <= C.TRAIN_END)].copy()
    sd = sd[~sd["date"].between(*C.COVID_LOCKDOWN)]
    sd["month"] = sd["date"].dt.month
    # Normalise each state's demand by its own 30-day rolling mean so that
    # 6-8 %/yr growth is not mistaken for seasonality.
    sd = sd.sort_values(["state", "date"])
    sd["rel"] = sd["max_demand_mw"] / sd.groupby("state")["max_demand_mw"].transform(
        lambda s: s.rolling(365, min_periods=60, center=True).mean())
    mon_dem = sd.groupby(["state", "month"])["rel"].mean().unstack()
    sig["summer_winter_ratio"] = mon_dem[[4, 5, 6]].mean(axis=1) / mon_dem[[12, 1, 2]].mean(axis=1)
    sig["monsoon_dip"] = mon_dem[[7, 8, 9]].mean(axis=1) / mon_dem[[4, 5, 6]].mean(axis=1)
    sig["peak_demand_month_sin"], sig["peak_demand_month_cos"] = _circular(mon_dem.idxmax(axis=1))

    # Heat sensitivity: OLS slope of relative demand on Tmax in Mar-Jun.
    # Example: Delhi ~ +2 %/C (AC-dominated), Himachal ~ 0 %/C.
    m = sd.merge(state_weather[["date", "state", "imd_tmax"]], on=["date", "state"])
    m = m[m["month"].isin([3, 4, 5, 6])]

    def _slope(g):
        x, y = g["imd_tmax"].to_numpy(), g["rel"].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        return np.polyfit(x[ok], y[ok], 1)[0] * 100 if ok.sum() > 30 else np.nan

    sig["heat_sensitivity_pct_per_C"] = m.groupby("state")[["imd_tmax", "rel"]].apply(_slope)

    meta = pd.DataFrame([{"state": s.imd_name, "lat": s.lat, "lon": s.lon,
                          "grid_region": s.grid_region} for s in C.STATES]).set_index("state")
    sig = sig.join(meta)
    return sig


def scale_signatures(sig: pd.DataFrame, geo_weight: float = 1.5):
    """
    Standardise features; up-weight lat/lon by `geo_weight`.

    Why up-weight geography? With pure climate features, Goa (humid coast,
    WR) and Kerala (humid coast, SR) may cluster together - climatically
    fair, but the zone would then mix two weather systems 600 km apart
    that do not experience the same heatwave on the same day. A modest
    geographic weight keeps zones spatially coherent (heatwaves are
    synoptic-scale, 500-1500 km) without overriding climate.
    """
    feat = sig.drop(columns=["grid_region"]).astype(float)
    feat = feat.fillna(feat.median())
    X = StandardScaler().fit_transform(feat)
    geo_idx = [feat.columns.get_loc(c) for c in ("lat", "lon")]
    X[:, geo_idx] *= geo_weight
    return X, feat.columns.tolist()


def compare_clusterings(X: np.ndarray, sig: pd.DataFrame, k_range=range(3, 9)) -> pd.DataFrame:
    """
    Fit 4 algorithms x several K and score them.

    Algorithms
      kmeans          spherical clusters, the default baseline
      ward            agglomerative, minimises within-cluster variance;
                      deterministic and good for small n (33 states)
      ward_spatial    Ward + connectivity graph (each state may only merge
                      with its 4 geographic nearest neighbours) -> zones
                      are contiguous by construction, like real regions
      gmm             Gaussian mixture (diagonal covariance), soft clusters
    Scores
      silhouette (higher better, -1..1), Calinski-Harabasz (higher better),
      Davies-Bouldin (lower better), ARI/NMI vs official grid regions
      (agreement, informative only - we do not want to copy them).
    """
    latlon = sig[["lat", "lon"]].to_numpy()
    conn = kneighbors_graph(latlon, n_neighbors=4, include_self=False)
    rows = []
    for k in k_range:
        models = {
            "kmeans": KMeans(n_clusters=k, n_init=50, random_state=C.RANDOM_SEED),
            "ward": AgglomerativeClustering(n_clusters=k, linkage="ward"),
            "ward_spatial": AgglomerativeClustering(n_clusters=k, linkage="ward", connectivity=conn),
            "gmm": GaussianMixture(n_components=k, covariance_type="diag", n_init=10,
                                   random_state=C.RANDOM_SEED),
        }
        for name, mdl in models.items():
            labels = mdl.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            sizes = np.bincount(labels)
            rows.append({
                "algorithm": name, "k": k,
                "silhouette": silhouette_score(X, labels),
                "calinski_harabasz": calinski_harabasz_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels),
                "ari_vs_grid": adjusted_rand_score(sig["grid_region"], labels),
                "nmi_vs_grid": normalized_mutual_info_score(sig["grid_region"], labels),
                "min_cluster_size": int(sizes.min()),
                "labels": labels,
            })
    return pd.DataFrame(rows)


def select_clustering(scores: pd.DataFrame, min_size: int = 2) -> pd.Series:
    """
    Pick the winning (algorithm, K).

    Rule: among solutions whose smallest zone has >= 2 states (a one-state
    "zone" is just that state again), rank by silhouette, Calinski-Harabasz
    and Davies-Bouldin (lower=better) and take the best mean rank.
    Averaging three criteria guards against the known bias of any single
    index (silhouette favours small K, CH favours large K).
    """
    s = scores[scores["min_cluster_size"] >= min_size].copy()
    s["rank_sil"] = s["silhouette"].rank(ascending=False)
    s["rank_ch"] = s["calinski_harabasz"].rank(ascending=False)
    s["rank_db"] = s["davies_bouldin"].rank(ascending=True)
    s["mean_rank"] = s[["rank_sil", "rank_ch", "rank_db"]].mean(axis=1)
    return s.sort_values(["mean_rank", "silhouette"], ascending=[True, False]).iloc[0]


def name_zones(sig: pd.DataFrame, labels: np.ndarray, weights: pd.Series) -> dict[int, str]:
    """Human-readable zone name = 'Z<k>_' + its two largest-demand states."""
    names = {}
    for z in np.unique(labels):
        members = sig.index[labels == z]
        top = weights.reindex(members).sort_values(ascending=False).index[:2]
        names[int(z)] = f"Z{int(z)}_" + "+".join(t.split()[0] for t in top)
    return names


# Daily state features that are aggregated to zone level (demand-weighted).
ZONE_FEATURES = [
    "imd_tmax", "imd_tmin", "imd_tmax_departure", "imd_tmin_departure",
    "hw_area_frac", "heat_severity", "ehf", "heat_accum_ewm", "cdd24", "cdd24_3d",
    "hot_day_run_length", "heat_day_run_length", "tmax_change_1d",
    "era5_heat_index_max", "era5_rh_mean", "era5_night_tmin", "era5_t_15h",
    "era5_cdh24", "era5_precip_sum", "era5_cloud_mean", "era5_solar_kwh_m2",
    "era5_wind_mean", "era5_dewpoint_mean",
]


def zone_daily_features(state_day: pd.DataFrame, zone_of: pd.Series,
                        weights: pd.Series, zone_names: dict[int, str]) -> pd.DataFrame:
    """
    Demand-weighted zone averages of daily state weather, wide format:
    one row per date, columns like 'Z2_Uttar+Rajasthan__imd_tmax'.

    Also adds per-zone heat exposure: share of the ZONE's demand in a
    state_heat_day - e.g. 0.8 means 80 % of the zone's load is in a heatwave.
    """
    cols = [c for c in ZONE_FEATURES if c in state_day.columns]
    df = state_day[["date", "state", "state_heat_day"] + cols].copy()
    df["zone"] = df["state"].map(zone_of)
    df["w"] = df["state"].map(weights).fillna(0)
    df["w"] = df["w"] / df.groupby(["date", "zone"])["w"].transform("sum")
    wcols = cols + ["state_heat_day"]
    weighted = df[wcols].mul(df["w"], axis=0)
    weighted[["date", "zone"]] = df[["date", "zone"]]
    z = weighted.groupby(["date", "zone"])[wcols].sum(min_count=1)
    z = z.rename(columns={"state_heat_day": "heat_exposure"})
    wide = z.unstack("zone")
    wide.columns = [f"{zone_names[int(zn)]}__{feat}" for feat, zn in wide.columns]
    return wide.sort_index(axis=1)
