# India Power Grid — NLDC Daily PSP Reports

**Structured, analysis-ready data scraped from the National Load Despatch Centre's (NLDC) daily Power System Position (PSP) reports — updated automatically every day.**

Covers the Indian national grid from 2018 onward across four complementary datasets: daily grid features, hourly regional load, 15-minute SCADA frequency data, and daily state-level power supply position.

---

## Files

| File | Rows | Cols | Period | Granularity |
|------|------|------|--------|-------------|
| `study1_daily.csv` | 2,680 | 144 | Dec 2018 → present | 1 row per day |
| `study1_hourly.csv` | 46,728 | 151 | Jan 2019 → Apr 2024 | 1 row per hour |
| `study2_scada.csv` | 57,084 | 164 | Nov 2024 → present | 1 row per 15-min block |
| `study3_states.csv` | 99,248 | 10 | Dec 2018 → present | 1 row per state/UT/entity per day |

Row counts grow daily via automated updates — treat as a snapshot, not a live figure.

---

## Source

All data is scraped from the **NLDC (National Load Despatch Centre) / Grid-India** daily PSP reports, published at [grid-india.in](https://grid-india.in/en/reports/daily-psp-report).

Reports are published in PDF format (up to ~2023) and XLS format (2023 onward). Each report covers the previous calendar day.

**Scraper and full methodology:** [github.com/HalcyonVector/Grid-Sentinel](https://github.com/HalcyonVector/Grid-Sentinel)

---

## Dataset Descriptions

### study1_daily.csv — Daily Grid Features

One row per calendar day. Built from the MOP_E sheet (XLS era) and equivalent PDF sections. Covers generation mix, regional demand, outages, inter-regional flows, cross-border exchange, and frequency statistics.

**Key column groups:**

| Group | Example columns | Notes |
|-------|----------------|-------|
| Peak demand met | `peak_demand_met_total_mw`, `peak_demand_met_nr_mw`, `..._wr_`, `..._sr_`, `..._er_`, `..._ner_` | Per region + national |
| Energy met & shortage | `energy_met_total_mu`, `energy_shortage_total_mu` | Daily MU |
| Generation mix | `gen_coal_mu`, `gen_hydro_mu`, `gen_nuclear_mu`, `gen_res_mu`, `gen_gas_mu`, `gen_lignite_mu`, `gen_total_mu` | |
| Outages | `outage_central_total_mw`, `outage_state_total_mw`, `outage_total_total_mw` | Per region × central/state |
| Frequency | `freq_fvi`, `freq_pct_below_497`, `freq_pct_497_498`, `freq_pct_498_499`, `freq_pct_499_5005`, `freq_pct_above_5005` | Daily frequency band distribution |
| IR-Line corridors | `ir_wr_nr_export_mu`, `ir_wr_nr_import_mu`, `ir_wr_nr_net_mu` (× 7 region pairs) | 21 cols; available ~2023 onward |
| Cross-border exchange | `xb_export_bhutan_mu`, `xb_import_nepal_mu`, `xb_net_bangladesh_mu` (× 4 countries) | 12 cols; available ~2023 onward |
| RES share | `share_res_pct`, `share_nonfossil_pct` | |
| Diversity factor | `diversity_regional`, `diversity_state` | |

Columns that start appearing only after a certain year (IR-line, cross-border, solar breakdown) are `NaN` before their schema onset date — these are not parse failures, simply sections that did not exist in earlier reports.

---

### study1_hourly.csv — Hourly Regional Load

One row per hour. Built by left-joining `study1_daily` features onto the [India Hourly Load dataset](https://www.kaggle.com/datasets/twinkle0705/state-wise-power-consumption-in-india) (Kaggle). The hourly load source ends April 2024, so this file has a fixed end date.

**Extra columns vs study1_daily:**

| Column | Description |
|--------|-------------|
| `datetime` | Hourly timestamp (UTC+5:30 implied) |
| `National Hourly Demand` | MW |
| `Northern Region Hourly Demand` | MW |
| `Western Region Hourly Demand` | MW |
| `Eastern Region Hourly Demand` | MW |
| `Southern Region Hourly Demand` | MW |
| `North-Eastern Region Hourly Demand` | MW |

All 144 daily PSP feature columns are broadcast (repeated) across the 24 hourly rows of each day. Rows where no PSP file exists for that day have `NaN` on all PSP columns (~3.6% of rows).

---

### study2_scada.csv — 15-Minute SCADA Frequency Data

One row per 15-minute block (96 per day). Built from the `TimeSeries` sheet in NLDC XLS files, available from November 2024 onward. Daily MOP_E, IR-Line, and CrossBorder features are broadcast onto each 15-minute row.

**Key column groups:**

| Group | Example columns |
|-------|----------------|
| Timestamp | `date`, `time`, `hhmm` |
| Real-time generation | `nuclear_mw`, `wind_mw`, `solar_mw`, `hydro_mw`, `gas_mw`, `thermal_mw`, `total_gen_mw` |
| Demand | `demand_met_mw`, `net_demand_met_mw` |
| Net transmission | `net_trans_exchange_mw` |
| Frequency | `freq_hz`, `freq_fvi`, `freq_pct_below_497`, `freq_pct_497_498`, `freq_pct_499_5005`, `freq_pct_above_5005` |
| Evening peak demand | `evening_peak_demand_nr_mw`, `..._wr_`, `..._sr_`, `..._er_`, `..._ner_` |
| IR-Line + CrossBorder | Same 21 + 12 cols as study1_daily, broadcast per slot |

---

### study3_states.csv — Daily State-Level Power Supply Position

One row per state/UT/entity per day (~37 per day), long format. Built from NLDC's §C "Power Supply Position in States" table, present in both the PDF (2019–2022) and XLS (2023+) report eras.

**Columns:**

| Column | Description |
|--------|-------------|
| `date` | Calendar date |
| `region` | NR / WR / SR / ER / NER |
| `state` | State, UT, or grid entity name (e.g. also includes distinct bulk-consumer/railway drawal entities NLDC reports alongside states) |
| `max_demand_met_mw` | Maximum demand met that day |
| `shortage_max_demand_mw` | Shortage during maximum demand |
| `energy_met_mu` | Energy met |
| `drawal_schedule_mu` | Scheduled drawal |
| `od_ud_mu` | Overdrawal(+)/Underdrawal(-) |
| `max_od_mw` | Maximum overdrawal |
| `energy_shortage_mu` | Energy shortage |

State naming changed over the dataset's history (e.g. J&K's Aug 2019 split into J&K and Ladakh UTs, Puducherry's older "Pondy" abbreviation) — these are normalized to one consistent name per entity across the full time range so the series doesn't fragment; see the scraper's `STATE_NAME_ALIASES` for the exact mapping.

---

## Known Gaps & Limitations

**study1_daily / study1_hourly — 69 missing dates (all irreducible, each individually verified against the raw archive):**

| Cause | Count |
|-------|-------|
| NLDC republished an existing date's report under a new filename instead of publishing new content (the date in between got no coverage) | 49 |
| File confirmed absent from NLDC's server under any filename — a genuine hole in what NLDC published | 16 |
| Edge cases (server downtime around 2025-05-22/23, 2 confirmed-absent dates found by direct probe) | 4 |

Missing dates are concentrated in 2020 (COVID-era publishing irregularities). These are source-level absences, not parse failures — every one has a named, individually-verified reason (not an assumed category); see the project repo's `Pipeline/known_gaps.json` for the full per-date list. Recommend forward-fill or time-series-aware imputation.

**study2_scada — slot irregularities and missing dates:**

A small number of days have 95, 97, or 98 slots instead of 96 (DST / file-truncation edge cases) — usable, not dropped. Three days are severely corrupted and should be dropped before training: 2025-10-02 (63 slots) plus 2024-11-20 and 2025-04-01 (1 slot each). Separately, **17 calendar dates have no rows at all**: 2 are the same server-downtime gap as above, 13 have only a PDF source published for that date (a PDF can't contain the 15-min `TimeSeries` sheet SCADA data comes from), and 2 (2025-07-15, 2025-08-20) have an XLS file whose `TimeSeries` sheet is present but contains zero data rows. All genuine source-side gaps, not parser bugs.

**IR-Line and CrossBorder columns** are `NaN` before ~2023 — these sections were not published in earlier reports.

**study3_states — one missing date:** 2019-05-09 (not 2019-05-10 as earlier documented — corrected after checking the source PDF's own subject line) has no rows because that day's source PDF was published entirely in Hindi/Devanagari (the only report in the archive rendered this way), and the state-section parser's header match is English-only. `study1_daily`/`study1_hourly` are unaffected — their parsers don't depend on that same text match.

---

## Update Schedule

`study1_daily`, `study2_scada`, and `study3_states` are updated automatically every day via GitHub Actions. The workflow downloads the latest PSP file from grid-india.in, parses it, appends new rows, and pushes here.

`study1_hourly` has a fixed end date of April 2024 (the hourly load source dataset is not live).

---

## Suggested Use Cases

- **Daily load forecasting** — predict next-day peak demand or energy met using generation mix, outage, IR-line, and cross-border features
- **Frequency violation classification** — classify 15-minute blocks by whether a frequency violation occurred, using real-time SCADA features from study2_scada
- **Grid stress analysis** — study the relationship between outages, inter-regional flows, and demand shortages
- **Renewable integration analysis** — track RES share growth and its correlation with frequency deviation over time
- **Cross-border power flow modelling** — analyse India's energy exchange with Bhutan, Nepal, Bangladesh, Myanmar

---

## License

**CC BY-SA 4.0** — You are free to share and adapt this dataset for any purpose, provided you give appropriate credit and distribute derivatives under the same license.

Source data is published by NLDC / Grid-India under India's National Data Sharing and Accessibility Policy (NDSAP).

---

## Citation

```
Halcyon Vector. (2026). India Power Grid — NLDC Daily PSP Reports [Dataset].
Kaggle. https://www.kaggle.com/datasets/halcyonvector/india-power-grid-nldc-daily-psp-reports
Scraper & methodology: https://github.com/HalcyonVector/Grid-Sentinel
```

---

## Acknowledgements

Data sourced from the **National Load Despatch Centre (NLDC)**, operated by **Posoco / Grid-India**, Government of India. Reports available at [grid-india.in](https://grid-india.in/en/reports/daily-psp-report).

Hourly load data (used in study1_hourly) from the [India Hourly Load dataset](https://www.kaggle.com/datasets/twinkle0705/state-wise-power-consumption-in-india) on Kaggle.
