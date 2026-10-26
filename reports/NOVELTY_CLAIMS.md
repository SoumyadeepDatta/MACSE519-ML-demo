# Novelty Claims and Evidence

**Title (T1):** Heatwave-Conditioned Peak Electricity Demand Forecasting Using Asymmetric Loss Optimization
**Source of the gaps:** *Research Objective Mapping* (G1-G11). **Evidence:** `notebooks/08_novelty_evidence_and_gap_mapping.ipynb`;
all numbers below are reproduced from `reports/metrics/evidence_*.csv`.

All tests use the walk-forward backtest (test years 2022-2025, 1,451 days, of which 381 are heat-or-critical-peak days).
"OOS folds" = 2022, 2023 and 2025, the years **not** used to tune HALO (it was tuned on 2024). Significance comes from a paired
moving-block bootstrap (7-day blocks, 1,000-2,000 resamples), Diebold-Mariano (HAC, HLN-corrected) and exact McNemar tests.

---

## 1. What already exists (so we do not over-claim)

A literature check found prior work on each *ingredient*, but not on their combination:

* **Asymmetric losses for load forecasting.** These exist, for example LSTMs with losses that penalise under-prediction, and gradient boosting
  with a branched/piecewise asymmetric loss ([Customizable Asymmetric Loss Functions](https://www.researchgate.net/publication/344899007_Customizable_Asymmetric_Loss_Functions_for_Machine_Learning-based_Predictive_Maintenance)).
  Custom asymmetric objectives in LightGBM are also a known practice ([Medium](https://medium.com/data-science/custom-loss-functions-for-gradient-boosting-f79c1b40466d)).
* **LINEX loss.** Standard since Varian (1975) and used in other domains, e.g. real-estate prediction ([arXiv 2410.09673](https://arxiv.org/pdf/2410.09673)), and as an
  evaluation metric ([sktime MeanLinexError](https://sktime-backup.readthedocs.io/en/stable/api_reference/auto_generated/sktime.performance_metrics.forecasting.MeanLinexError.html)).
* **Heat and peak demand.** Known to be asymmetric in the tail, and mean-based models under-predict it ([PMC7331730](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7331730/)).
* **India.** ML peak-demand work with ERA5 and POSOCO data exists (e.g. [ACPET Policy Bulletin](https://www.acpet.ashoka.edu.in/pdf/ACPET_EFL_PolicyBulletin.pdf), Delhi NCR, XGBoost),
  but we found none that conditions the loss on IMD heatwave criteria.

**Therefore do NOT claim:** "the first asymmetric loss for load forecasting", "the first use of LINEX", or "the first custom objective in LightGBM".

## 2. The novelty statement (safe wording)

> To our knowledge, this is the first study to train a gradient-boosted peak-demand forecaster with a **LINEX objective whose
> asymmetry varies per day with (i) an officially defined, demand-weighted heatwave severity (IMD criteria combined with the Excess Heat Factor)
> and (ii) grid scarcity**, and to show on the Indian national grid (2019-2025) that this **conditioned** asymmetry gives significantly
> more peak safety than **uniform** asymmetric losses (LINEX or pinball) **at equal ordinary-day accuracy**.

The last clause matters. Any asymmetric loss reduces shortfalls by biasing forecasts upward. The novel result is that *conditioning*
makes that trade-off more efficient (see the frontier figure `reports/figures/08_frontier.png`).

## 3. Gap-by-gap verdict

| Gap | Verdict | Evidence (OOS folds unless stated) | Claim it as |
|---|---|---|---|
| **G1** symmetric losses | ✅ Supported | HALO vs L2: expected shortfall on heat/critical days **−536 MW/day** (CI −665..−402, p<0.001); cost −1.57 crore/day (p<0.001); DM test on daily cost p<0.001. **Cost:** ordinary-day MAE +243 MW (p<0.001) | "An asymmetric objective significantly reduces peak shortfall relative to MSE, at a measurable accuracy cost." |
| **G2** rare heatwave regimes | ✅ Supported | Under-forecast rate on heat/critical days **65.7 % → 52.5 %** (p<0.001); McNemar 42 vs 2 discordant days (p<0.001) | "The symmetric model under-forecasts two thirds of heat/critical days; HALO cuts this significantly." |
| **G3** heatwave as a conditioning regime | 🟡 Partially | **Through the loss:** adding the heat term to a scarcity-only HALO lowers heatwave-day expected shortfall **1,205 → 905 MW (−300 MW, p<0.001)**. **As features:** removing explicit regime features changes nothing (p=0.65) | "Heatwave severity is most useful as a *conditioning signal in the loss*, not as an extra input feature." |
| **G4** custom asymmetric objective in boosting | ✅ Supported | At equal ordinary-day accuracy, HALO vs uniform **pinball**: expected shortfall **−296 MW** (CI −396..−116), cost −0.75 crore/day (p=0.011), under-forecast rate −8.5 pts (p=0.005) | "HALO dominates quantile (pinball) boosting on the accuracy-safety frontier." |
| **G5** severity-adaptive asymmetry | ✅ Supported | At equal accuracy, HALO vs **constant-asymmetry LINEX**: expected shortfall **−227 MW** (CI −372..−102), cost −0.51 crore/day (p=0.034), under-forecast rate −9.6 pts (p<0.001). All 4 HALO settings tested lie below the uniform curves | "Making the asymmetry adaptive is what improves the trade-off; a constant asymmetry cannot reach it." |
| **G6** metrics hide peak risk | ✅ Supported | Ranking 6 models by MAPE is **exactly reversed** by every peak-risk metric (Kendall τ = −1.00) | "MAPE/RMSE alone would select the model that is worst on heat and record days." |
| **G7** heat accumulation features | 🟡 Partially | Removing persistence/thermal-lag features worsens **HALO** shortfall by +55 MW (p=0.006) and cost by +0.21 crore (p=0.009), with **no effect on L2**. Humidity features: no effect | "Heat-accumulation features matter in combination with a heat-conditioned loss." Do not claim humidity effects. |
| **G8** Indian grid | ✅ Supported (scope) | All-India 2019-2025, three stitched Grid-India sources, IMD official gridded data and heatwave rules, record 2024 heatwave in the evaluation | "Applied to the Indian national grid." |
| **G9** lightweight and interpretable | 🟡 Partially | Training 1.6 s, inference 6 ms per year, 650 kB model, exact TreeSHAP; heat features push HALO's forecast up on heatwave days (+0.26 % of base) but not L2's (−0.10 %) | "Lightweight and SHAP-interpretable." No user study, so no claim of *better* interpretability. |
| **G10** point vs probabilistic | 🟡 Partially | Compared with pinball (quantile) curves only | Do not claim a comparison with conformal methods. |
| **G11** distribution shift | 🟡 Partially | On 162 days with demand at or above 98 % of the training maximum, a raw-MW LightGBM under-forecasts **100 %** (bias −18.5 GW, MAPE 8.6 %); our target transform gives 1.4 % MAPE; HALO bias −51 MW | "Robust to growth-driven level shift." Climate-scenario shift was not tested. |
| *(design)* zone clustering | ❌ Not supported | For the national target, national-average weather is as good as 8 zones (HALO heat/critical cost 6.11 vs 6.45) | Present clustering as **descriptive**: it maps India's thermal-demand geography. Do not claim it improves the forecast. |

## 4. Limitations to state in the report

1. **Ordinary-day accuracy cost.** HALO's ordinary-day MAE is 5-10 % higher than L2's (2025: +10.3 %; backtest: +5.2 %). The 5 % tolerance holds on the
   validation year only. Versus quantile 0.6 the accuracy is statistically indistinguishable in the backtest, but worse on 2025 alone.
2. **The heat term alone is not robust out of sample.** Heat-only HALO beats uniform LINEX on all folds but not on the OOS folds (p 0.10-0.32).
   The full effect needs heat **and** scarcity conditioning.
3. **The scarcity term uses the realised peak** (a label). That is legitimate inside a training loss, but it means scarcity is learned
   through proxies. An ex-ante scarcity variant was tested and was worse.
4. **Selection rule revised once** after the first test-year results (disclosed in the README). All evidence here uses the final rule,
   and the OOS-fold results do not involve 2024.
5. **2025 alone has only 52 heat/critical days.** Several 2025-only differences are not significant. The backtest is the primary evidence.
6. **The stylised cost model** (Rs 10/kWh short, Rs 2.5/kWh surplus, 4-hour window) is for comparing models, not for settlement.
7. **Perfect-forecast weather** (observed weather for day D), as is standard ex-post practice. Operational use would add IMD forecast error.

## 5. One-paragraph abstract claim (ready to paste)

> We propose HALO, a heat-adaptive, load-aware LINEX objective for LightGBM whose asymmetry grows with a demand-weighted heatwave severity
> index (IMD criteria combined with the Excess Heat Factor) and with grid scarcity. On All-India daily peak demand (2019-2025, walk-forward backtest),
> HALO reduces the expected shortfall on heatwave and critical-peak days by 33 % and the under-forecast rate from 66 % to 52 % relative to a
> symmetric LightGBM (p < 0.001). Crucially, at equal ordinary-day accuracy it lowers shortfall by 227-296 MW/day more than uniform LINEX and
> pinball losses (p < 0.001 on years not used for tuning), showing that conditioning the asymmetry, rather than asymmetry itself, drives the gain.
> We also show that MAPE ranks the models in exactly the reverse order of peak risk (Kendall τ = −1), which motivates regime-specific evaluation.
