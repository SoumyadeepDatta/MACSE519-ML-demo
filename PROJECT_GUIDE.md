# PROJECT GUIDE

## Heatwave-Conditioned Peak Electricity Demand Forecasting Using Asymmetric Loss Optimization

**MACSE519 ML Lab Project** · Soumyadeep Datta (26MAI0002) · Debrup Sarkar (26MAI0030) · Kaushiki Mandal (26MAI0036)

This guide explains **everything** in this project folder in plain language: what problem we solve, where every number comes from,
what every piece of code does, why each decision was made, what every output file and figure means, and what is genuinely new.
If you know only the basic idea of machine learning ("a model learns patterns from past examples to predict new ones"), you
can follow all of it. Every technical word is explained the first time it appears, and again in the Glossary (Chapter 3).

---

## Table of contents

**Part A: The big picture**

1. [The problem in one story](#1-the-problem-in-one-story)
2. [What we built, in one page](#2-what-we-built-in-one-page)
3. [Glossary: every word in plain language](#3-glossary-every-word-in-plain-language)

**Part B: Foundations**

4. [Machine learning basics, told through our project](#4-machine-learning-basics-told-through-our-project)
5. [Decision trees, explained with our data](#5-decision-trees-explained-with-our-data)
6. [LightGBM: boosting explained in minute detail](#6-lightgbm-boosting-explained-in-minute-detail)
7. [Loss functions: how a model is told what a &#34;mistake&#34; costs](#7-loss-functions-how-a-model-is-told-what-a-mistake-costs)
8. [HALO, our custom asymmetric loss, explained in minute detail](#8-halo-our-custom-asymmetric-loss-explained-in-minute-detail)

**Part C: Data**

9. [The datasets and why each one was chosen](#9-the-datasets-and-why-each-one-was-chosen)
10. [Folder structure, file formats, and why Parquet](#10-folder-structure-file-formats-and-why-parquet)

**Part D: The end-to-end workflow, stage by stage**

11. [The whole pipeline at a glance](#11-the-whole-pipeline-at-a-glance)
12. [Stage 1: Downloading](#12-stage-1-downloading)
13. [Stage 2: Cleaning the electricity data and finding the daily peak](#13-stage-2-cleaning-the-electricity-data-and-finding-the-daily-peak)
14. [Stage 3: Weather and deciding what a heatwave is](#14-stage-3-weather-and-deciding-what-a-heatwave-is)
15. [Stage 4: Grouping states into zones (clustering)](#15-stage-4-grouping-states-into-zones-clustering)
16. [Stage 5: Building the model inputs (features)](#16-stage-5-building-the-model-inputs-features)
17. [Stage 6: Training all the models](#17-stage-6-training-all-the-models)
18. [Stage 7: Measuring how good the models are](#18-stage-7-measuring-how-good-the-models-are)
19. [Stage 8: Proving the novelty (evidence tests)](#19-stage-8-proving-the-novelty-evidence-tests)

**Part E: Novelty**

20. [What is new, and how we achieved it](#20-what-is-new-and-how-we-achieved-it)

**Part F: Every output explained**

21. [Every output file](#21-every-output-file)
22. [Every figure](#22-every-figure)
23. [Every notebook and what it prints](#23-every-notebook-and-what-it-prints)

**Part G: The code, file by file**

24. [Code walkthrough with project examples](#24-code-walkthrough-with-project-examples)

**Part H: Decisions, limits and running it**

25. [Decision log: what, why, how](#25-decision-log-what-why-how)
26. [Honest limitations and the revision history](#26-honest-limitations-and-the-revision-history)
27. [How to run everything](#27-how-to-run-everything)
28. [Frequently asked questions](#28-frequently-asked-questions)

---

# Part A: The big picture

## 1. The problem in one story

On **30 May 2024**, India used more electricity at one moment than ever before: about **250 gigawatts** (GW). A gigawatt is a
thousand megawatts. One megawatt (MW) is roughly what 1,000 homes running air conditioners draw at once. A brutal heatwave was
sitting over Rajasthan, Delhi, Haryana, Punjab, Uttar Pradesh and Madhya Pradesh, with temperatures of 45–49 °C, and hundreds of
millions of air conditioners and coolers were running at the same time.

Every morning, India's grid operator (the **National Load Despatch Centre, NLDC**, part of **Grid-India**) has to decide how many
power stations to keep ready for **tomorrow**. To do that, they need a forecast of **tomorrow's highest demand**: the *daily peak*.

Two kinds of mistakes are possible:

| Mistake                                                      | What happens                                                                                                                                                                                                                                | How bad                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| **Over-forecast** (we predicted more than was needed)  | Some power stations were kept ready and turned out not to be needed. They are told to reduce output, but they still get paid "capacity charges".                                                                                            | Costs money (roughly ₹2.5 per unit of electricity) |
| **Under-forecast** (we predicted less than was needed) | Not enough power is available. The grid must buy emergency electricity at the market's maximum allowed price (₹10 per unit), and if none is available, it **cuts power to neighbourhoods** (load-shedding) during a 45 °C heatwave. | Expensive, and dangerous for people's health        |

**The core problem:** almost every forecasting model is trained to treat both mistakes as **equally bad**. That is wrong for
electricity, and it is most wrong exactly when it matters most: on heatwave days, when demand is extreme and a shortfall means
blackouts in dangerous heat.

**Our goal:** build a model that forecasts tomorrow's All-India peak demand accurately on normal days, but deliberately leans towards
**not being short** on heatwave days and record-demand days.

## 2. What we built, in one page

1. **Real Indian data**, all public:
   * Hourly electricity demand for all of India, 2019–2025: about 61,000 hours stitched together from three Grid-India sources.
   * Daily electricity demand for each state.
   * Official **India Meteorological Department (IMD)** temperature maps from 1991 to 2025.
   * Hourly weather (temperature, humidity, rain, sun and more) for 33 states from the ERA5 weather archive, about 2 million rows.
2. **Daily peak:** from the hourly data we take the highest hour of each day, giving 2,557 days of "peak demand".
3. **Heatwaves defined properly:** using IMD's official heatwave rule, plus an internationally used heat measure (the Excess Heat Factor)
   that also catches humid heat.
4. **State grouping:** the 33 states are grouped into 8 "zones" of similar climate and electricity behaviour, using clustering.
5. **Features:** 309 input columns per day (yesterday's demand, day of week, festivals, heat in each zone, and so on).
6. **The model:** **LightGBM**, a fast, accurate method that builds many small decision trees, each one correcting the previous ones' mistakes.
7. **The novelty: HALO**, a custom "mistake-cost" rule (a *loss function*) that:
   * treats small errors normally,
   * punishes being **short** much more than being **over**,
   * becomes **much stricter** on heatwave days and when demand is near its all-time record.
8. **Fair, honest testing:**
   * trained on 2019–2023, tuned on 2024, tested once on 2025,
   * plus a 4-year "walk-forward" re-test (2022–2025),
   * plus statistical significance tests showing the gains are real and not luck.

**Headline results**

* **Everyday accuracy:** the forecast is off by only about **1.2–1.4 %** on average (about 2.5–2.8 GW on a 200–245 GW peak).
  A simple "same as last week" guess is off by 3.6–3.8 %.
* **On heatwave and record days** (2022–2025 test years), HALO compared with a normal model:
  * cuts the average shortfall by **33 %** (1,873 MW → 1,254 MW per day),
  * cuts how often it is short from **66 % to 52 %** of such days,
  * halves the days with a dangerously big shortfall (4.7 % → 2.1 %).
* **The novelty:** at the **same** everyday accuracy, HALO is significantly safer than other "lean-towards-safety" methods that use a
  fixed bias. Conditioning the safety margin on heat and grid stress, not the bias itself, is what makes the difference.
* **The price** (we are honest about it): on ordinary days HALO is about **5–10 % less accurate** than the normal model.

## 3. Glossary: every word in plain language

Words are grouped by topic. Each is explained with our project as the example.

### Electricity words

| Word                                 | Plain meaning                                                                                                   | Example from our project                                   |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **MW / GW**                    | Megawatt / gigawatt: units of electrical *power* (how much is being used at one moment). 1 GW = 1,000 MW.     | India's peak on 30-May-2024 ≈ 246,779 MW (hourly average) |
| **MWh / MU**                   | Megawatt-hour, an amount of *energy* (power × time). 1 MU ("million units") = 1,000 MWh.                     | Running 1,000 MW for 4 hours uses 4,000 MWh                |
| **Demand / load**              | How much electricity everyone together is using                                                                 | "National Hourly Demand" column                            |
| **Daily peak**                 | The highest demand of the day                                                                                   | Our forecasting **target**                           |
| **Peak hour**                  | The clock hour at which the peak happened                                                                       | 14:00 on 30-May-2024                                       |
| **Grid region**                | India's grid is run as 5 regions: Northern (NR), Western (WR), Southern (SR), Eastern (ER), North-Eastern (NER) | Delhi is in NR                                             |
| **NLDC / Grid-India / POSOCO** | The national grid operator (POSOCO was renamed Grid-India)                                                      | Publishes the data we use                                  |
| **PSP report**                 | "Power Supply Position": NLDC's daily report on demand, generation, shortages, per region and per state         | Source of state-wise demand                                |
| **SCADA**                      | The grid's live measurement system (readings every few seconds)                                                 | 15-minute demand data from Nov 2024                        |
| **PMU**                        | A very precise grid measuring device                                                                            | Source of the Mendeley dataset                             |
| **Reserves**                   | Spare generation kept ready (roughly 3–5 % of demand in India)                                                 | A shortfall bigger than ~3 % can exhaust them              |
| **Load-shedding**              | Deliberately cutting power to some areas when supply is short                                                   | What we want to avoid                                      |
| **Load factor**                | Average demand ÷ peak demand (0–1). Low = spiky day.                                                          | 0.914 on 30-May-2024                                       |

### Weather words

| Word                                 | Plain meaning                                                                                                                 | Example                                        |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| **Tmax / Tmin**                | The day's highest / lowest temperature                                                                                        | Delhi Tmax 46 °C                              |
| **Normal**                     | The average temperature for that calendar day over 1991–2020                                                                 | Delhi's normal Tmax for 17 May ≈ 39.5 °C     |
| **Departure**                  | Today's temperature minus the normal                                                                                          | 44 °C on a 39.5 °C normal day = +4.5 °C     |
| **IMD**                        | India Meteorological Department (Government of India)                                                                         | Defines official heatwaves                     |
| **Gridded data**               | Temperatures given on a map grid (here 1° × 1° squares, about 110 km wide) instead of at individual stations               | 31 × 31 squares cover India                   |
| **ERA5**                       | A global weather "reanalysis": a best reconstruction of past weather, hour by hour, made by the European weather centre ECMWF | Hourly humidity for Chennai                    |
| **Relative humidity (RH)**     | How full of water vapour the air is, 0–100 %                                                                                 | Chennai afternoons ≈ 60–70 %                 |
| **Dew point**                  | The temperature at which the air's moisture would condense; a high dew point means sticky air                                 |                                                |
| **Heat index**                 | "Feels-like" temperature that combines heat and humidity (US weather service formula)                                         | Chennai 36 °C at 65 % RH feels like ≈ 50 °C |
| **Cooling degree-hours (CDH)** | Add up, hour by hour, how far the temperature is above 24 °C. More = more cooling needed.                                    | 24 hours at 34 °C = 240 degree-hours          |
| **Heatwave (IMD)**             | Official rule; see Chapter 14                                                                                                 |                                                |
| **EHF (Excess Heat Factor)**   | An international heatwave measure that compares the last 3 days' heat with the local "hot" level and with the previous month  | See Chapter 14                                 |

### Machine-learning words

| Word                               | Plain meaning                                                                                                            | Example                                                         |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| **Feature** (input)          | A piece of information the model sees before predicting                                                                  | "Yesterday's peak", "Day of week", "Heat in North India"        |
| **Target** (label)           | The thing we want to predict                                                                                             | Tomorrow's peak (expressed as % above last week; see Ch. 16)    |
| **Model**                    | A recipe that turns features into a prediction                                                                           | LightGBM                                                        |
| **Training**                 | Showing the model past days with their true answers so it can learn                                                      | 2019–2023                                                      |
| **Validation set**           | Past days kept aside to tune settings and decide when to stop training                                                   | 2024                                                            |
| **Test set**                 | Days kept completely hidden until the very end, used once to report honest results                                       | 2025                                                            |
| **Overfitting**              | Memorising training days instead of learning general patterns, which fails on new days                                   | A tree that learns "13-May-2022 was high"                       |
| **Loss function**            | The rule that says how bad each mistake is; the model tries to make the total as small as possible                       | MSE, pinball, **HALO**                                    |
| **Symmetric loss**           | Over and under mistakes of the same size cost the same                                                                   | MSE                                                             |
| **Asymmetric loss**          | Under-forecasting costs more than over-forecasting (or the reverse)                                                      | HALO, pinball                                                   |
| **Gradient**                 | "Which direction, and how strongly, should this prediction move to reduce the loss?"                                     | Negative gradient = "move up"                                   |
| **Hessian**                  | "How sharply does the loss curve bend here?" It tells the model how confidently to take a step                           | Big Hessian = small, careful steps                              |
| **Hyper-parameter**          | A setting chosen by us, not learned from data                                                                            | Number of leaves per tree = 15                                  |
| **Early stopping**           | Stop adding trees when the validation score stops improving                                                              | Stopped L2 at 700 trees                                         |
| **Bias (of a forecast)**     | Average of (forecast − actual). Positive = tends to over-forecast                                                       | HALO bias ≈ +850 MW                                            |
| **MAE**                      | Mean Absolute Error: the average size of the mistake in MW                                                               | 2,462 MW for L2 in 2025                                         |
| **RMSE**                     | Root Mean Squared Error: like MAE but punishes big mistakes more                                                         |                                                                 |
| **MAPE**                     | Mean Absolute Percentage Error: average mistake as a % of actual                                                         | 1.16 %                                                          |
| **R²**                      | "Share of the day-to-day ups and downs the model explains" (1 = perfect, 0 = no better than always guessing the average) | 0.93                                                            |
| **Clustering**               | Automatically grouping similar items                                                                                     | 33 states → 8 zones                                            |
| **Bootstrap**                | Re-sampling the test days many times to see how much a result could change by luck                                       | 2,000 resamples                                                 |
| **p-value**                  | The chance of seeing a difference this big if there were really no difference. Below 0.05 = "significant"                | p < 0.001                                                       |
| **Confidence interval (CI)** | The range the true value probably lies in (95 %)                                                                         | −665 to −402 MW                                               |
| **Ablation**                 | Remove one ingredient, keep everything else the same, and see what changes                                               | "No heat-accumulation features"                                 |
| **SHAP**                     | A method that splits each prediction into "how much each feature pushed it up or down"                                   | Heat features pushed HALO up by +0.26 % on heatwave days        |
| **Backtest / walk-forward**  | Repeatedly train on past years and test on the next year, as if living through time                                      | Train ≤2021 → test 2022, train ≤2022 → test 2023, and so on |

---

# Part B: Foundations

## 4. Machine learning basics, told through our project

### 4.1 What a "forecast" is here

Each morning we want **one number**: tomorrow's All-India daily peak in MW. That is a **regression** problem (predicting a
number, not a category).

### 4.2 Learning from examples

We have 2,557 past days (1-Jan-2019 to 31-Dec-2025). For each day we know:

* **what was known the morning before** (yesterday's demand, the weather for the day, whether it is a festival, and so on). These are the **features**.
* **what actually happened** (that day's peak). This is the **target**.

The model looks at thousands of such pairs and learns patterns like "when North India is 3 °C hotter than last week and it is a
weekday, the peak is about 3 % above last week's average".

### 4.3 Why time order matters (no "peeking at the future")

In everyday ML you often shuffle data randomly into training and testing. **With time series that is cheating.** If 30-May-2024 were in
training and 29-May-2024 in testing, the model would effectively have seen the answer to a neighbouring day of the same heatwave.
So we always split **by time**:

```
2019 ──────────── 2023 | 2024     | 2025
      TRAINING         | VALIDATION | TEST (looked at once, at the very end)
```

### 4.4 Training, validation and test, and what each one is for

* **Training (2019–2023, 1,720 usable days):** the model learns from these.
* **Validation (2024, 366 days):** used to (a) decide how many trees to build ("early stopping") and (b) choose HALO's three
  strictness settings. 2024 holds the record heatwave, so it is a demanding year to tune on.
* **Test (2025, 365 days, 355 with measured data):** the final, honest score. After choosing everything on 2024, every model is
  re-trained on 2019–2024 and scored on 2025 once.

### 4.5 Over-fitting and how we avoid it

A model can "memorise" the training days. We prevent that with small trees, minimum day counts per leaf, random sampling of rows and
columns, shrinkage, and early stopping (Chapter 6.6 explains each one).

## 5. Decision trees, explained with our data

A **decision tree** is a flowchart of yes/no questions that ends in a number.

```
                  Is North India (zone Z5) at least 2.5 °C hotter than its last-week average?
                 /                                                                    \
               yes                                                                     no
               /                                                                        \
   Is it a Sunday?                                                        Was yesterday's peak above last week's average?
     /        \                                                              /                    \
   yes         no                                                          yes                    no
    |           |                                                           |                      |
 predict      predict                                                    predict                predict
 -1.5 %       +3.2 %                                                     +1.1 %                 -0.8 %
```

* Each **question** is a "split" on one feature.
* Each **end box** is a "leaf". It holds a number: here, the % by which tomorrow's peak will differ from last week's average.
* The tree is **learned**: the algorithm tries many possible questions and keeps those that best separate high-demand days from low-demand days.

A single tree is crude. Its predictions come in a few fixed steps, and it either over-fits (if it is deep) or misses detail (if it
is shallow). **Boosting** fixes this by adding many small trees together.

## 6. LightGBM: boosting explained in minute detail

### 6.1 The idea of boosting, with a story

Imagine a team of 700 junior analysts, each allowed to draw **one small flowchart**.

1. Analyst 1 makes a rough guess for every day. Mistakes remain.
2. Analyst 2 does **not** predict demand. They look only at **Analyst 1's mistakes** and draw a flowchart that predicts those
   mistakes ("Analyst 1 is too low on hot weekdays").
3. Analyst 3 looks at the mistakes that remain after Analysts 1 and 2, and so on.

The final forecast = the starting guess + a **small** fraction of each analyst's correction. Each analyst fixes what the team still
gets wrong. That is **gradient boosting**, and LightGBM is a fast, carefully engineered implementation of it.

### 6.2 What "gradient" means in boosting (no maths version)

For each training day, the model asks: *"If I nudge my prediction for this day, does the total mistake-cost go up or down, and how
fast?"* That direction-and-strength number is the **gradient**.

* With ordinary squared error, the gradient for a day is simply **(prediction − actual)**. If we predicted +1 % and the actual was
  +4 %, the gradient is −3: "move up, fairly strongly".
* The next tree is built to predict these gradients (with the sign flipped), i.e. to push each day's prediction in the right direction.

The **Hessian** is the second number LightGBM uses: how sharply the mistake-cost curve bends at the current prediction. Think of it
as **confidence in the step size**:

* A steep, sharply curving cost (big Hessian) means the minimum is close, so take a **small, precise step**.
* A flat cost (small Hessian) means it is safe to take a bigger step.

For squared error the Hessian is always 1, so every day gets the same "step confidence". HALO gives different days different
Hessians, and that turns out to matter (Chapter 8).

### 6.3 How one tree is built (what LightGBM actually does)

1. **Compute** the gradient *g* and Hessian *h* of every training day under the chosen loss.
2. **Find the best question (split).** For a candidate split, add up the *g*s and *h*s on the left and the right. A split is good if
   it separates days that need pushing up from days that need pushing down. LightGBM scores every candidate with a "gain" built from
   (sum of g)² ÷ (sum of h + λ) on each side.
3. **Grow leaf-wise.** LightGBM always splits the leaf that gives the biggest gain next ("leaf-wise" growth). This is
   faster and more accurate than growing level by level, but can over-fit, so we cap leaves (15) and depth (5).
4. **Set each leaf's value** with the Newton formula:

   ```
   leaf value = − (sum of g in the leaf) / (sum of h in the leaf + λ)
   ```

   then multiply by the **learning rate** (0.02) before adding it to the predictions.
5. **Speed trick (histograms).** LightGBM first sorts each feature into at most 255 "buckets" and only tests splits between buckets.
   With 309 features and ~2,000 days, a tree takes milliseconds.

### 6.4 A worked leaf example with our numbers

Suppose a leaf contains three training days whose current predictions are 0 % and whose true targets are **+3 %, +1 % and −1 %**
(all relative to last week). With squared error, *g* = prediction − actual = −3, −1, +1, and *h* = 1, 1, 1. With λ = 2:

```
leaf value = −(−3 − 1 + 1) / (1 + 1 + 1 + 2) = 3 / 5 = 0.6
```

The plain average of the three targets is 1.0. The λ term ("ridge") shrinks it to 0.6 so a leaf with few days cannot shout too
loudly. Multiplied by the learning rate 0.02, this tree moves those days up by 0.012 %. Hundreds of trees add up to the final answer.

The same leaf under HALO is shown in Chapter 8.8. On a heatwave day the leaf value becomes **1.47** instead of 0.6, because the loss
says "being short here is very costly".

### 6.5 Why LightGBM for this project

| Reason                                        | Explanation                                                                                                         |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Works brilliantly on tables of mixed features | Our data is a table: numbers, flags, day-of-week                                                                    |
| Small data friendly                           | ~1,700 training days is small; trees with strong limits handle this well, while deep neural networks would over-fit |
| **Accepts a custom loss**               | We only need to supply *g* and *h*. This is exactly what makes HALO possible                                    |
| Fast                                          | One HALO model trains in about 1.6 s, so we can run hundreds of experiments                                         |
| Explainable                                   | Exact per-prediction explanations (SHAP) are built in                                                               |
| Handles missing values                        | e.g. grid-ops columns that did not exist before 2023                                                                |

### 6.6 Every LightGBM setting we used, and why

These live in `src/heatwave_peak/models.py` as `BASE_PARAMS` and are **identical for every LightGBM model**, so the only
difference between models is the loss function.

| Setting                                      | Value       | Plain meaning                                                    | Why this value                                                                    |
| -------------------------------------------- | ----------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `learning_rate`                            | 0.02        | Each tree's correction is shrunk to 2 %                          | Small steps plus many trees is more accurate and less jumpy                       |
| `num_leaves`                               | 15          | At most 15 end-boxes per tree                                    | Keeps each tree simple                                                            |
| `max_depth`                                | 5           | At most 5 questions in a row                                     | Allows interactions of up to 5 factors (heat × weekday × month …) but not more |
| `min_data_in_leaf`                         | 25          | Each end-box must contain at least 25 days                       | About a month of days; a leaf cannot memorise one heatwave                        |
| `feature_fraction`                         | 0.6         | Each tree sees a random 60 % of the 309 features                 | Our zone features are highly correlated; this forces variety                      |
| `bagging_fraction`, `bagging_freq`       | 0.8, 1      | Each tree is trained on a random 80 % of days                    | Reduces over-fitting to particular days                                           |
| `lambda_l2`                                | 2.0         | The λ in the leaf formula                                       | Shrinks leaf values (see 6.4)                                                     |
| `deterministic`, `seed`, `num_threads` | True, 42, 4 | Same result every run                                            | Reproducibility                                                                   |
| `MAX_ROUNDS`                               | 4,000       | Upper limit on trees                                             | Never reached except by the quantile-0.6 model                                    |
| `EARLY_STOP`                               | 300         | Stop if the 2024 validation score has not improved for 300 trees | Prevents over-training                                                            |

Final number of trees (decided on 2024): L2 700, quantile-0.6 3,999, quantile-0.8 915, prototype 700 (copied from L2), LINEX 381,
**HALO 373**.

### 6.7 The "custom objective" plug-in

LightGBM lets you replace its built-in loss with a Python function that receives the current predictions and returns *g* and *h*
for every day. HALO is exactly such a function (`HALOObjective.objective` in `losses.py`). For early stopping we also supply a
matching scoring function (`HALOObjective.feval`), so the model is judged on 2024 with the same rule it is trained with.

## 7. Loss functions: how a model is told what a "mistake" costs

Let **e = actual − forecast** (in % of last week's average peak).

* **e > 0** means we were **short** (under-forecast).
* **e < 0** means we were **over** (over-forecast).

### 7.1 Squared error (MSE, "L2"): symmetric

`loss = e² / 2`. Short by 4 % costs 8. Over by 4 % also costs 8. The best forecast under MSE is the **average** outcome, so it is
short about half the time, including on heatwave days. In our 2022–2025 backtest, the L2 model was short on **66 %** of
heat/critical days.

### 7.2 Absolute error (MAE): symmetric

`loss = |e|`. Also symmetric. Its best forecast is the median.

### 7.3 Pinball / quantile loss: asymmetric but straight-lined

For a chosen level τ (tau), being short costs τ·|e| and being over costs (1−τ)·|e|. With τ = 0.8, short is 4× worse than over at
**every** size of error. Its best forecast is the 80th percentile of possible outcomes: it is **always** biased upwards by the same
amount, on a cool January Sunday just as on a 47 °C heatwave day.

*Why τ = 0.8 is the textbook choice here:* our stylised costs are ₹10 for short and ₹2.5 for over, and the cost-optimal quantile is
10 ÷ (10 + 2.5) = 0.8.

### 7.4 What the grid actually needs

1. **Small mistakes: treat them normally.** Reserves absorb them.
2. **Big shortfalls: much worse than big surpluses**, and increasingly so: a 1 % shortfall is absorbed, a 5 % shortfall means load-shedding.
3. **Be stricter on dangerous days** (heatwaves, near-record demand) and relaxed on harmless days.

None of MSE, MAE or pinball can do all three. HALO can.

## 8. HALO, our custom asymmetric loss, explained in minute detail

**HALO = Heat-Adaptive, Load-aware LINEX Objective.** Code: `src/heatwave_peak/losses.py`, class `HALOObjective`.

### 8.1 The building block: LINEX

LINEX ("LINear-EXponential") is a known asymmetric loss (Varian, 1975). We use it in this scaled form:

```
L(e) = ( exp(a·e) − a·e − 1 ) / a²
```

* `exp(x)` is the exponential function (e^x).
* `a` is the **strictness** (asymmetry) number: a > 0 punishes being short.
* The division by a² makes L(e) ≈ e²/2 for small errors, i.e. **the same as ordinary squared error near zero**.

**What the shape does:**

* For **positive e (short)**, `exp(a·e)` grows **exponentially**: being a bit more short costs a lot more.
* For **negative e (over)**, `exp(a·e)` shrinks towards 0, so the loss grows only **in a straight line** (≈ |e|/a).

### 8.2 The novelty part: the strictness `a` is different for every day

```
a_i = a0  +  a_heat × severity_i  +  a_scarcity × scarcity_i
```

| Symbol         | What it is                                                             | Range  | Selected value |
| -------------- | ---------------------------------------------------------------------- | ------ | -------------- |
| `a0`         | Base strictness on every day                                           | ≥ 0   | **0.2**  |
| `severity_i` | How severe the heat is on day *i* (Chapter 8.3)                      | 0 to 1 | from data      |
| `a_heat`     | How much extra strictness full heat severity adds                      | ≥ 0   | **12**   |
| `scarcity_i` | How close day *i*'s true peak is to the 365-day record (Chapter 8.4) | 0 to 1 | from data      |
| `a_scarcity` | How much extra strictness a record-level day adds                      | ≥ 0   | **6**    |

So:

* A cool winter day: severity 0, scarcity 0, so **a = 0.2**, which is almost ordinary squared error.
* A moderately hot day with severity 0.25: **a = 0.2 + 12 × 0.25 = 3.2**.
* 30-May-2024 (severity 1, scarcity 1): **a = 0.2 + 12 + 6 = 18.2**, extremely strict against shortfall.

In the 2019–2023 training data, **55.5 %** of usable days have a = 0.2 exactly (no heat, no scarcity), **24.1 %** have a > 1, and the
top 1 % have a > 14.8.

### 8.3 Where "severity" comes from

1. For each state and day: `heat_severity` = the larger of
   * the **IMD part**: (share of the state's area in heatwave + share in *severe* heatwave) ÷ 2, and
   * the **EHF part**: EHF ÷ (the state's 85th-percentile EHF) ÷ 3, capped at 1.
2. National `heat_severity_idx` = Σ over states of (state's share of national demand × state's heat_severity). A heatwave over UP
   (10 % of demand) counts ten times more than one over Uttarakhand (1 %).
3. `loss_severity` = heat_severity_idx ÷ 0.161, capped to [0, 1]. The number 0.161 is the 99th percentile of heat_severity_idx in
   the training years, so roughly the top 1 % of heat days get the full value 1.

### 8.4 Where "scarcity" comes from

```
record_365 = the highest daily peak in the previous 365 days
scarcity   = clip( (peak_today / record_365 − 0.95) / 0.05 , 0, 1 )
```

* Peak below 95 % of the record gives scarcity 0.
* Peak equal to or above the record gives 1.
* *Example:* on 28-May-2024 the peak was 234.64 GW and the record of the previous 365 days was 237.36 GW (set in 2023):
  234.64 / 237.36 = 0.9886, so scarcity = (0.9886 − 0.95) / 0.05 = **0.771**. On 29-May (241.9 GW, a new record) and 30-May
  (246.8 GW) scarcity = **1.0**.

**Why use the true peak here?** A loss function is only used **during training**, when the true answer is known anyway. Using it
lets the loss say "near-record days are more dangerous". At prediction time no loss is computed, so nothing leaks. (We also tested
an alternative that uses only last week's information; it was worse. See Chapter 26.)

**Real-life meaning:** near the all-time record there is no spare capacity left. A shortfall then means load-shedding, not just an
expensive market purchase.

### 8.5 Two safety guards: the Huber cap and the Hessian floor

1. **Huber cap (`u_max = 4`).** The exponential can explode: with a = 18.2 and e = 2 %, exp(36.4) would be about 6 × 10¹⁵. A single
   odd day could then dominate a whole tree. So once a·e passes 4, the exponential is continued as a **straight line**
   (`exp(4)·(1 + u − 4)`), which keeps the gradient finite and smooth. exp(4) ≈ 54.6, so the curvature is capped at 54.6× that of squared error.
2. **Hessian floor (`h_min = 0.05`).** On the over side the curvature exp(a·e) approaches 0. Dividing by nearly 0 in the leaf
   formula (Chapter 6.3) would create giant leaf values. So the Hessian is never allowed below 0.05.

### 8.6 MW consistency (scale) and centring (offset)

* **Scale.** Every day's *g* and *h* are multiplied by `base_mw ÷ average base_mw` (training average 176,292 MW; the ratio ranges
  0.76–1.31). A 1 % miss in 2025 (240 GW base) is 2.4 GW, but in 2019 (160 GW) it is only 1.6 GW, so later days count more.
  The loss then tracks megawatts, i.e. real cost.
* **Offset.** LightGBM's custom objectives start from a prediction of 0. We subtract the (weighted) average target of the training
  days (**0.092 %**) before training and add it back afterwards. For a fair comparison we do this for **every** model.

### 8.7 What the loss, gradient and Hessian look like: real numbers from our code

Computed with the selected settings (a0 = 0.2, a_heat = 12, a_scarcity = 6). Remember **e = actual − forecast**; positive e = short.
The gradient is the slope with respect to the *forecast*: negative = "raise the forecast".

**Ordinary day (severity 0, scarcity 0, a = 0.2): almost ordinary squared error**

| e (%)    | −2 (over) | −1    | −0.5  | +0.5    | +1      | +2 (short) |
| -------- | ---------- | ------ | ------ | ------- | ------- | ---------- |
| loss     | 1.758      | 0.468  | 0.121  | 0.129   | 0.535   | 2.296      |
| gradient | +1.648     | +0.906 | +0.476 | −0.526 | −1.107 | −2.459    |
| Hessian  | 0.670      | 0.819  | 0.905  | 1.105   | 1.221   | 1.492      |

Short by 2 % costs 2.30 against 1.76 for over by 2 %: only **1.3×** worse. This is mild, as intended.

**Moderately hot day (severity 0.25, a = 3.2)**

| e (%)    | −2    | −1    | −0.5  | +0.5    | +1      | +2               |
| -------- | ------ | ------ | ------ | ------- | ------- | ---------------- |
| loss     | 0.528  | 0.219  | 0.078  | 0.230   | 1.986   | **32.761** |
| gradient | +0.312 | +0.300 | +0.249 | −1.235 | −7.354 | −57.698         |
| Hessian  | 0.050  | 0.050  | 0.202  | 4.953   | 24.533  | 54.598           |

Short by 2 % now costs **62×** more than over by 2 %. Note how the over-forecast gradient is tiny (≈ 0.3): the model barely cares
about being a little over on such days.

**Record heatwave day like 30-May-2024 (severity 1, scarcity 1, a = 18.2)**

| e (%)    | −2    | −1    | −0.5  | +0.5     | +1       | +2               |
| -------- | ------ | ------ | ------ | -------- | -------- | ---------------- |
| loss     | 0.107  | 0.052  | 0.024  | 3.119    | 19.066   | **91.908** |
| gradient | +0.055 | +0.055 | +0.055 | −18.244 | −45.544 | −100.142        |
| Hessian  | 0.050  | 0.050  | 0.050  | 54.598   | 54.598   | 54.598           |

Here even being short by half a percent costs 3.1, while being over by 2 % costs only 0.107. The Huber cap is active (the Hessian
sits at its cap of 54.6). In plain words, on the hottest record days HALO says "**do not be short**".

### 8.8 The same toy leaf as Chapter 6.4, now under HALO

Three days in a leaf, current prediction 0 %, true targets +3 %, +1 %, −1 %, λ = 2:

| Loss                          | Gradients                  | Hessians              | Leaf value      |
| ----------------------------- | -------------------------- | --------------------- | --------------- |
| Squared error                 | −3, −1, +1               | 1, 1, 1               | **0.60**  |
| HALO, ordinary days (a = 0.2) | −4.111, −1.107, +0.906   | 1.822, 1.221, 0.819   | **0.735** |
| HALO, hot days (a = 3.2)      | −112.296, −7.354, +0.300 | 54.598, 24.533, 0.050 | **1.47**  |

On ordinary days HALO's leaf is close to squared error's (0.735 vs 0.6). On hot days it more than doubles, so the model
deliberately leans upward exactly where heat makes shortfalls dangerous.

### 8.9 Why LINEX gives a *smart* safety margin

If tomorrow's outcome is uncertain with typical spread σ (sigma), the forecast that minimises the average LINEX loss is approximately

```
best forecast ≈ average outcome + (a / 2) × σ²
```

We verified this numerically in notebook 06:

| σ | a   | best forecast found by search | a·σ²/2 |
| -- | --- | ----------------------------- | --------- |
| 1  | 0.2 | 0.10                          | 0.10      |
| 1  | 0.6 | 0.30                          | 0.30      |
| 2  | 0.2 | 0.40                          | 0.40      |
| 2  | 0.6 | 1.20                          | 1.20      |
| 3  | 0.2 | 0.91                          | 0.90      |
| 3  | 0.6 | 2.69                          | 2.70      |

**Why this matters:** the margin grows with **both** strictness (a) **and** uncertainty (σ²). Heatwave days are high-stakes (big a)
**and** hard to predict (big σ), so they get the biggest margin, while calm, predictable days get almost none. That is how a
sensible grid operator thinks: "keep more spare power on days that are both risky and uncertain". The pinball loss adds the
**same** margin every day, and squared error adds none.

### 8.10 How HALO is plugged into LightGBM, step by step

1. `models.add_loss_side_info` computes `loss_severity` and `loss_scarcity` for every day.
2. `models.fit_halo` creates a LightGBM dataset of training days with target `z − offset`.
3. `HALOObjective.register(...)` attaches each day's strictness `a_i` and weight (scale × training weight) to that dataset.
4. LightGBM calls `HALOObjective.objective(preds, data)` before every tree. It returns *g* and *h* (Chapter 8.7) for all days.
5. After every tree, LightGBM calls `HALOObjective.feval(preds, valid_data)`, the average HALO loss on 2024. If it does not improve for
   300 trees, training stops. HALO stopped at **373** trees.
6. Predictions: `z_hat = booster.predict(X) + offset`, then **MW forecast = base_mw × (1 + z_hat / 100)**.

### 8.11 Checks that the maths is right (`tests/test_losses.py`)

A wrong gradient does not crash LightGBM; it silently trains a bad model. So we test:

1. **Gradient check:** compare our formula with a numerical slope (tiny nudges). The maximum difference is about 9 × 10⁻⁸.
2. **Reduces to squared error:** with a tiny `a`, the loss equals e²/2.
3. **Asymmetry:** short by 4 % costs more than 4× over by 4 % (at a = 0.6).
4. **Monotone:** strictness grows with heat severity.
5. **Safe:** the Hessian is always positive and the gradient stays bounded even for e = ±50 %.

All 5 tests pass.

### 8.12 HALO compared with the team's earlier prototype loss

The prototype (`notebooks/reference/ML_Course_PROJECT__Prototype.ipynb`, kept as the `lgbm_prototype` baseline) multiplied the
gradient by 21 on under-forecast heatwave days and by 4 on other under-forecasts, with the Hessian fixed at 1.

|                                       | Prototype                                | HALO                                                            |
| ------------------------------------- | ---------------------------------------- | --------------------------------------------------------------- |
| Heat signal                           | Yes/no switch                            | **Continuous** severity 0–1 (IMD + EHF, demand-weighted) |
| Grid stress                           | Not used                                 | **Scarcity** term                                         |
| Shape                                 | Straight multiplier                      | Exponential for big shortfalls, mild for small                  |
| Hessian                               | Fixed at 1 (curvature thrown away)       | True curvature, with safety cap and floor                       |
| Accounts for MW size                  | No                                       | Yes (scale)                                                     |
| Backtest result on heat/critical days | Short 62.7 % of days, cost ₹7.60 cr/day | Short **52.5 %**, cost **₹6.45 cr/day**            |

---

# Part C: Data

## 9. The datasets and why each one was chosen

All datasets are public and India-specific, and need no login. Download code: `src/heatwave_peak/data_download.py`.

### 9.1 Dataset A: Kaggle "Hourly Load India" (Grid-India data)

* **What:** hourly demand (MW) for All-India plus the 5 regions, 1-Jan-2019 to 30-Apr-2024. **46,728 rows × 7 columns**.
* **Why:** the longest clean hourly series. It also contains the regional split, which we use for exploration.
* **Check performed:** national = sum of the 5 regions (to within ±0.02 MW), and no missing or duplicate hours.
* **Stored in:** `data/raw/load/kaggle_hourly_load_india/hourlyLoadDataIndia.xlsx` (plus `monthly_temp.xlsx`, included in the download, not used).

### 9.2 Dataset B: Mendeley Data "Electricity Demand, Solar and Wind Generation of India at 1-hour interval"

* **What:** published by NERLDC (a Grid-India regional centre) with IIT Guwahati, licence CC-BY-4.0. We use the file
  `January 2024- June 2025.xlsx`: hourly All-India demand, **12,984 hours**, 1-Jan-2024 to 24-Jun-2025.
* **Why it is essential:** May–October 2024 exists **only** here, and that period contains the record heatwave.
* **Stored in:** `data/raw/load/mendeley_nerldc_hourly/`.

### 9.3 Dataset C: Kaggle "India Power Grid – NLDC Daily PSP Reports"

Scraped from Grid-India's daily reports and updated daily. Four files:

| File                  | Size                                | Content                                                                                                | How we use it                       |
| --------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------- |
| `study1_hourly.csv` | 46,728 rows × 151 cols             | Same hourly load as A, joined with 144 daily grid-operations columns                                   | Inventory / exploration             |
| `study1_daily.csv`  | ~2,760 rows × 144 cols             | Daily: generation by fuel, outages, regional flows, frequency, official **daily maximum demand** | Fallback peaks; 3 grid-ops features |
| `study2_scada.csv`  | 64,860 rows (15-minute) × 164 cols | Live-system demand from 4-Nov-2024                                                                     | Hourly demand Nov-2024 onward       |
| `study3_states.csv` | ~102,000 rows                       | **Daily maximum demand of every state**                                                          | State demand weights, clustering    |

### 9.4 Dataset D: IMD gridded daily Tmax/Tmin (Government of India)

* **What:** daily maximum and minimum temperature on a 1° × 1° grid (31 × 31 squares) covering India, built by IMD from about 395 weather
  stations. **35 years (1991–2025) × 2 variables**, one binary `.GRD` file per year (70 files).
* **Why:** it is the **official** dataset IMD itself uses for heatwave climatology. Using it means our heatwave labels follow the
  Government's definition, not one we invented.
* **Why from 1991 when we model 2019–2025:** a heatwave is defined relative to the **30-year normal** (1991–2020).
* **Tool:** the `imdlib` Python package downloads the files and also implements IMD's heatwave rule and state boundaries.
* **Stored in:** `data/raw/weather/imd_gridded/tmax/1991.GRD … 2025.GRD` and `…/tmin/…`.

### 9.5 Dataset E: ERA5 hourly weather via the Open-Meteo archive

* **What:** hourly weather at each state's main load centre (for example Lucknow for UP, Chennai for Tamil Nadu),
  1-Dec-2018 to 31-Dec-2025, in Indian time. **33 states × 62,112 hours × 8 variables ≈ 2.05 million rows.**
* **The 8 variables, and why each:**

| Variable             | Why it matters for electricity                                                              |
| -------------------- | ------------------------------------------------------------------------------------------- |
| temperature_2m       | Heat drives air-conditioner load                                                            |
| relative_humidity_2m | Humid heat feels worse, so more AC                                                          |
| dew_point_2m         | Absolute stickiness, especially at night                                                    |
| apparent_temperature | "Feels-like" temperature                                                                    |
| precipitation        | Rain cools the air and stops irrigation pumps (agriculture is a big share of Indian demand) |
| cloud_cover          | Clouds reduce heat and rooftop-solar output                                                 |
| shortwave_radiation  | Sunlight, which drives rooftop solar that hides demand                                      |
| wind_speed_10m       | Wind helps desert coolers and ventilation                                                   |

* **Why we need it on top of IMD:** IMD gives only two numbers a day. ERA5 tells us **when** it was hot (afternoon versus night) and **how humid**.
* **Stored in:** `data/raw/weather/era5_openmeteo_hourly/era5_hourly_<State>.csv`.

### 9.6 Reference table: the 33 states

`data/raw/reference/state_metadata.csv` (written by `write_state_reference_table`), one row per state with:

* the name IMD uses,
* the name(s) NLDC uses (e.g. "UP"; Jharkhand = "Jharkhand" + "DVC"; DNH&DD = "DD" + "DNH" + "DNHDDPDCL"),
* its grid region,
* its load-centre city and coordinates.

Islands (Andaman, Lakshadweep) are excluded because they are not connected to the national grid.

## 10. Folder structure, file formats, and why Parquet

### 10.1 The project tree

```
ml-project/
├── PROJECT_GUIDE.md                 ← this guide
├── README.md                        ← short summary + results
├── pyproject.toml, requirements.txt ← installable package + library list
├── data/
│   ├── raw/        untouched downloads (never edited)
│   ├── interim/    cleaned per-source tables
│   └── processed/  model-ready tables, predictions
├── notebooks/      01 … 08 (executed, with outputs) + reference/ (old prototype)
├── src/heatwave_peak/  all project code (12 modules)
├── scripts/        run_pipeline.py, run_notebooks.py
├── models/         6 trained LightGBM models (.txt)
├── reports/
│   ├── figures/    25 figures (.png)
│   ├── metrics/    all result tables (.csv, .json)
│   └── NOVELTY_CLAIMS.md
└── tests/          test_losses.py
```

**Why three data folders?** It is the standard "raw → interim → processed" ladder:

* **raw** is never touched, so we can always go back to the original;
* **interim** holds each source cleaned, but not yet combined for modelling;
* **processed** holds exactly what the models read.

If a cleaning rule changes, only interim and processed need rebuilding.

### 10.2 Which formats are used where

| Where                                  | Format                                                  | Why                                                            |
| -------------------------------------- | ------------------------------------------------------- | -------------------------------------------------------------- |
| `data/raw/`                          | Whatever the source gave: `.xlsx`, `.csv`, `.GRD` | Untouched copies of the originals                              |
| `data/interim/`, `data/processed/` | **`.parquet`** (plus two small `.csv`)        | See below                                                      |
| `reports/metrics/`                   | `.csv`, `.json`                                     | Small tables meant for people (open in Excel or a text editor) |
| `models/`                            | `.txt`                                                | LightGBM's own human-readable model format                     |

### 10.3 Why Parquet (with measurements from this project)

**Parquet** is a file format that stores a table **column by column**, compresses each column, and **remembers the data type of
every column** (date, number, true/false).

We measured it on our biggest table (`era5_hourly_states_2018_2025.parquet`, 2,049,696 rows):

|                                   | CSV                            | Parquet                                   |
| --------------------------------- | ------------------------------ | ----------------------------------------- |
| File size                         | 186 MB                         | **20 MB** (about 9× smaller)       |
| Time to read                      | 0.44 s                         | **0.14 s** (about 3× faster)       |
| `datetime` column after reading | plain **text** (`str`) | real **date-time** (`datetime64`) |

**The three reasons in plain words:**

1. **Smaller.** Columns like "state name" repeat the same value 62,112 times. Parquet stores it once with a count. The whole
   interim + processed folders take ~68 MB instead of several hundred.
2. **Types survive.** This is the most important reason. A CSV is just text. Save a date and read it back, and you get a string; a true/false
   flag like `target_imputed` can come back as the text "True". If one notebook forgot to convert a column, a comparison such as
   `date >= "2024-01-01"` would quietly become a **text** comparison and give wrong results with no error. Parquet stores the
   type with the data, so notebook 07 reads exactly what notebook 05 wrote.
3. **Wide tables load fast.** `model_table.parquet` has 320 columns. Parquet can read only the columns you ask for:
   `pd.read_parquet(path, columns=["peak_mw", "target_z"])`.

**Why some outputs are still CSV:** `state_demand_weights.csv`, `state_signatures_and_zones.csv` and everything in
`reports/metrics/` are small and meant to be **read by people** (for the report, or opened in Excel).

**How to open a Parquet file:**

```python
import pandas as pd
df = pd.read_parquet("data/processed/model_table.parquet")
df.to_csv("model_table.csv")          # if you ever need a CSV copy
```

---

# Part D: The end-to-end workflow, stage by stage

## 11. The whole pipeline at a glance

```
 ┌──────────────────────────── STAGE 1: DOWNLOAD ─────────────────────────────┐
 │ Kaggle hourly load (A) · Mendeley hourly load (B) · NLDC PSP reports (C)    │
 │ IMD Tmax/Tmin 1991-2025 (D) · ERA5 hourly for 33 states (E)                 │
 └──────────────────────────────────┬──────────────────────────────────────────┘
                                    ▼
 STAGE 2: LOAD CLEANING                     STAGE 3: WEATHER + HEATWAVES
 remove sensor glitches                     IMD grid → state averages
 calibrate B and C to A's level             1991-2020 normals, departures
 stitch A → B → C (2019-2025 hourly)        IMD heatwave rule + EHF
 daily peak = max hour of each day          ERA5 hourly → daily features
 critical-peak days, state weights          national heat exposure
                    │                                    │
                    └────────────────┬───────────────────┘
                                     ▼
                     STAGE 4: CLUSTER 33(28 States+5 U.T) regions → 8 ZONES
                                     ▼
                     STAGE 5: FEATURES (309 per day) + TARGET
                     target = % above last week's average peak
                                     ▼
                     STAGE 6: TRAIN 6 MODELS (only the loss differs)
                     tune HALO on 2024 → refit on 2019-2024 → test 2025
                     + walk-forward backtest 2022-2025
                                     ▼
                     STAGE 7: EVALUATE (accuracy, peak risk, ₹ cost)
                                     ▼
                     STAGE 8: EVIDENCE (significance, fair comparison, ablations)
```

Each stage is a function in `src/heatwave_peak/`, explained in a notebook, and runnable from the command line:
`python scripts/run_pipeline.py --stages download load weather cluster features train evidence`.

## 12. Stage 1: Downloading

**Code:** `data_download.py` → `download_all()`. **Notebook:** 01.

| Step                              | What happens                                                                        | Detail                                                                                                                          |
| --------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `write_state_reference_table()` | Saves the 33-state table to `data/raw/reference/state_metadata.csv`               | From `config.STATES`                                                                                                          |
| `download_load_datasets()`      | Downloads Kaggle A and C as zip files and unpacks them; downloads the Mendeley file | Public Kaggle datasets download without a key from `kaggle.com/api/v1/datasets/download/<owner>/<name>`                       |
| `download_imd_gridded()`        | Downloads 1991–2025 Tmax and Tmin via `imdlib.get_data`                          | Retries up to 4 times if the IMD server hiccups                                                                                 |
| `download_era5_hourly()`        | One request per state to the Open-Meteo archive                                     | Waits 20 s between states; if the server says "too many requests" (HTTP 429), waits 90 s, 180 s, … and retries (up to 8 times) |

**Why "idempotent" (safe to re-run):** every function first checks whether the file already exists and skips it if so. Re-running
the notebook never re-downloads ~380 MB.

**Why IMD 2026 is not used:** IMD has not published the 2026 yearly file (the server returns an empty file). So the study ends on
31-Dec-2025.

## 13. Stage 2: Cleaning the electricity data and finding the daily peak

**Code:** `load_processing.py`. **Notebook:** 02.

### 13.1 Reading each source (`read_kaggle_hourly`, `read_mendeley_hourly`, `read_scada_hourly`, `read_psp_daily`)

* **A:** 46,728 hours, 2019-01-01 00:00 → 2024-04-30 23:00, no gaps.
* **B:** 12,984 hours, 2024-01-01 → 2025-06-24. The Excel sheet ends with three summary rows ("Average", "Sum" …). They fail
  date-parsing and are dropped automatically.
* **C:** 15-minute readings → hourly averages. An hour must have at least **3 of its 4** quarter-hours, otherwise it is left empty (some
  report days are corrupted). Zeros are treated as missing. Result: 16,680 hours (4-Nov-2024 → 29-Sep-2026), 464 of them empty.

### 13.2 Removing sensor glitches, but keeping real events (`flag_spikes`)

**Rule.** An hour is a glitch if it is **both**

* more than **10 %** away from the average of the hour before and the hour after, **and**
* more than **15 %** away from the median of the *same clock hour* over the surrounding 7 days,

**or** more than **30 %** away from that 7-day median.

**Why two conditions?** India has genuine one-off events that a simple outlier filter would wrongly delete:

* **5-Apr-2020, 21:00:** the "lights off for 9 minutes" appeal dropped demand by about 31 GW. It is only ~10 % from its neighbours. **Kept.**
* **Holi (25-Mar-2024)** and **Independence Day** afternoons: genuine 15–17 % dips. **Kept**, because they are not isolated single-hour spikes.

**Removed:** 17-Dec-2021 05:00 (a one-hour +17 % spike) and 13-Jul-2024 13:00. Only **2** hours in 7 years.

### 13.3 Calibrating the sources to one scale (`overlap_ratio`)

Different instruments read slightly differently. On the hours where two sources overlap we take the **median** of the ratio (the
median ignores the occasional glitch):

| Overlap                       | Hours | Finding                         | Correction                               |
| ----------------------------- | ----- | ------------------------------- | ---------------------------------------- |
| A vs B (Jan–Apr 2024)        | 2,903 | B reads **0.77 % higher** | B × **0.9923**                    |
| B vs C (Nov-2024 – Jun-2025) | 5,392 | C reads 0.11 % higher than B    | C × 0.9923 × 0.9989 = **0.9912** |

*Why bother with less than 1 %?* 0.8 % of 200 GW is **1.6 GW**, the size of a big power station and about a third of the model's
typical error. Without correction, the model would see a fake "jump" in demand on 1-May-2024, exactly where the sources switch.

### 13.4 Stitching (`build_national_hourly`)

1. Use A up to 30-Apr-2024 23:00, then calibrated B, then calibrated C from 4-Nov-2024.
2. Where C is missing but B exists (until 24-Jun-2025), fill from B. A real measurement beats a guess.
3. Re-run the glitch filter on the stitched series.
4. Fill gaps of **3 hours or less** by straight-line interpolation in time. Longer gaps (whole lost report days) stay empty.

**Result:** `data/interim/load_hourly_national_2019_2025.parquet`, **61,368 hours** (2019-01-01 → 2025-12-31). Sources: A 46,728,
C 9,960, B 4,680 hours. 29 short gaps were interpolated; 197 hours (whole lost days) remain empty.

### 13.5 The daily peak, our target (`build_daily_peaks`)

For every day:

| Column              | Meaning                                       | 30-May-2024 |
| ------------------- | --------------------------------------------- | ----------- |
| `peak_mw`         | Highest hourly demand **(the target)**  | 246,779 MW  |
| `peak_hour`       | Clock hour of the peak                        | 14          |
| `min_mw`          | Lowest hour (night base load)                 | 209,705 MW  |
| `mean_mw`         | Average (= daily energy ÷ 24)                | 225,661 MW  |
| `solar_peak_mw`   | Highest hour between 09:00 and 16:00          | 246,779 MW  |
| `evening_peak_mw` | Highest hour between 17:00 and 23:00          | 229,459 MW  |
| `load_factor`     | mean ÷ peak                                  | 0.914       |
| `n_valid_hours`   | Hours with data                               | 24          |
| `target_imputed`  | Was the peak filled from the official report? | False       |

**Why the peak and not the daily energy:** grid failures happen at the peak moment, when every power station must be online.
Energy tells you how much coal to burn; the peak tells you whether you have enough plants running at 3 pm.

**Fallback for days with fewer than 20 valid hours (10 days, all in 2025):** NLDC's official "maximum demand met" for that day is used,
multiplied by **0.988**. The official number is an instantaneous maximum, about 1.2 % higher than a maximum of hourly averages; 0.988 is
the median ratio on days where both exist. These 10 days (28-Jun, 13-Jul, 15-Jul, 29-Jul, 7-Aug, 20-Aug, 21-Aug, 9-Sep, 2-Oct and 10-Nov-2025) are
**flagged, never used for training, and never scored**. They exist only so that "yesterday's peak" is defined for the day after.

**The peak hour moved over the years** (median peak hour):

| Year | Winter (Dec–Feb) | Pre-monsoon (Mar–May) | Monsoon (Jun–Sep) | Post-monsoon (Oct–Nov) |
| ---- | ----------------- | ---------------------- | ------------------ | ----------------------- |
| 2019 | 9                 | 19                     | 20                 | 18                      |
| 2020 | 9                 | 15                     | 20                 | 18                      |
| 2021 | 10                | 11                     | 12                 | 18                      |
| 2022 | 10                | 11                     | 19                 | 11                      |
| 2023 | 10                | 11                     | 14                 | 11                      |
| 2024 | 10                | 11                     | 15                 | 11                      |
| 2025 | 10                | 15                     | 19                 | 18                      |

Winter always peaks in the morning (heating plus irrigation pumps). Summer peaks moved from the evening (lighting) to the
afternoon (air-conditioning at the hottest hour). That is why afternoon weather features (`era5_t_15h`) matter.

### 13.6 Critical peak days (`add_critical_peak_flag`)

**Definition:** a day is a *critical peak day* if its peak is above the **90th percentile of the previous 365 days' peaks** (at least
180 days of history are required).

**Why not a fixed number like "above 220 GW"?** Demand grows 6–8 % a year. 220 GW was a record in 2023 and a normal June day in
2025. A rolling threshold adjusts for growth, and it uses only the past, so an operator could compute it every morning.

**Counts per year:** 2019: 0 (not enough history yet) · 2020: 29 (COVID collapse) · 2021: 89 · 2022: 118 · 2023: 121 · 2024: 60 · 2025: 45.

### 13.7 State demand and state weights (`load_state_daily_demand`, `state_demand_weights`)

* From NLDC's state table: map NLDC names to IMD names, add up merged entities (Jharkhand + DVC; DD + DNH, which merged in 2022), and
  **drop industrial and railway entities** (AMNSIL, BALCO, Essar Steel, RIL Jamnagar, railway and bulk consumers). Those run all day
  regardless of weather and would only dilute the heat signal. Result: **90,994 state-days, 33 states**.
* **Weight of a state** = its average daily maximum demand in 2019–2023 ÷ the sum over all states. **Only training years** are used, so
  nothing from 2024–25 leaks in.

| State          | Weight | State                                       | Weight     |
| -------------- | ------ | ------------------------------------------- | ---------- |
| Maharashtra    | 0.119  | Karnataka                                   | 0.059      |
| Uttar Pradesh  | 0.099  | Telangana                                   | 0.053      |
| Gujarat        | 0.088  | Andhra Pradesh                              | 0.049      |
| Tamil Nadu     | 0.077  | Punjab                                      | 0.044      |
| Rajasthan      | 0.066  | West Bengal / Haryana                       | 0.041 each |
| Madhya Pradesh | 0.061  | … smallest (NE states, Sikkim, Chandigarh) | ≈ 0.001   |

**Why weights matter:** a heatwave over Lucknow (UP, 10 % of demand) moves the national peak far more than one over Dehradun
(Uttarakhand, 1 %). Every "national" or "zone" weather number in this project is a **demand-weighted** average.

### 13.8 Regional hourly (exploration only)

`build_regional_hourly` cleans A's five regional series (2019 → Apr-2024) and saves them to `load_hourly_regional_2019_2024.parquet`.
Notebook 02 uses them to show seasonality: NR peaks in summer, SR in March–April, WR in winter (irrigation). They are not model inputs.

## 14. Stage 3: Weather and deciding what a heatwave is

**Code:** `weather_processing.py`, `heatwave.py`. **Notebook:** 03.

### 14.1 IMD grid → state averages (`imd_state_daily`)

* `imdlib`'s `region(state=…)` averages all 1° squares inside each state, weighting each square by **how much of it lies inside the state**
  × cos(latitude) (squares are slightly smaller further north).
* The same is done for IMD's **cell-level heatwave classification**: `imdlib`'s `heatwave(output="daily")` labels every square every day as
  0 = none, 1 = heatwave, 2 = severe, using IMD's rule and 1991–2020 normals. We convert this to:
  * `hw_area_frac` = share of the state's area in heatwave (1 or 2),
  * `severe_hw_area_frac` = share in severe heatwave (2).
* Missing temperatures are filled from the previous day (up to 2 days; temperature is very persistent). Missing fractions become 0.
* Result: **421,872 state-days** (33 states × 12,784 days, 1991–2025).

**IMD's official heatwave rule (applied per grid square):**

1. **Entry gate:** Tmax ≥ 40 °C on plains, ≥ 37 °C on the coast, ≥ 30 °C in the hills.
2. Then **either** departure from normal ≥ 4.5 °C (severe if > 6.4 °C) **or** Tmax ≥ 45 °C (severe if ≥ 47 °C).

### 14.2 Normals and departures (`daily_normals`)

For each state and each day of the year, the **normal** is the 1991–2020 average for that date, smoothed with a 15-day moving window that
wraps around the year end (so 31-Dec sits next to 1-Jan). Smoothing uses ~450 values instead of ~30 and removes day-to-day noise.
`imd_tmax_departure` = today − normal. *Example:* 40 °C is normal in Delhi in mid-May (normal ≈ 39.5 °C) but would be a severe
anomaly in Bengaluru.

### 14.3 State-level features (`add_heatwave_features`)

| Feature                   | Rule                                                                    | Why                                                                                                                 |
| ------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `state_hw_day`          | ≥ 25 % of area in IMD heatwave **today and yesterday**           | IMD declares a heatwave only after **2 consecutive days**; looking only backwards means no future information |
| `state_severe_hw_day`   | heatwave day **and** ≥ 25 % of area severe                       | Severity                                                                                                            |
| `warm_night`            | Tmax ≥ 40 °C and Tmin departure ≥ 4.5 °C (IMD's warm-night rule)    | Buildings cannot cool overnight (Delhi, 18–19 June 2024: 35 °C nights)                                            |
| `hw_run_length`         | Consecutive heatwave days so far                                        | Persistence                                                                                                         |
| `hot_day_run_length`    | Consecutive days with Tmax ≥ 40 °C                                    | Long "normal" heat that still drives demand                                                                         |
| `heat_accum_ewm`        | Smoothed running total of heat above 35 °C (half-life 2 days)          | Four 44 °C days in a row score **≈ 2.6×** a single 44 °C day (6.75 vs 2.64)                               |
| `cdd24`, `cdd24_3d`   | Cooling degree-days above 24 °C on the daily mean, and their 3-day sum | Cooling need, with a building "thermal lag"                                                                         |
| `tmax_change_1d`        | Tmax today − yesterday                                                 | Sudden heat surges                                                                                                  |
| `ehf`, `ehf_severity` | Excess Heat Factor (14.4)                                               | Second heatwave criterion                                                                                           |
| `state_heat_day`        | **IMD-declared heatwave OR EHF > 0**                              | The final state heatwave label                                                                                      |
| `heat_day_run_length`   | Consecutive `state_heat_day`s                                         | Persistence                                                                                                         |
| `heat_severity`         | max(IMD part, EHF part), 0–1                                           | Feeds HALO (Chapter 8.3)                                                                                            |

**Why 25 % of the area?** Rajasthan covers about 34 grid squares. One hot square near Barmer is a local hot spot, not a state-wide heatwave.
25 % mirrors IMD's requirement that at least 2 stations in a sub-division meet the criteria.

### 14.4 Why IMD alone was not enough, and the Excess Heat Factor

**Measured problem:** with only the IMD rule, days where at least 15 % of national demand is in a heatwave:

| Year          | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025        |
| ------------- | ---- | ---- | ---- | ---- | ---- | ---- | ----------- |
| IMD-only days | 4    | 2    | 0    | 6    | 2    | 12   | **0** |

That is 26 days in 7 years, and **none in the 2025 test year**. The rule needs Tmax ≥ 40 °C, so it almost never fires in humid states
(Kerala, Tamil Nadu, West Bengal, the North-East). Yet humid heat sets real records: September 2023 and 2024 "monsoon-break" heat
pushed national demand to ~240 GW with almost no IMD heatwave.

**The Excess Heat Factor (EHF)**, used operationally by Australia's Bureau of Meteorology (Nairn & Fawcett, 2015):

```
DMT       = (Tmax + Tmin) / 2                      daily mean temperature, so it includes night heat
EHI_sig   = average DMT of the last 3 days − T95   T95 = the state's 95th-percentile DMT in 1991-2020
EHI_accl  = average DMT of the last 3 days − average DMT of the 30 days before
EHF       = max(EHI_sig, 0) × max(EHI_accl, 1)
EHF > 0  → heatwave;   EHF ÷ EHF85 ≥ 1 → severe;   ≥ 3 → extreme
```

* **EHI_sig** asks: is it hot compared with **this state's own** hot level? Kerala's T95 is 28.1 °C; Rajasthan's is 34.5 °C.
* **EHI_accl** asks: is it hotter than what people got **used to** over the past month?

**Final rule:** `state_heat_day = IMD-declared heatwave OR EHF > 0`.

### 14.5 National heat exposure (`national_heat_exposure`)

Demand-weighted sums over states (weights from 13.7):

| Column                  | Meaning                                                                  |
| ----------------------- | ------------------------------------------------------------------------ |
| `imd_hw_exposure`     | Share of national demand in **IMD-declared** heatwave states       |
| `heat_exposure`       | Share of national demand in **final** heatwave states (IMD or EHF) |
| `severe_exposure`     | Share in IMD **severe** heatwave states                            |
| `heat_severity_idx`   | Weighted average of `heat_severity` (0–1); feeds HALO                 |
| `warm_night_exposure` | Share in warm-night states                                               |
| `hot40_exposure`      | Share in states with IMD Tmax ≥ 40 °C                                  |
| `national_hw_regime`  | **1 if `heat_exposure` ≥ 0.20**: a "heatwave regime day"        |

**Choosing 20 %: the sensitivity table** (days per year above each threshold, final rule):

| Threshold         | 2019         | 2020         | 2021        | 2022         | 2023         | 2024         | 2025         | Total         |
| ----------------- | ------------ | ------------ | ----------- | ------------ | ------------ | ------------ | ------------ | ------------- |
| ≥ 10 %           | 68           | 30           | 10          | 49           | 40           | 65           | 23           | 285           |
| ≥ 15 %           | 67           | 25           | 2           | 43           | 30           | 59           | 14           | 240           |
| **≥ 20 %** | **53** | **19** | **2** | **39** | **23** | **55** | **14** | **205** |
| ≥ 25 %           | 49           | 16           | 0           | 34           | 17           | 53           | 11           | 180           |
| ≥ 30 %           | 44           | 13           | 0           | 30           | 11           | 48           | 9            | 155           |

20 % keeps heatwaves rare (about 8 % of days, as they should be), leaves 55 regime days in the 2024 validation year and 14 in the 2025
test year, and corresponds to roughly UP + Rajasthan + MP together: one big heat dome over the north-west.

**The record heatwave in numbers:**

| Date        | IMD-only exposure | Final heat exposure | Daily peak                  |
| ----------- | ----------------- | ------------------- | --------------------------- |
| 24-May-2024 | 0.07              | 0.43                |                             |
| 25-May      | 0.07              | 0.63                |                             |
| 26-May      | 0.08              | 0.65                | 218.8 GW (Sunday)           |
| 27-May      | 0.14              | 0.56                | 229.1 GW                    |
| 28-May      | **0.37**    | 0.54                | 234.6 GW                    |
| 29-May      | 0.38              | 0.59                | 241.9 GW                    |
| 30-May      | 0.30              | 0.70                | **246.8 GW (record)** |
| 31-May      | 0.24              | 0.81                | 244.9 GW                    |

EHF flagged the event **four days before** the official IMD declaration took effect.

### 14.6 ERA5 hourly → daily (`aggregate_era5_daily`, `heat_index_c`)

For every state and day, from 24 hourly rows:

| Feature                                      | How it is computed                                                  |
| -------------------------------------------- | ------------------------------------------------------------------- |
| `era5_tmax`, `era5_tmean`, `era5_tmin` | Highest, average and lowest hourly temperature                      |
| `era5_night_tmin`                          | Lowest temperature between 00:00 and 06:00                          |
| `era5_t_15h`, `era5_rh_15h`              | Temperature and humidity at 15:00 (the summer peak hour)            |
| `era5_rh_mean`, `era5_dewpoint_mean`     | Daily averages                                                      |
| `era5_apparent_max`                        | Highest "feels-like" temperature                                    |
| `era5_heat_index_max`                      | Highest NOAA heat index (formula below)                             |
| `era5_cdh24`                               | Sum over hours of max(T − 24, 0): cooling degree-hours             |
| `era5_hours_ge_40`                         | Hours at ≥ 40 °C                                                  |
| `era5_hours_hi_ge_41`                      | Hours with heat index ≥ 41 °C (US weather service "danger" level) |
| `era5_precip_sum`, `era5_rain_hours`     | Daily rain (mm) and hours with ≥ 0.5 mm                            |
| `era5_cloud_mean`                          | Average cloud cover %                                               |
| `era5_solar_kwh_m2`                        | Sum of sunlight ÷ 1000 (kWh per m²)                               |
| `era5_wind_mean`                           | Average wind speed                                                  |

**Heat index (`heat_index_c`):** the US National Weather Service formula (Rothfusz regression, with the simpler Steadman formula when it
is mild, plus their two corrections for very dry and very humid air), converted to °C. *Example from the data:* Tamil Nadu (Chennai),
2-May-2024: Tmax 38.3 °C at 63 % afternoon humidity gives a heat index of **50.2 °C**.

Result: **85,404 state-days** of ERA5 features (33 × 2,588 days). Merged with the IMD features into
`data/interim/state_daily_weather_1991_2025.parquet`: **421,872 rows × 45 columns**. ERA5 columns are empty before Dec-2018, which is
fine because only the IMD columns are needed for the 1991–2020 normals.

## 15. Stage 4: Grouping states into zones (clustering)

**Code:** `clustering.py`. **Notebook:** 04.

### 15.1 Why cluster at all

The target is **one** national number, but weather comes from 33 states.

* Giving the model 33 states × ~24 weather columns ≈ 800 features for ~1,700 training days invites memorising noise.
* Averaging everything nationally would cancel a Rajasthan heatwave against a rainy Kerala.

Grouping similar states into a few zones was the planned compromise. (The ablation in Chapter 19 later showed that, for the national
target, a demand-weighted national average does just as well. We report that honestly.)

**Why not simply use the 5 official grid regions?** They are electrical and administrative, not climatic. NR mixes desert Rajasthan (47 °C)
with snowy, winter-peaking Himachal.

### 15.2 Each state's "signature" (`build_state_signatures`): 19 numbers per state

| Group                              | Features (and what they mean)                                                                                                                                                                                                                                                                                            |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Climate, IMD 1991–2020            | `tmax_annual_mean`; `tmax_seasonal_amplitude` (hottest month − coldest month); `tmax_premonsoon_AMJ` (April–June average); `diurnal_range` (Tmax − Tmin); `dmt_p95`; `heat_days_per_year`; `hot_month_sin`/`cos` (which month is hottest, as a position on a circle so December sits next to January) |
| Humidity and rain, ERA5 2019–2023 | `rh_mean`; `dewpoint_premonsoon`; `heat_index_AMJ`; `monsoon_rain_share` (share of rain in June–September)                                                                                                                                                                                                      |
| Demand behaviour, NLDC 2019–2023  | `summer_winter_ratio`; `monsoon_dip` (Jul–Sep vs Apr–Jun demand); `peak_demand_month_sin`/`cos`; `heat_sensitivity_pct_per_C` (% change in demand per +1 °C Tmax, March–June, from a straight-line fit)                                                                                                    |
| Geography                          | `lat`, `lon` (of the load centre)                                                                                                                                                                                                                                                                                    |

Demand is first divided by the state's own 365-day rolling average, so growth is not mistaken for seasonality, and the COVID lockdown is excluded.

**Real examples from the signatures:**

| State            | Heat sensitivity (% per °C) | Summer/winter demand | Meaning                                       |
| ---------------- | ---------------------------- | -------------------- | --------------------------------------------- |
| Kerala           | +4.28                        | 1.02                 | Humid, AC-driven                              |
| West Bengal      | +4.11                        | 1.34                 | Strong summer peak                            |
| Delhi            | +3.26                        | 1.23                 | Classic AC city                               |
| Rajasthan        | +0.75                        | 0.83                 | **Winter** peak (rabi irrigation pumps) |
| Madhya Pradesh   | +0.26                        | 0.72                 | Winter irrigation peak                        |
| Himachal Pradesh | −0.67                       | 0.82                 | Heating-driven; heat *lowers* demand        |

### 15.3 Scaling (`scale_signatures`)

Each feature is rescaled to average 0 and spread 1 (otherwise "rain in mm" would outweigh "a ratio near 1"). Missing values are filled
with the median. Latitude and longitude are then multiplied by **1.5** so zones come out geographically connected: heatwaves are
500–1,500 km weather systems, so a zone should share the same weather on the same day.

### 15.4 Four algorithms × K = 3…8 (`compare_clusterings`)

| Algorithm                             | Idea                                                                                                       |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| k-means                               | Places K centres and assigns each state to the nearest one                                                 |
| Ward                                  | Starts with each state alone and repeatedly merges the two groups whose merge adds the least spread        |
| **Ward + spatial connectivity** | Same as Ward, but a state may merge only with its 4 nearest geographic neighbours, so zones are contiguous |
| Gaussian mixture                      | Fits K bell-shaped clouds and gives soft memberships                                                       |

**Scores** (all computed by scikit-learn):

* **Silhouette** (−1…1, higher is better): are states closer to their own zone than to the next zone?
* **Calinski-Harabasz** (higher is better): spread between zones ÷ spread within zones.
* **Davies-Bouldin** (lower is better): how much zones overlap.
* **ARI / NMI** against the official grid regions: agreement, for information only.

Silhouette scores found:

| K | GMM   | k-means | Ward  | Ward-spatial |
| - | ----- | ------- | ----- | ------------ |
| 3 | 0.297 | 0.296   | 0.265 | 0.243        |
| 4 | 0.310 | 0.316   | 0.276 | 0.244        |
| 5 | 0.333 | 0.345   | 0.306 | 0.307        |
| 6 | 0.344 | 0.353   | 0.353 | 0.353        |
| 7 | 0.336 | 0.361   | 0.361 | 0.361        |
| 8 | 0.319 | 0.361   | 0.361 | 0.361        |

### 15.5 Selection rule (`select_clustering`)

Reject any solution with a one-state "zone", rank the rest on each of the three scores, and pick the best **average rank**. A single
score is biased (silhouette likes few zones, Calinski-Harabasz likes many). **Winner: Ward + spatial connectivity, K = 8**
(silhouette 0.361, Calinski-Harabasz 16.4, Davies-Bouldin 0.816, ARI with grid regions 0.42). For K ≥ 6, three algorithms give
identical groupings, a sign the structure is stable.

### 15.6 The 8 zones (`name_zones` names each zone after its two largest-demand states)

| Zone                   | Members                                                                                     | Share of demand | Character                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------- | --------------- | --------------------------------------------------------------------------------- |
| Z0_Maharashtra+Gujarat | Chhattisgarh, Gujarat, Jharkhand, Madhya Pradesh, Maharashtra, Rajasthan, Telangana, DNH&DD | 0.44            | Hot semi-arid, irrigation-heavy (low heat sensitivity 0.80 %/°C, winter-leaning) |
| Z1_Karnataka+Kerala    | Goa, Karnataka, Kerala                                                                      | 0.08            | Humid south-west, high sensitivity 3.22 %/°C                                     |
| Z2_Meghalaya+Manipur   | Arunachal, Manipur, Meghalaya, Mizoram, Nagaland, Sikkim                                    | 0.01            | Cool hills                                                                        |
| Z3_Tamil+Andhra        | Andhra Pradesh, Puducherry, Tamil Nadu                                                      | 0.13            | Hot humid south-east coast                                                        |
| Z4_Jammu+Uttarakhand   | Himachal, J&K, Uttarakhand                                                                  | 0.03            | Himalayan, heating-driven                                                         |
| Z5_Uttar+Punjab        | Chandigarh, Delhi, Haryana, Punjab, Uttar Pradesh                                           | 0.21            | Indo-Gangetic north-west: summer peak, 2.82 %/°C                                 |
| Z6_Assam+Tripura       | Assam, Tripura                                                                              | 0.01            | Humid north-east                                                                  |
| Z7_West+Bihar          | Bihar, Odisha, West Bengal                                                                  | 0.09            | Humid east, strong summer peak                                                    |

**Against the official grid regions** (number of states):

| Zone | ER | NER | NR | SR | WR |
| ---- | -- | --- | -- | -- | -- |
| Z0   | 1  | 0   | 1  | 1  | 5  |
| Z1   | 0  | 0   | 0  | 2  | 1  |
| Z2   | 1  | 5   | 0  | 0  | 0  |
| Z3   | 0  | 0   | 0  | 3  | 0  |
| Z4   | 0  | 0   | 3  | 0  | 0  |
| Z5   | 0  | 0   | 5  | 0  | 0  |
| Z6   | 0  | 2   | 0  | 0  | 0  |
| Z7   | 3  | 0   | 0  | 0  | 0  |

The zones are clearly related to the grid regions but not copies. For example, NR splits into the plains (Z5), the Himalaya (Z4) and
Rajasthan (into Z0).

### 15.7 Zone weather (`zone_daily_features`)

For each zone and day, every weather feature in `ZONE_FEATURES` (23 of them) is averaged over the zone's states, **weighted by
demand**. A 24th, `heat_exposure`, is the share of the zone's demand in heatwave states. Column names look like
`Z5_Uttar+Punjab__imd_tmax`. The same function with all states in one group "IN" gives the **national** weather columns (`IN__…`).

## 16. Stage 5: Building the model inputs (features)

**Code:** `features.py` (`build_model_table`). **Notebook:** 05.

### 16.1 The forecasting setting (what an operator really knows)

Every morning NLDC schedules generation for **tomorrow, day D**. It knows all demand up to **yesterday (D-1)** and has a **weather
forecast for D**. So:

* every load feature comes from **D-1 or earlier**;
* weather features describe **day D**. We use the observed weather of D as a stand-in for a perfect forecast. This is the standard
  "ex-post" set-up in load-forecasting research; in real use IMD's forecast would be plugged in and would add some error.

### 16.2 The target: "% above last week", not megawatts

```
base_D   = average of the daily peaks on D-1 … D-7            (known in the morning)
target_z = 100 × (peak_D / base_D − 1)                       (what the model predicts)
forecast_MW = base_D × (1 + predicted_z / 100)               (converted back)
```

*Example, 30-May-2024:* base = 231,949 MW, actual peak 246,779 MW, so target_z = **+6.39 %**.

**Why not predict MW directly?** A tree's leaf can only output an average of **training** values. Trained on 2019–2023 (maximum
237.4 GW), a megawatt model can **never** predict 246 GW. It would be short on precisely the record days. This is not
theoretical: in the evidence stage (Chapter 19, E5), a megawatt-trained model was short on **100 %** of 162 "beyond-training-level"
days (bias −18.5 GW), while our % target gives 1.4 % error. `target_z` is stable over the years (about −10 to +10 %).

### 16.3 All 309 features

**(a) Load history: 17 features** (`_load_history`; all "relative" ones are % differences from base_D)

| Feature                                                                                             | Meaning                                                           |
| --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `peak_lag1_rel`, `peak_lag2_rel`, `peak_lag3_rel`                                             | Peaks 1, 2 and 3 days ago vs base                                 |
| `peak_lag7_rel`, `peak_lag14_rel`                                                               | Same weekday 1 and 2 weeks ago                                    |
| `peak_lag364_rel`                                                                                 | Same weekday last year                                            |
| `peak_max7_rel`                                                                                   | Highest peak of the last 7 days                                   |
| `peak_std7_pct`                                                                                   | How jumpy the last week was                                       |
| `trend_1v7_pct`                                                                                   | Average of the last 3 days vs base (momentum)                     |
| `yoy_growth_pct`                                                                                  | How much the weekly base grew vs a year ago                       |
| `mean_mw_lag1_rel`, `min_mw_lag1_rel`, `solar_peak_mw_lag1_rel`, `evening_peak_mw_lag1_rel` | Yesterday's average, night minimum, daytime peak and evening peak |
| `load_factor_lag1`, `peak_hour_lag1`                                                            | Yesterday's shape                                                 |
| `weekday_effect_lag7`                                                                             | Last week's same weekday vs its own weekly average                |

**(b) Calendar: 12 features** (`_calendar`)

| Feature                                                    | Meaning / why                                                                                                                                        |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dow`, `is_sunday`, `is_saturday`                    | In training data, Sunday's peak averages **−3.7 %** vs the weekly average, while weekdays average +0.4 … +1.2 % (offices and factories shut) |
| `month`, `doy_sin`, `doy_cos`                        | Season; the day of year as a point on a circle (31-Dec next to 1-Jan)                                                                                |
| `is_holiday`, `days_to_holiday`, `days_from_holiday` | Gazetted holidays (from the `holidays` package, India list), distances capped at 7                                                                 |
| `is_major_festival`                                      | Holi, Diwali, Dussehra, Independence Day, Republic Day, Eid (both), Christmas, Gandhi Jayanti                                                        |
| `major_festival_yesterday`                               | The **day after** Holi rebounds about +11 % (25→26-Mar-2024). Yesterday's low peak drags the lag features down, so the model must know        |
| `diwali_window`                                          | Diwali −1 … +3 days (Govardhan Puja, Bhai Dooj)                                                                                                    |

*Why festivals were split from ordinary holidays:* the first validation run's biggest errors were Holi, Diwali and the days after them.
Minor holidays (e.g. Mahavir Jayanti) barely move national demand.

**(c) National heat: 6 features:** `imd_hw_exposure`, `heat_exposure`, `severe_exposure`, `heat_severity_idx`,
`warm_night_exposure`, `hot40_exposure` (Chapter 14.5).

**(d) Grid operations: 3 features**, from **yesterday's** NLDC report: `share_res_pct_lag1` (renewable share), `energy_shortage_mu_lag1`,
`solar_gen_mu_lag1`. Rising solar changes the daytime shape.

**(e) National weather: 23 features** (`IN__…`): demand-weighted national averages of `cdd24`, `cdd24_3d`, `ehf`, `era5_cdh24`,
`era5_cloud_mean`, `era5_dewpoint_mean`, `era5_heat_index_max`, `era5_night_tmin`, `era5_precip_sum`, `era5_rh_mean`,
`era5_solar_kwh_m2`, `era5_t_15h`, `era5_wind_mean`, `heat_accum_ewm`, `heat_day_run_length`, `heat_severity`,
`hot_day_run_length`, `hw_area_frac`, `imd_tmax`, `imd_tmax_departure`, `imd_tmin`, `imd_tmin_departure`, `tmax_change_1d`.

**(f) Zone weather: 192 features** = 8 zones × (the same 23 + the zone's `heat_exposure`).

**(g) Weekly weather changes: 56 features** (`_weather_deltas`, suffix `__d7`) = weather on D minus the average of D-1…D-7, for
`imd_tmax`, `era5_cdh24`, `era5_heat_index_max`, `era5_precip_sum`, `heat_severity`, `era5_night_tmin` (nationally and for each zone:
9 × 6 = 54), plus `heat_exposure` and `heat_severity_idx`.

*Why weekly changes?* The target is "peak vs last week", so the matching weather signal is "weather vs last week". Delhi at 44 °C after
a week at 44 °C is already in the base; 44 °C after a week at 38 °C is a surge.

**Total: 17 + 12 + 6 + 3 + 23 + 192 + 56 = 309.** (Notebook 05's quick tally counts the 3 grid-ops features with load history,
because their names contain "lag".)

### 16.4 Columns that are not features

`peak_mw`, `base_mw`, `target_z`, `is_critical_peak`, `target_imputed`, `split`, `train_weight`, `national_hw_regime`,
`covid_lockdown`, and (added in modelling) `loss_severity`, `loss_scarcity`. The table has **2,557 rows × 320 columns**.

### 16.5 Training weights (`train_weight`)

Set to **0** (ignored in training) for:

* **COVID lockdown** (22-Mar → 30-Jun-2020): demand fell about 25 % for non-weather reasons. Training on it would teach "hot April–May =
  low demand".
* **Imputed targets** (the 10 report-less days).
* The first days with no 7-day history.

The rows stay in the table so that "yesterday" and "last week" remain defined.

### 16.6 Splits and their contents

| Split              | Days  | Usable for training | Heat-regime days | Critical-peak days |
| ------------------ | ----- | ------------------- | ---------------- | ------------------ |
| train (2019–2023) | 1,826 | 1,720               | 136              | 357                |
| valid (2024)       | 366   | 366                 | 55               | 60                 |
| test (2025)        | 365   | 355 (10 imputed)    | 14               | 45                 |

### 16.7 Sanity checks in notebook 05

* **Heat explains the target.** In April–June training days, the correlation of `target_z` with the weekly change in national
  cooling degree-hours is **0.716**, with weekly Tmax change **0.686**, with weekly heat-exposure change **0.400**, and with weekly rain change **−0.401**.
* **Leakage audit.** No feature correlates suspiciously with the target. The highest is yesterday's peak (0.604), as expected, and the
  target itself is confirmed absent from the features.

## 17. Stage 6: Training all the models

**Code:** `models.py`, `experiment.py` (`run_full_experiment`), `losses.py`. **Notebook:** 06.

### 17.1 The models compared (a controlled experiment)

Every LightGBM model gets the **same 309 features, the same target, the same days, the same tree settings and the same weights**. Only
the **loss** differs, so any difference in results is caused by the loss.

| Name in files           | What it is                                                                                          | Why it is included                                                                  |
| ----------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `naive_lag7`          | Forecast = the peak on the same weekday last week (computed as `base × (1 + peak_lag7_rel/100)`) | A planner's spreadsheet method; every model must beat it                            |
| `lgbm_l2`             | LightGBM with squared error (symmetric)                                                             | **Main baseline:** what most studies do (gap G1)                              |
| `lgbm_quantile60`     | LightGBM with pinball loss, τ = 0.6                                                                | Textbook asymmetric method; similar overall accuracy to HALO, for a fair comparison |
| `lgbm_quantile80`     | Pinball loss, τ = 0.8                                                                              | The **cost-optimal** quantile for a 10 : 2.5 cost ratio (gap G4 baseline)     |
| `lgbm_prototype`      | The team's earlier piecewise loss (×21 / ×4 multipliers, Hessian 1)                               | Shows the improvement over our own first idea                                       |
| `lgbm_linex`          | HALO with `a_heat = a_scarcity = 0` (plain LINEX with a = 0.2)                                    | **Ablation:** is it the *conditioning* that helps, or just LINEX?           |
| **`lgbm_halo`** | **HALO, our proposal**                                                                        |                                                                                     |

### 17.2 Common preparation (`add_loss_side_info`, `make_split`, `_offset`, `_scale`)

1. `add_loss_side_info` adds `loss_severity` and `loss_scarcity` (Chapters 8.3–8.4). The severity reference (0.161) is taken from
   training days only.
2. `make_split` selects rows by date and packs the features, target, base, weights, severity, scarcity and heatwave flag into a `SplitData` object.
3. Each model is trained on `target_z − offset` (offset 0.092 %), with weights `train_weight × base/mean_base`.

### 17.3 Step 1: tuning HALO on 2024 (`tune_halo`)

* **Grid:** a0 ∈ {0.05, 0.1, 0.2, 0.3} × a_heat ∈ {0, 3, 5, 8, 12} × a_scarcity ∈ {0, 2, 4, 6} = **80 settings**. Each is trained on
  2019–2023 with early stopping on 2024.
* **Why such large values?** From Chapter 8.9, the margin is about a·σ²/2 with σ ≈ 2 %. A first coarse grid with a ≤ 1.5 barely
  changed anything (heatwave under-forecast rate only fell 60 % → 51 %).
* **Selection rule:**

  > Among settings whose **ordinary-day MAE is at most 105 % of the L2 model's** on 2024, choose the one with the **lowest grid cost
  > on heatwave-or-critical-peak days**.
  >

  This turns the prototype's two success criteria ("cut heatwave shortfalls" and "do not hurt normal days by more than 5 %") into a
  selection rule. 78 of the 80 settings passed the 105 % limit.
* **Selected: a0 = 0.2, a_heat = 12, a_scarcity = 6.** On 2024 it gave heat/critical cost ₹8.56 crore/day (L2: ₹11.44), ordinary MAE
  ratio 1.003, and heatwave-day under-forecast rate 38.2 % (L2: 60 %). *Note:* a_heat and a_scarcity landed on the **top edge** of the
  grid, so even stricter values might score slightly better on 2024. We did not widen further, to avoid tuning too hard on one year.

### 17.4 Step 2: validation-stage models (`fit_all`)

All six LightGBM models are trained on 2019–2023, with early stopping on 2024:

* L2 and the quantile models use LightGBM's own score for their loss;
* HALO and LINEX use `HALOObjective.feval`;
* the prototype has no matching score, so it uses L2's tree count.

Resulting tree counts: L2 700, Q0.6 3,999 (it never stopped improving before the 4,000 cap), Q0.8 915, prototype 700, LINEX 381,
HALO 373. Predictions are saved to `predictions_valid_2024.parquet`.

### 17.5 Step 3: refit on 2019–2024 and forecast 2025 once

Every model is retrained on **2019–2024** with the tree counts above (no early stopping, since 2025 must stay unseen), then forecasts
2025. *Why refit?* 2024 contains the record heatwave; a model that never saw it would be handicapped needlessly. The final models are saved
as `models/<name>.txt`, and predictions as `predictions_test_2025.parquet`.

### 17.6 Step 4: walk-forward backtest 2022–2025

For each year Y in 2022, 2023, 2024 and 2025: train on 2019…Y−1 and forecast year Y, using the same tree counts and the same HALO
settings. This gives **1,451 scored days** (381 heat-or-critical) instead of 2025's 355 (52 heat-or-critical), so the conclusions are
much more reliable. *Caveat:* HALO's settings were tuned on 2024, so the 2024 fold is not fully independent. That is why Chapter 19 also
reports the **out-of-sample folds 2022, 2023 and 2025** separately. Saved as `predictions_backtest_2022_2025.parquet`.

## 18. Stage 7: Measuring how good the models are

**Code:** `metrics.py`. **Notebooks:** 06 (validation), 07 (test and backtest).

### 18.1 Day groups ("regimes")

| Regime               | Definition in `metrics.py`                                                              |
| -------------------- | ----------------------------------------------------------------------------------------- |
| all                  | Every scored day                                                                          |
| ordinary             | `national_hw_regime == 0` (not a heatwave regime day; may still be a critical-peak day) |
| heatwave             | `national_hw_regime == 1`                                                               |
| critical_peak        | `is_critical_peak == 1`                                                                 |
| heatwave_or_critical | Either of the above                                                                       |

*(In the evidence stage, Chapter 19, "ordinary" is stricter: neither heatwave nor critical-peak. The tables say which is used.)*

Days with imputed targets are never scored (`_eval_frame`).

### 18.2 Every metric, in plain words

| Metric                 | Formula in words                                                                                                                                                                                                                                                                             | Good when                               |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| `MAE_MW`             | Average size of the error in MW                                                                                                                                                                                                                                                              | Low                                     |
| `RMSE_MW`            | Square root of the average squared error (punishes big misses)                                                                                                                                                                                                                               | Low                                     |
| `MAPE_pct`           | Average error as % of the actual peak                                                                                                                                                                                                                                                        | Low                                     |
| `R2`                 | 1 − (squared errors ÷ squared distance of actuals from their mean). Near 1 means the ups and downs are explained. *Within a small group of similar days (e.g. critical peaks) R² can be low or negative even when errors are small, because those days differ little from one another.* | Close to 1                              |
| `bias_MW`            | Average of (forecast − actual). Positive = tends to over-forecast                                                                                                                                                                                                                           | Near 0, or slightly positive for safety |
| `under_rate_pct`     | % of days where forecast < actual (short)                                                                                                                                                                                                                                                    | Low                                     |
| `mean_shortfall_MW`  | Average MW short, over the short days only                                                                                                                                                                                                                                                   | Low                                     |
| `p95_shortfall_MW`   | 95th percentile of shortfall over all days (0 on non-short days): "a bad day's shortfall"                                                                                                                                                                                                    | Low                                     |
| `max_shortfall_MW`   | Worst shortfall                                                                                                                                                                                                                                                                              | Low                                     |
| `reserve_breach_pct` | % of days short by **more than 3 %** of the peak. India keeps roughly 3–5 % of demand as reserves; beyond 3 % short, reserves run out                                                                                                                                                 | Low                                     |
| `cost_crore_per_day` | Average stylised cost: shortfall MW × 4 h × ₹10/kWh + surplus MW × 4 h × ₹2.5/kWh, in ₹ crore (1 crore = 10 million). *Example:* 2,000 MW short for 4 h = 8,000 MWh = 8 million kWh × ₹10 = **₹8 crore**                                                                   | Low                                     |

### 18.3 Results: 2024 validation year (models trained on 2019–2023)

| Model          | MAPE %         | R²            | MAE (MW)        | Heatwave-day under-forecast rate | Critical-day under-forecast rate | Cost, all (₹ cr/day) |
| -------------- | -------------- | -------------- | --------------- | -------------------------------- | -------------------------------- | --------------------- |
| Naive          | 3.63           | 0.23           | 7,544           | 65.5 %                           | 81.7 %                           | 18.87                 |
| L2             | **1.45** | **0.84** | **3,002** | 60.0 %                           | 75.0 %                           | 7.53                  |
| Quantile 0.6   | 1.49           | 0.83           | 3,049           | 52.7 %                           | 73.3 %                           | 6.63                  |
| Quantile 0.8   | 1.85           | 0.75           | 3,759           | **30.9 %**                 | **51.7 %**                 | **5.82**        |
| Prototype      | 1.47           | 0.84           | 3,015           | 47.3 %                           | 66.7 %                           | 6.66                  |
| LINEX          | 1.46           | 0.83           | 3,000           | 56.4 %                           | 73.3 %                           | 7.23                  |
| **HALO** | 1.48           | 0.83           | 3,037           | 38.2 %                           | 60.0 %                           | 6.31                  |

### 18.4 Results: 2025 test year (models refit on 2019–2024; 355 measured days)

| Model          | MAPE %         | R²            | MAE (MW)        | Ordinary-day MAE | Under-forecast rate, heat/critical | Reserve breach, heat/critical | Max shortfall (MW) | Cost (₹ cr/day) |
| -------------- | -------------- | -------------- | --------------- | ---------------- | ---------------------------------- | ----------------------------- | ------------------ | ---------------- |
| Naive          | 3.57           | 0.34           | 7,593           | 7,207            | 80.8 %                             | 46.2 %                        | 31,182             | 19.84            |
| L2             | **1.16** | **0.93** | **2,462** | **2,444**  | 71.2 %                             | 3.8 %                         | 12,361             | 6.38             |
| Quantile 0.6   | 1.20           | 0.91           | 2,518           | 2,501            | 67.3 %                             | 3.8 %                         | **7,559**    | 5.21             |
| Quantile 0.8   | 1.50           | 0.86           | 3,134           | 3,118            | **50.0 %**                   | **1.9 %**               | 7,961              | **4.62**   |
| Prototype      | 1.22           | 0.92           | 2,573           | 2,547            | 69.2 %                             | 3.8 %                         | 12,138             | 6.12             |
| LINEX          | 1.21           | 0.91           | 2,536           | 2,515            | 67.3 %                             | 3.8 %                         | 11,245             | 5.99             |
| **HALO** | 1.29           | 0.89           | 2,709           | 2,696            | 57.7 %                             | **1.9 %**               | 8,467              | 5.40             |

### 18.5 Results: walk-forward backtest 2022–2025 (1,451 days; 381 heat-or-critical)

| Model          | MAPE %         | Ordinary MAE    | Under-forecast rate, heat/critical | Reserve breach, heat/critical | Cost, heat/critical (₹ cr) | Cost, all (₹ cr) |
| -------------- | -------------- | --------------- | ---------------------------------- | ----------------------------- | --------------------------- | ----------------- |
| Naive          | 3.81           | 7,435           | 73.2 %                             | 33.1 %                        | 23.81                       | 19.45             |
| L2             | **1.32** | **2,564** | 66.1 %                             | 4.7 %                         | 8.29                        | 6.56              |
| Quantile 0.6   | 1.38           | 2,675           | 60.6 %                             | 2.9 %                         | 7.23                        | 5.81              |
| Quantile 0.8   | 1.71           | 3,307           | **44.4 %**                   | **0.8 %**               | **5.32**              | **5.06**    |
| Prototype      | 1.36           | 2,634           | 62.7 %                             | 3.4 %                         | 7.60                        | 6.14              |
| LINEX          | 1.35           | 2,605           | 65.6 %                             | 3.9 %                         | 8.07                        | 6.36              |
| **HALO** | 1.40           | 2,697           | 52.5 %                             | 2.1 %                         | 6.44                        | 5.68              |

MAPE per test year (the backtest's 4 folds):

| Model        | 2022 | 2023 | 2024 | 2025 |
| ------------ | ---- | ---- | ---- | ---- |
| L2           | 1.22 | 1.46 | 1.45 | 1.16 |
| Quantile 0.6 | 1.24 | 1.60 | 1.49 | 1.20 |
| Quantile 0.8 | 1.49 | 2.00 | 1.85 | 1.50 |
| Prototype    | 1.23 | 1.52 | 1.47 | 1.22 |
| LINEX        | 1.23 | 1.49 | 1.46 | 1.21 |
| HALO         | 1.28 | 1.56 | 1.48 | 1.29 |
| Naive        | 3.08 | 4.94 | 3.63 | 3.57 |

### 18.6 How to read these results (plain words)

1. **All LightGBM models are accurate:** 1.2–1.7 % average error versus 3.6–3.8 % for the naive method.
2. **L2 is the most accurate on average, but short on two thirds of the dangerous days.** That is the problem the project is about.
3. **Quantile 0.8 is the "safest" and cheapest on the stylised cost**, but only because it adds about **+2.2 GW to every day**, cool
   Sundays included. Its average error is 0.3–0.4 points worse everywhere, and an operator would be paying for unneeded capacity all year.
4. **HALO sits in between, and it is efficient.** Against L2 it halves reserve breaches on dangerous days (4.7 % → 2.1 %) and cuts the
   worst shortfall from 12.4 GW to 8.5 GW (2025), at the price of 5–10 % worse ordinary-day error. At similar ordinary-day accuracy to
   quantile 0.6 (backtest 2,697 vs 2,675 MW), it is clearly safer on dangerous days. Chapter 19 proves this is statistically real.
5. **LINEX without conditioning barely differs from L2**, so the gain comes from the conditioning, not from the LINEX shape alone.

### 18.7 Case study: the record week (backtest fold 2024; errors in GW, forecast − actual, negative = short)

| Date        | Actual (GW) | Heat exposure | L2               | Quantile 0.8 | HALO             |
| ----------- | ----------- | ------------- | ---------------- | ------------ | ---------------- |
| 26-May-2024 | 218.79      | 0.65          | +4.60            | +8.94        | +8.23            |
| 27-May      | 229.08      | 0.56          | −0.22           | +3.85        | +5.12            |
| 28-May      | 234.64      | 0.54          | −2.89           | +0.62        | +1.37            |
| 29-May      | 241.90      | 0.59          | **−5.16** | −1.90       | **−2.72** |
| 30-May      | 246.78      | 0.70          | −2.23           | −0.95       | −1.51           |
| 31-May      | 244.94      | 0.81          | +1.88            | +2.75        | +1.35            |
| 1-Jun       | 241.52      | 0.85          | −2.55           | +0.54        | +0.97            |

On 29-May, the day before the all-time record, L2 was 5.2 GW short and HALO 2.7 GW short. HALO over-forecast the Sunday (26-May): that is the
price of leaning up during a heatwave.

### 18.8 Explaining the models (SHAP, notebook 07)

LightGBM's built-in `pred_contrib=True` splits every 2025 prediction into "how much each feature pushed it up or down" (exact TreeSHAP).
Units: % of the weekly base.

**Most influential features in 2025** (average absolute push):

* **L2:** yesterday's peak (0.90), day of week (0.81), yesterday's evening peak (0.40), Sunday flag (0.32), weekly change in
  national cooling degree-hours (0.31), yesterday's daytime peak (0.24), weekly change in national Tmax (0.18), major festival (0.15).
* **HALO:** day of week (0.97), yesterday's peak (0.84), **weekly change in national Tmax (0.25)**, yesterday's evening peak (0.22),
  yesterday's daytime peak (0.20), Sunday (0.19), **weekly change in cooling degree-hours of the north-west zone Z5 (0.18)**, 3-day
  momentum (0.17).

**Heat features on heatwave days vs ordinary days** (sum of all heat-related features' pushes):

|                       | L2       | HALO              |
| --------------------- | -------- | ----------------- |
| Heatwave days of 2025 | −0.10 % | **+0.26 %** |
| Ordinary days of 2025 | −0.06 % | −0.05 %          |

On ordinary days both models use heat the same way. On heatwave days **HALO's heat features push the forecast up**, while L2's
push it slightly down. The extra safety margin enters through the heat features, exactly as designed.

## 19. Stage 8: Proving the novelty (evidence tests)

**Code:** `evidence.py`. **Notebook:** 08. **Write-up:** `reports/NOVELTY_CLAIMS.md`.

A claim needs three things:

1. the effect is **real, not luck** (significance tests);
2. it comes from **the part we say is new** (ablations);
3. the comparison is **fair**: at **equal everyday accuracy**, because any model that simply biases upward will trivially be short less often.

All tests use the backtest predictions (2022–2025). "OOS folds" = 2022, 2023, 2025 (not used to tune HALO).

### 19.1 The statistical tools, in plain words

* **Moving-block bootstrap.** Make 2,000 "alternative histories" by re-drawing the test days in **blocks of 7 consecutive days**
  (errors are correlated within a week: a model short on Monday of a heatwave is likely short on Tuesday too). Recompute the difference
  between two models in each history. The middle 95 % of results is the **confidence interval**; the share of histories where the sign
  flips gives the **p-value**. It is "paired": both models are always scored on the same days.
* **Diebold–Mariano test.** The standard forecasting test for "do two forecasts have the same average cost?", corrected for
  day-to-day correlation (7-day window) and for sample size.
* **McNemar test.** Among dangerous days where exactly **one** of two models was short, is it one model much more often than the other?

The extra statistics used here:

* `expected_shortfall_hc_MW` = average MW short per heat/critical day, with 0 on days that were not short. Unlike "mean shortfall when
  short", it cannot improve just by being slightly short less often.
* `expected_shortfall_heatwave_MW` = the same, on heatwave days only.
* `MAE_ordinary_MW` (here) = MAE on days that are neither heatwave nor critical.

### 19.2 E1: Is the improvement real? (`exp_significance`)

**HALO vs L2, OOS folds** (HALO − L2; negative = HALO lower):

| Statistic                          | L2       | HALO     | Difference         | 95 % CI          | p       |
| ---------------------------------- | -------- | -------- | ------------------ | ---------------- | ------- |
| Under-forecast rate, heat/critical | 65.7 %   | 52.5 %   | −13.2 pts         | −17.4 … −8.8  | < 0.001 |
| Reserve breach, heat/critical      | 3.3 %    | 1.3 %    | −2.0 pts          | −4.0 … −0.3   | 0.043   |
| Expected shortfall, heat/critical  | 1,704 MW | 1,169 MW | **−536 MW** | −665 … −402   | < 0.001 |
| Expected shortfall, heatwave days  | 1,536 MW | 905 MW   | −630 MW           | −884 … −375   | < 0.001 |
| Cost, heat/critical (₹ cr/day)    | 7.47     | 5.90     | −1.57             | −2.13 … −1.00 | < 0.001 |
| Cost, all days                     | 6.23     | 5.47     | −0.76             | −1.04 … −0.48 | < 0.001 |
| Ordinary-day MAE (the price)       | 2,568 MW | 2,811 MW | **+243 MW**  | +121 … +364     | < 0.001 |
| MAPE, all days                     | 1.28 %   | 1.38 %   | +0.10              | +0.05 … +0.15   | < 0.001 |

Diebold–Mariano on daily cost: statistic 5.20, p < 0.001. McNemar: on 44 dangerous days only one model was short; **42 times it was
L2, 2 times HALO** (p < 0.001). On 2025 alone the shortfall and cost gains are also significant (p ≤ 0.012). The reserve-breach
difference is not (only 52 dangerous days).

**HALO vs Quantile 0.6, backtest:** ordinary-day accuracy is **statistically indistinguishable** (+12 MW, p = 0.78; MAPE p = 0.28), yet
HALO is significantly safer: under-forecast rate −8.1 points (p < 0.001), expected shortfall −315 MW (p < 0.001), cost on
dangerous days −₹0.78 crore (p < 0.001), McNemar 42 vs 11 (p < 0.001). On 2025 alone only the shortfall gain is significant (p = 0.032),
and there HALO is less accurate than Q0.6 on ordinary days.

### 19.3 E2: The core test, conditioned vs uniform at equal accuracy (`exp_frontier`, `exp_frontier_bootstrap`)

We trace the **accuracy–safety curve** of each "uniform" method by turning its one knob:

* uniform LINEX with a = 0.2, 0.4, 0.6, 0.8, 1.0, 1.5;
* pinball with τ = 0.55, 0.6, 0.65, 0.7, 0.75.

Each point is plotted as (ordinary-day MAE, risk on dangerous days). Then we place HALO variants on the same chart (figure
`08_frontier.png`).

| Model                                     | Family      | Ordinary MAE                                  | Expected shortfall, heat/critical             | Cost, heat/critical                     |
| ----------------------------------------- | ----------- | --------------------------------------------- | --------------------------------------------- | --------------------------------------- |
| L2                                        | symmetric   | 2,623                                         | 1,873                                         | 8.29                                    |
| LINEX a=0.2 / 0.4 / 0.6 / 0.8 / 1.0 / 1.5 | uniform     | 2,676 / 2,780 / 2,866 / 2,969 / 3,047 / 3,237 | 1,801 / 1,666 / 1,531 / 1,428 / 1,325 / 1,190 | 8.07 / 7.59 / 7.14 / 6.83 / 6.49 / 6.12 |
| Pinball τ=0.55 / 0.6 / 0.65 / 0.7 / 0.75 | uniform     | 2,803 / 2,889 / 2,993 / 3,163 / 3,330         | 1,668 / 1,551 / 1,401 / 1,192 / 1,068         | 7.58 / 7.18 / 6.68 / 6.06 / 5.72        |
| **HALO (selected 0.2/12/6)**        | conditioned | **2,816**                               | **1,254**                               | **6.45**                          |
| HALO 0.05/12/6                            | conditioned | 2,774                                         | 1,305                                         | 6.61                                    |
| HALO 0.1/8/4                              | conditioned | 2,780                                         | 1,355                                         | 6.75                                    |
| HALO 0.3/12/6                             | conditioned | 2,874                                         | 1,242                                         | 6.42                                    |
| HALO heat-only 0.2/12/0                   | ablation    | 2,778                                         | 1,467                                         | 7.11                                    |
| HALO scarcity-only 0.2/0/6                | ablation    | 2,723                                         | 1,430                                         | 6.97                                    |

**How to read it:** look at the selected HALO, with ordinary MAE 2,816. A uniform LINEX with the same accuracy (between a = 0.4 and
0.6) has an expected shortfall of about 1,610 MW; HALO has 1,254. **Every HALO variant lies below both uniform curves.**

**Gain at equal accuracy (HALO minus the uniform curve, with 95 % CI from 1,000 bootstrap histories):**

| Compared with   | Days                | Expected shortfall                           | Cost (₹ cr/day)                   | Under-forecast rate        |
| --------------- | ------------------- | -------------------------------------------- | ---------------------------------- | -------------------------- |
| uniform LINEX   | all folds           | −356 MW (−482…−224)                      | −0.96 (−1.41…−0.48)            | −9.9 pts (−14.0…−5.8)  |
| uniform pinball | all folds           | −396 MW (−468…−216)                      | −1.07 (−1.39…−0.44)            | −10.2 pts (−13.8…−3.8) |
| uniform LINEX   | **OOS folds** | **−227 MW** (−372…−102), p < 0.001 | −0.51 (−1.03…−0.04), p = 0.034 | −9.6 pts, p < 0.001       |
| uniform pinball | **OOS folds** | **−296 MW** (−396…−116), p < 0.001 | −0.75 (−1.14…−0.11), p = 0.011 | −8.5 pts, p = 0.005       |

**Which ingredient matters?**

* **Scarcity-only** beats uniform LINEX significantly on all folds and on OOS folds (e.g. −267 MW, p < 0.001).
* **Heat-only** beats it on all folds (−202 MW, p < 0.001) but **not on OOS folds** (−106 MW, p = 0.12). On its own, the heat term is not robust outside 2024.
* **Adding heat on top of scarcity** (`exp_component_contrast`, same a0 and scarcity strength) significantly lowers heatwave-day shortfall:
  OOS **1,205 → 905 MW (−300 MW, p < 0.001)**, and heat/critical shortfall −145 MW (p < 0.001). It also raises ordinary MAE by 116 MW.

**Conclusion:** both conditioning signals are needed. Scarcity handles record-demand days; heat severity handles heatwave days.

### 19.4 E3: Feature ablations (`exp_feature_ablation`, `ablation_bootstrap`)

Remove a group of features, retrain L2 and HALO on the same backtest, and compare with the full feature set.

| Removed group                                                                              | Features left | Effect on L2     | Effect on HALO                                                             |
| ------------------------------------------------------------------------------------------ | ------------- | ---------------- | -------------------------------------------------------------------------- |
| Explicit heatwave-regime features (exposure, severity, heatwave area, EHF)                 | 257           | none (p ≥ 0.57) | none (p ≥ 0.52)                                                           |
| **Heat accumulation** (run lengths, smoothed heat load, 3-day CDD, EHF, warm nights) | 263           | none (p ≥ 0.53) | **worse: shortfall +55 MW (p = 0.006), cost +₹0.21 cr (p = 0.009)** |
| Humidity (heat index, apparent temperature, dew point, RH)                                 | 273           | none             | none (p ≥ 0.14)                                                           |

**Meaning:**

* Heat-persistence features matter **when the loss is heat-conditioned**: HALO uses them to decide where to add margin.
* The heatwave regime is valuable **as a signal inside the loss** (E2), **not** as an extra input column: trees recover heat from raw temperatures.
* Humidity features add nothing at national scale.

### 19.5 E4: Does the zone clustering help the national forecast? (`exp_spatial_ablation`)

| Weather representation  | Features | L2 MAPE | HALO MAPE | HALO cost, heat/critical |
| ----------------------- | -------- | ------- | --------- | ------------------------ |
| Our 8 zones             | 309      | 1.324   | 1.404     | 6.45                     |
| 5 official grid regions | 219      | 1.326   | 1.427     | 6.39                     |
| National average only   | 69       | 1.318   | 1.410     | **6.11**           |

**Honest negative result:** for the national target, zones are **not** better than a single demand-weighted national average. The
useful ingredient is the **demand-weighted aggregation of state weather**, not the regional split. The clustering remains a useful,
interpretable map of India's heat-and-demand geography and is ready for future regional forecasts, but we do **not** claim it improves
this forecast. The zone-based model stays the main model because that decision was made before this test; switching after seeing
results would be unfair.

### 19.6 E5: Demand beyond anything seen in training (`exp_level_shift`)

On the **162 backtest days whose peak was at or above 98 % of the highest peak in that fold's training years:**

| Model                  | Bias                  | Short on                | MAPE   | Worst shortfall |
| ---------------------- | --------------------- | ----------------------- | ------ | --------------- |
| L2 trained on raw MW   | **−18,468 MW** | **100 %** of days | 8.65 % | 38,888 MW       |
| L2 with our % target   | −1,260 MW            | 71.6 %                  | 1.40 % | 10,473 MW       |
| HALO with our % target | **−51 MW**     | 58.6 %                  | 1.35 % | 8,479 MW        |

This is evidence that the target transform handles **growth-driven** distribution shift. Climate-driven future extremes were not tested.

### 19.7 E6: Standard metrics hide peak risk (`exp_metric_disagreement`)

Ranking the six LightGBM models by MAPE gives: L2 1st, LINEX 2nd, prototype 3rd, Q0.6 4th, HALO 5th, Q0.8 6th. Ranking them by **any**
peak-risk metric (under-forecast rate, reserve breach, expected shortfall, cost) gives **exactly the reverse order**: Kendall τ = −1.0, a perfect
reversal. A study reporting only MAPE/RMSE would pick the model that is **worst** on heat and record days.

### 19.8 E7: Lightweight (`exp_lightweight`)

|      | Training (2,187 days × 309 features) | Forecast a full year | Model file       |
| ---- | ------------------------------------- | -------------------- | ---------------- |
| L2   | 2.4 s                                 | 9.9 ms               | 1,074 kB         |
| HALO | **1.6 s**                       | **6.4 ms**     | **650 kB** |

(HALO uses fewer trees: 373 vs 700.)

### 19.9 The gap-by-gap verdict (`evidence_gap_verdicts.csv`, computed automatically)

| Gap (from the Research Objective Mapping)           | Verdict              | Evidence                                                                                          |
| --------------------------------------------------- | -------------------- | ------------------------------------------------------------------------------------------------- |
| G1 symmetric losses ignore costlier under-forecasts | ✅ Supported         | −536 MW shortfall vs L2 (OOS, p < 0.001); accuracy cost stated                                   |
| G2 rare heatwave regimes poorly forecast            | ✅ Supported         | under-forecast rate −13.2 pts (OOS, p < 0.001)                                                   |
| G3 heatwave as an explicit conditioning regime      | 🟡 Partially         | via the loss: −300 MW heatwave-day shortfall beyond scarcity (p < 0.001); as features: no effect |
| G4 custom asymmetric objective inside boosting      | ✅ Supported         | −296 MW vs pinball at equal accuracy (OOS, p < 0.001)                                            |
| G5 asymmetry adapted to severity / grid state       | ✅ Supported         | −227 MW vs constant LINEX at equal accuracy (OOS, p < 0.001)                                     |
| G6 MAPE/RMSE hide peak under-forecasting            | ✅ Supported         | Kendall τ = −1.00                                                                               |
| G7 heat accumulation as features                    | 🟡 Partially         | +55 MW shortfall when removed from HALO (p = 0.006); none for L2; humidity none                   |
| G8 Indian grid                                      | ✅ Supported (scope) | All-India 2019–2025 with IMD rules                                                               |
| G9 lightweight and interpretable                    | 🟡 Partially         | 1.6 s, 650 kB, SHAP; no user study                                                                |
| G10 point vs probabilistic                          | 🟡 Partially         | pinball compared; conformal not tested                                                            |
| G11 distribution shift                              | 🟡 Partially         | growth shift handled; climate scenarios not tested                                                |
| (design) zone clustering improves the forecast      | ❌ Not supported     | national average does as well                                                                     |

---

# Part E: Novelty

## 20. What is new, and how we achieved it

### 20.1 What already existed (so we never over-claim)

* Losses that punish under-forecasting more than over-forecasting have been used in load forecasting before (with neural networks, and in
  gradient boosting as fixed piecewise multipliers).
* LINEX is a known loss (since 1975) used in other fields.
* Quantile (pinball) losses are the standard asymmetric method.
* Machine learning for Indian peak demand with weather data exists (e.g. Delhi studies).

So we do **not** claim "the first asymmetric loss" or "the first LINEX".

### 20.2 What is new, in one sentence

> A LINEX training loss inside gradient boosting whose strictness **changes every day** according to an **official, demand-weighted
> heatwave severity** (IMD rule + Excess Heat Factor) and **grid scarcity**, shown on the Indian national grid to give **more safety
> than a fixed strictness at the same everyday accuracy**.

### 20.3 The idea in plain words, with an analogy

Think of a shopkeeper ordering ice for tomorrow.

* **Squared error:** always order the average amount. On a scorching day you run out half the time.
* **Pinball / quantile:** always order 20 % extra. You rarely run out, but you waste ice every cool day of the year.
* **HALO:** order normally on cool days. On hot days, and on days when the city is already at record demand, order extra. The extra grows
  with **how hot** it is and **how uncertain** tomorrow is. You waste less on normal days and run out less on dangerous days.

### 20.4 How the novelty is achieved, step by step

1. **An honest definition of "dangerous day"** (Chapter 14): IMD's official rule plus EHF for humid heat, weighted by where the electricity
   demand actually is.
2. **A continuous severity number** (0–1) for every day (Chapter 8.3), not a yes/no switch.
3. **A scarcity number** (0–1): how close demand is to its record (Chapter 8.4).
4. **A per-day strictness:** a = 0.2 + 12 × severity + 6 × scarcity (Chapter 8.2).
5. **A loss shape that matches grid reality** (LINEX): normal for small errors, exponential for big shortfalls, gentle for surpluses (Chapter 8.1).
6. **Made safe for LightGBM:** gradient and Hessian formulas, Huber cap, Hessian floor, MW scaling, centring (Chapters 8.5–8.6), all tested (8.11).
7. **A side-effect that works for us:** LINEX's best forecast adds a margin of a·σ²/2, so risky and uncertain days get the most (Chapter 8.9).
8. **A fair proof:** comparison at equal ordinary-day accuracy against uniform LINEX and pinball, with confidence intervals on years not
   used for tuning (Chapter 19.3).

### 20.5 What the evidence lets you claim (and not claim)

**Claim:**

* conditioned asymmetry beats fixed asymmetry at equal accuracy (significant);
* HALO significantly cuts shortfalls on heat and record days versus a normal model;
* MAPE hides peak risk;
* heat-persistence features help a heat-conditioned loss;
* the target transform handles demand growth.

**Do not claim:**

* that HALO is more accurate on ordinary days (it is not);
* that clustering improves the forecast;
* that heatwave indicators help as input features;
* that humidity features help;
* that the heat term alone is robust out of sample;
* a comparison with conformal methods or climate-scenario robustness.

Ready-to-paste abstract wording is in `reports/NOVELTY_CLAIMS.md`, section 5.

---

# Part F: Every output explained

## 21. Every output file

### 21.1 `data/interim/`: cleaned, per-source tables

| File                                       | Rows × columns                   | Columns and meaning                                                                                                                                                                                                                   | Made by                                                                |
| ------------------------------------------ | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `load_hourly_national_2019_2025.parquet` | 61,368 × 4 (index: `datetime`) | `national_mw` cleaned, calibrated All-India demand · `source` (`kaggle_A` / `mendeley_B` / `scada_C` / `missing`) · `was_glitch` (True if the raw value was a removed spike) · `was_filled` (True if interpolated) | `build_national_hourly`                                              |
| `load_hourly_regional_2019_2024.parquet` | 46,728 × 5                       | `NR`, `WR`, `SR`, `ER`, `NER` hourly MW (glitch-cleaned)                                                                                                                                                                    | `build_regional_hourly`                                              |
| `state_daily_demand.parquet`             | 90,994 × 4                       | `date`, `state` (IMD name), `max_demand_mw` (state's maximum demand met that day), `energy_met_mu`                                                                                                                            | `load_state_daily_demand`                                            |
| `state_daily_weather_1991_2025.parquet`  | 421,872 × 45                     | One row per state per day: IMD temperatures, normals, departures, heatwave fractions and flags, EHF, persistence features, heat severity, and (from Dec-2018) all ERA5 daily features (Chapter 14)                                    | `imd_state_daily` + `add_heatwave_features` + `era5_state_daily` |
| `era5_hourly_states_2018_2025.parquet`   | 2,049,696 × 13                   | Raw hourly ERA5 for all 33 states stacked: `datetime`, `state`, `grid_lat`, `grid_lon`, `elevation_m` + the 8 weather variables                                                                                             | `era5_hourly_long`                                                   |

### 21.2 `data/processed/`: model-ready tables and predictions

| File                                           | Shape                     | Content                                                                                                                                                                                                                                                                                     |
| ---------------------------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `daily_peak_national.parquet`                | 2,557 × 11               | `peak_mw` (target), `peak_hour`, `min_mw`, `mean_mw`, `n_valid_hours`, `solar_peak_mw`, `evening_peak_mw`, `target_imputed`, `load_factor`, `critical_peak_threshold_mw`, `is_critical_peak`                                                                          |
| `state_demand_weights.csv`                   | 33 rows                   | `state`, `demand_weight` (shares summing to 1)                                                                                                                                                                                                                                          |
| `state_signatures_and_zones.csv`             | 33 rows                   | The 19 signature features + `grid_region` + `zone_id` (0–7) + `zone` (name)                                                                                                                                                                                                          |
| `national_heat_exposure.parquet`             | 2,557 × 7 (2019 → 2025) | `imd_hw_exposure`, `heat_exposure`, `severe_exposure`, `heat_severity_idx`, `warm_night_exposure`, `hot40_exposure`, `national_hw_regime`                                                                                                                                     |
| `model_table.parquet`                        | 2,557 × 318              | Everything the models read: the 309 features + `peak_mw`, `base_mw`, `target_z`, `is_critical_peak`, `target_imputed`, `split`, `train_weight`, `national_hw_regime`, `covid_lockdown`. (`loss_severity` and `loss_scarcity` are added in memory, giving 320 columns) |
| `predictions_valid_2024.parquet`             | 366 rows                  | Models trained on 2019–2023, predicting 2024                                                                                                                                                                                                                                               |
| `predictions_test_2025.parquet`              | 365 rows                  | Final models (trained 2019–2024), predicting 2025                                                                                                                                                                                                                                          |
| `predictions_backtest_2022_2025.parquet`     | 1,461 rows                | Walk-forward predictions; extra column `fold` (test year)                                                                                                                                                                                                                                 |
| `evidence_frontier_backtest.parquet`         | 1,451 rows                | Backtest predictions of the 18 models in the frontier test (`pred_L2`, `pred_LINEX a=…`, `pred_Quantile …`, `pred_HALO …`) + `fold`                                                                                                                                            |
| `evidence_feature_ablation_backtest.parquet` | 1,451 rows                | Columns like `pred_HALO\|no_accumulation_features`                                                                                                                                                                                                                                         |

**Prediction file columns:**

* `peak_mw` (actual), `base_mw`, `national_hw_regime`, `is_critical_peak`, `heat_exposure`, `heat_severity_idx`,
  `loss_severity`, `loss_scarcity`, `train_weight`, `target_imputed`;
* one `pred_<model>` column per model, in **MW**: `pred_naive_lag7`, `pred_lgbm_l2`, `pred_lgbm_quantile60`,
  `pred_lgbm_quantile80`, `pred_lgbm_prototype`, `pred_lgbm_linex`, `pred_lgbm_halo`.

### 21.3 `models/`: trained models

`lgbm_l2.txt`, `lgbm_quantile60.txt`, `lgbm_quantile80.txt`, `lgbm_prototype.txt`, `lgbm_linex.txt`, `lgbm_halo.txt`: LightGBM's own
text format (every tree written out). All were trained on 2019–2024.

To use one:

```python
import lightgbm as lgb
b = lgb.Booster(model_file="models/lgbm_halo.txt")
z_hat = b.predict(X) + 0.0799   # add back the training offset of 2019-2024
mw = base_mw * (1 + z_hat / 100)
```

**Note:** the offset for the saved final models is the weighted mean target of 2019–2024 = **0.0799 %** (for the 2019–2023 validation-stage
models it was 0.092 %). `experiment.run_full_experiment()` returns ready-made `Trained` objects whose `predict_mw` handles this automatically.
`X` must contain the 309 feature columns, in the order given by `features.feature_columns`.
Size differences between files reflect tree counts; the quantile-0.6 model is largest (3,999 trees).

### 21.4 `reports/metrics/`: every result table

| File                                                                                      | What it contains                                                                                                                                                                                                                                                      |
| ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `load_calibration.json`                                                                 | `factor_mendeley_to_kaggle` 0.99232, `overlap_hours_A_B` 2903, `factor_scada_to_kaggle` 0.99122, `overlap_hours_B_C` 5392, `psp_to_hourly_peak_ratio` 0.98803                                                                                               |
| `clustering_model_selection.csv`                                                        | For each algorithm × K: `silhouette`, `calinski_harabasz`, `davies_bouldin`, `ari_vs_grid`, `nmi_vs_grid`, `min_cluster_size`                                                                                                                            |
| `halo_tuning_grid_valid2024.csv`                                                        | 80 rows, one per HALO setting: `a0`, `a_heat`, `a_scarcity`, `best_iter`, `cost_all`, `cost_hw_or_critical`, `MAPE_all`, `MAE_ordinary`, `MAE_hw`, `under_rate_hw`, `under_rate_critical`, `reserve_breach_all`, `ordinary_mae_ratio_vs_l2` |
| `experiment_summary.json`                                                               | Chosen HALO settings, tree counts per model, severity reference (0.161), number of features (309) and rows (train 1,720 / valid 366 / test 365)                                                                                                                       |
| `metrics_valid_2024.csv`, `metrics_test_2025.csv`, `metrics_backtest_2022_2025.csv` | Long tables: one row per (model, regime) with all metrics of Chapter 18.2                                                                                                                                                                                             |
| `metrics_test_2025_named.csv`, `metrics_backtest_named.csv`                           | Same, with readable model names (written by notebook 07)                                                                                                                                                                                                              |
| `run1_allday_cost_rule/`                                                                | The **first** run's outputs, before the selection rule was revised (kept for honesty; Chapter 26)                                                                                                                                                               |
| `evidence_significance.csv`                                                             | E1: for each evaluation set × baseline × statistic: values, `diff`, `ci_low`, `ci_high`, `p_value`, `significant_5pct`, Diebold–Mariano and McNemar results                                                                                              |
| `evidence_frontier.csv`, `evidence_frontier_oos_folds.csv`                            | E2: risk and accuracy statistics of all 18 frontier models (all folds / OOS folds)                                                                                                                                                                                    |
| `evidence_frontier_bootstrap.csv`                                                       | E2: equal-accuracy gains with CIs for selected, heat-only and scarcity-only HALO                                                                                                                                                                                      |
| `evidence_component_contrast.csv`                                                       | E2: full HALO vs scarcity-only HALO                                                                                                                                                                                                                                   |
| `evidence_feature_ablation.csv`, `evidence_feature_ablation_bootstrap.csv`            | E3                                                                                                                                                                                                                                                                    |
| `evidence_spatial_ablation.csv`                                                         | E4                                                                                                                                                                                                                                                                    |
| `evidence_level_shift.csv`                                                              | E5                                                                                                                                                                                                                                                                    |
| `evidence_metric_disagreement.csv`                                                      | E6: statistics, ranks and Kendall τ                                                                                                                                                                                                                                  |
| `evidence_lightweight.csv`                                                              | E7                                                                                                                                                                                                                                                                    |
| `evidence_gap_verdicts.csv`                                                             | The 12-row gap verdict table                                                                                                                                                                                                                                          |

### 21.5 Other outputs

* `reports/NOVELTY_CLAIMS.md`: literature positioning, novelty statement, gap table, limitations, abstract paragraph.
* `README.md`: short summary with result tables.
* The executed notebooks themselves (all outputs saved inside).

## 22. Every figure

All are saved to `reports/figures/` by the notebooks.

| Figure                              | Notebook | What it shows                                                                                                                                                                | What to notice                                                                         |
| ----------------------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `01_imd_tmax_2024.png`            | 01       | Left: IMD Tmax map on 28-May-2024. Right: area-average Tmax for Rajasthan, Delhi, UP and Kerala through 2024                                                                 | The north-west is 45 °C+; Kerala stays flat around 30–33 °C                         |
| `01_era5_delhi_may2024.png`       | 01       | Delhi hourly temperature, humidity and feels-like, 27–31 May 2024                                                                                                           | Daily cycle; nights stay hot                                                           |
| `02_glitch_vs_real_event.png`     | 02       | Left: the 17-Dec-2021 spike (removed). Right: 5-Apr-2020 lights-off (kept)                                                                                                   | Why the glitch rule needs two conditions                                               |
| `02_calibration_overlap.png`      | 02       | Scatter of A vs B hourly GW in Jan–Apr 2024, and histogram of their % difference                                                                                            | B sits about 0.8 % above A, hence the calibration                                      |
| `02_stitched_hourly.png`          | 02       | 2019–2025 hourly demand coloured by source                                                                                                                                  | Seamless joins, growth, the COVID dip in 2020                                          |
| `02_daily_peak_and_peak_hour.png` | 02       | Top: daily peak. Bottom: peak hour (coloured by month)                                                                                                                       | The peak hour shifting from evening to afternoon                                       |
| `02_critical_peaks.png`           | 02       | Daily peak, the rolling 90th-percentile threshold, critical days in red                                                                                                      | The threshold rises with growth                                                        |
| `02_regional_seasonality.png`     | 02       | Each region's monthly average peak, relative to its mean                                                                                                                     | NR summer peak; WR winter peak                                                         |
| `02_state_weights.png`            | 02       | Top-12 state demand shares                                                                                                                                                   | Maharashtra, UP, Gujarat lead                                                          |
| `03_delhi_2024_heatwave.png`      | 03       | Delhi 2024 Tmax, the 1991–2020 normal, IMD heatwave days shaded                                                                                                             | Heatwaves = big departures from normal                                                 |
| `03_heat_regime_days.png`         | 03       | Daily peak; heat-regime days coloured by exposure                                                                                                                            | Heat regimes coincide with the highest peaks                                           |
| `03_heat_index_surface.png`       | 03       | Heat index for every temperature × humidity combination                                                                                                                     | Humid 36 °C can feel like 50 °C                                                      |
| `04_signature_correlation.png`    | 04       | Correlation between the 19 signature features                                                                                                                                | Which climate traits go together                                                       |
| `04_cluster_model_selection.png`  | 04       | Silhouette, Calinski-Harabasz, Davies-Bouldin vs K for 4 algorithms                                                                                                          | K = 7–8 scores best; algorithms agree from K = 6                                      |
| `04_zone_map.png`                 | 04       | States on a longitude/latitude map, coloured by zone, bubble size = demand share                                                                                             | Zones are geographically compact                                                       |
| `05_target_transform.png`         | 05       | Left: MW peak with the training maximum line. Right: the % target                                                                                                            | Why % is stable and MW is not                                                          |
| `05_heat_vs_target.png`           | 05       | Target vs weekly Tmax change, vs heat-exposure change, and boxplot by heat regime                                                                                            | Heat clearly lifts the target                                                          |
| `06_halo_loss_shapes.png`         | 06       | HALO loss curves for a = 0.2 / 1.2 / 3.2 versus MSE and pinball; their gradients                                                                                             | Exponential short side, flat over side, bounded gradients                              |
| `06_halo_tuning.png`              | 06       | Left: 2024 heat/critical cost for each (a_heat, a_scarcity), best allowed a0. Right: every HALO setting's ordinary-MAE ratio vs heatwave under-forecast rate                 | Stronger conditioning is safer; most settings stay within 5 % accuracy on 2024         |
| `07_backtest_by_fold.png`         | 07       | MAPE, critical-day under-forecast rate and cost per test year, per model                                                                                                     | Consistent ordering across years                                                       |
| `07_accuracy_vs_safety.png`       | 07       | Each model: overall MAPE vs under-forecast rate on dangerous days                                                                                                            | The trade-off; HALO between L2 and Q0.8                                                |
| `07_case_studies.png`             | 07       | Actual vs L2 / Q0.8 / HALO for May–June 2024 and April–June 2025, heat days marked                                                                                         | HALO tracks the heat spikes from above                                                 |
| `07_error_by_regime.png`          | 07       | Box plots of % error by regime for L2, Q0.8 and HALO, with the −3 % reserve line                                                                                            | HALO's dangerous-day errors are shifted up                                             |
| `07_shap_importance.png`          | 07       | Average absolute SHAP of the top 20 features, L2 vs HALO                                                                                                                     | HALO relies more on heat-change features                                               |
| `08_frontier.png`                 | 08       | **The novelty figure:** ordinary-day MAE vs expected shortfall (left) and cost (right) on dangerous days, for the uniform LINEX and pinball curves and the HALO points | Every HALO point lies **below** the uniform curves, i.e. safer at equal accuracy |

## 23. Every notebook and what it prints

| Notebook                                         | Main steps                                                                                                                                                                                                   | Key printed outputs                                                                                                                                                                       |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **01 Data acquisition**                    | Runs the downloaders; inventories `data/raw`; peeks at every dataset                                                                                                                                       | 113 raw files (~379 MB); dataset A shape (46,728 × 6 after naming); national = sum of regions check; B 12,984 hours; PSP shapes; IMD map; ERA5 Delhi sample; the 33-state table          |
| **02 Load preprocessing**                  | Reading, glitch demo, calibration, stitching, daily peaks, peak-hour table, critical peaks, regional seasonality, state weights; saves interim/processed                                                     | Calibration ratios; source counts; "glitches removed: 2, short gaps interpolated: 29, hours still missing: 197"; 10 imputed dates; daily statistics; critical days per year               |
| **03 Weather & heatwaves**                 | IMD state averages, normals, IMD-only counts, EHF, T95 per state, IMD vs EHF days per state, threshold sensitivity, exposure in the record week, ERA5 daily sample, heat-index surface; saves weather tables | The IMD-only table (0 days in 2025); T95 list; per-state counts (e.g. UP: 16 IMD days, 104 EHF days, 105 final); sensitivity table; late-May 2024 exposures; Chennai heat index of 50 °C |
| **04 Clustering**                          | Signatures, correlation heatmap, 4 algorithms × 6 K, winner, map, zone profiles, crosstab with grid regions; saves zones                                                                                    | Silhouette pivot; "WINNER: ward_spatial K = 8 …"; member list; profile table; crosstab                                                                                                   |
| **05 Features**                            | Builds the 309-feature table, shows the target transform, feature groups, heat-vs-target plots, leakage audit, split summary; saves `model_table.parquet`                                                  | Table (2,557 × 318); 2024 days above the training maximum; correlations (0.716 CDH change …); max feature–target correlation 0.604; split table                                        |
| **06 Modelling**                           | HALO loss plots; gradient check (≈ 9 × 10⁻⁸); the a·σ²/2 verification table; runs the full experiment (~7 min); tuning plots and top settings; validation comparison; tree counts                     | Selected HALO parameters; tree counts; top-8 tuning rows; 2024 metrics table                                                                                                              |
| **07 Evaluation**                          | 2025 metrics for all days and per regime; backtest metrics; per-fold MAPE; accuracy-vs-safety plot; case studies; error box plots; SHAP; research-criteria check                                             | Tables of Chapter 18; the criteria printout (C1 under-forecast drop 13.5 pts; C2 ordinary MAE +10.28 % on 2025 / +5.19 % backtest; C3 reserve breach halved; C4 cost −15.4 % / −13.4 %) |
| **08 Novelty evidence**                    | E1–E7 with explanations; frontier figure; the gap-verdict table                                                                                                                                             | All tables of Chapter 19                                                                                                                                                                  |
| `reference/ML_Course_PROJECT__Prototype.ipynb` | The team's earlier synthetic-data prototype (unchanged)                                                                                                                                                      | Source of the prototype loss and success criteria                                                                                                                                         |

---

# Part G: The code, file by file

## 24. Code walkthrough with project examples

The package lives in `src/heatwave_peak/` and is installed with `pip install -e .`, so every notebook can `import heatwave_peak`.
Every function has a docstring explaining its decision; below is a plain-language map of all of them.

### 24.1 `config.py`: every setting in one place

| Name                                                                                                                    | Value                                 | Meaning                                                                                                         |
| ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `PROJECT_ROOT`, `RAW_DIR`, `INTERIM_DIR`, `PROCESSED_DIR`, `MODELS_DIR`, `FIGURES_DIR`, `METRICS_DIR`, … | paths                                 | Worked out from the file's own location, so code runs from any folder; output folders are created automatically |
| `STUDY_START`, `STUDY_END`                                                                                          | 2019-01-01, 2025-12-31                | Modelling window                                                                                                |
| `TRAIN_END`, `VALID_START/END`, `TEST_START/END`                                                                  | 2023-12-31; 2024; 2025                | Time splits                                                                                                     |
| `COVID_LOCKDOWN`                                                                                                      | 2020-03-22 → 2020-06-30              | Training weight 0                                                                                               |
| `CLIMATE_NORMAL_START/END`                                                                                            | 1991, 2020                            | WMO 30-year normal                                                                                              |
| `IMD_HW_DEPARTURE`, `IMD_SEVERE_DEPARTURE`, `IMD_PLAINS/COASTAL/HILL_ENTRY`                                       | 4.5, 6.5, 40/37/30 °C                | IMD thresholds (the warm-night rule uses 4.5 and 40)                                                            |
| `STATE_HW_AREA_FRACTION`                                                                                              | 0.25                                  | Share of a state's area needed                                                                                  |
| `NATIONAL_HW_EXPOSURE_THRESHOLD`                                                                                      | 0.20                                  | National heat-regime threshold                                                                                  |
| `CRITICAL_PEAK_WINDOW_DAYS`, `CRITICAL_PEAK_QUANTILE`                                                               | 365, 0.90                             | Critical-peak definition                                                                                        |
| `COST_UNDER_RS_PER_KWH`, `COST_OVER_RS_PER_KWH`, `PEAK_WINDOW_HOURS`                                              | 10, 2.5, 4                            | Stylised cost                                                                                                   |
| `RANDOM_SEED`                                                                                                         | 42                                    | Reproducibility                                                                                                 |
| `StateMeta`, `STATES`                                                                                               | 33 entries                            | Each state's IMD name, NLDC names, grid region, load-centre city and coordinates                                |
| `ERA5_HOURLY_VARS`, `ERA5_START/END`                                                                                | 8 variables; 2018-12-01 → 2025-12-31 | One month of lead-in for 7-day features                                                                         |

### 24.2 `data_download.py`

| Function                                       | What it does                                                                           | Project example                                                      |
| ---------------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `_download_kaggle_zip(slug, out_dir)`        | Downloads a public Kaggle dataset zip and unpacks it; skips if the folder is not empty | `halcyonvector/india-power-grid-nldc-daily-psp-reports` → 6 files |
| `download_load_datasets()`                   | Datasets A, C and the Mendeley file                                                    |                                                                      |
| `download_imd_gridded(start_year, end_year)` | Missing IMD years for tmax and tmin, with retries                                      | 70 `.GRD` files                                                    |
| `download_era5_hourly(sleep_s=20)`           | One Open-Meteo request per state, with back-off on HTTP 429                            | `era5_hourly_Tamil_Nadu.csv` (62,112 rows)                         |
| `write_state_reference_table()`              | Writes `state_metadata.csv`                                                          |                                                                      |
| `download_all()`                             | All of the above                                                                       |                                                                      |

### 24.3 `load_processing.py`

| Function                                    | What it does                                                                                | Project example                                 |
| ------------------------------------------- | ------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `read_kaggle_hourly()`                    | Reads A and renames columns (`national_mw`, `NR`, …; note the source spells "Northen") | 46,728 rows                                     |
| `read_mendeley_hourly()`                  | Reads B and drops summary rows                                                              | 12,984 rows                                     |
| `read_scada_hourly(min_slots_per_hour=3)` | 15-minute → hourly average, needing at least 3 slots                                       | Hour 14:00 = mean of 14:00, 14:15, 14:30, 14:45 |
| `read_psp_daily()`                        | Daily PSP table indexed by date                                                             | `max_demand_met_total_mw`                     |
| `flag_spikes(s, 0.10, 0.15, 0.30)`        | Glitch rule (13.2)                                                                          | Flags 17-Dec-2021 05:00                         |
| `overlap_ratio(ref, other)`               | Median ratio on clean common hours                                                          | 0.9923 for A/B                                  |
| `build_national_hourly()`                 | Stitch + calibrate + glitch-clean + short-gap fill; returns (table, calibration dict)       | 61,368 hours                                    |
| `build_regional_hourly()`                 | Cleans A's regional columns                                                                 |                                                 |
| `build_daily_peaks(hourly)`               | Daily statistics + PSP fallback                                                             | 30-May-2024 → 246,779 MW at 14:00              |
| `add_critical_peak_flag(daily)`           | Rolling P90 threshold of the previous 365 days                                              | 30-May-2024 threshold 220,169 MW → critical    |
| `load_state_daily_demand()`               | NLDC state table → IMD names, merges entities, drops industrial ones                       | 90,994 rows                                     |
| `state_demand_weights(state_daily)`       | 2019–2023 average shares                                                                   | Maharashtra 0.119                               |

### 24.4 `weather_processing.py`

| Function                       | What it does                                                                   | Project example              |
| ------------------------------ | ------------------------------------------------------------------------------ | ---------------------------- |
| `_open_imd(var, start, end)` | Opens the downloaded IMD yearly files as one object                            |                              |
| `imd_state_daily()`          | State area-means of Tmax/Tmin + heatwave and severe area fractions, 1991–2025 | Delhi Tmax on 28-May-2024    |
| `heat_index_c(t_c, rh)`      | NOAA heat index in °C                                                         | (38.3 °C, 63 %) → 50.2 °C |
| `read_era5_state(state)`     | Reads one state's CSV                                                          |                              |
| `aggregate_era5_daily(df)`   | 24 hours → the daily features of 14.6                                         | `era5_cdh24` for Chennai   |
| `era5_state_daily()`         | All 33 states                                                                  | 85,404 rows                  |
| `era5_hourly_long()`         | Stacks all hourly CSVs                                                         | 2,049,696 rows               |

### 24.5 `heatwave.py`

| Function                                       | What it does                                                                                                                | Project example                    |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `daily_normals(df, col, window=15)`          | 1991–2020 day-of-year normals, smoothed with a wrap-around 15-day window                                                   | Delhi's mid-May normal ≈ 39.5 °C |
| `_run_length(flag)`                          | Counts consecutive True values                                                                                              | [F,T,T,T,F,T] → [0,1,2,3,0,1]     |
| `add_heatwave_features(state_day)`           | All state features of 14.3 (departures, IMD declaration, warm nights, persistence, CDD, EHF, final heat day, heat severity) | `state_heat_day` for UP          |
| `national_heat_exposure(state_day, weights)` | Demand-weighted national indicators + the regime flag                                                                       | 30-May-2024: heat_exposure 0.70    |

### 24.6 `clustering.py`

| Function                                                    | What it does                                                    |
| ----------------------------------------------------------- | --------------------------------------------------------------- |
| `_circular(month)`                                        | Month → (sin, cos) on a circle                                 |
| `build_state_signatures(state_weather, state_demand)`     | 19 features per state (15.2)                                    |
| `scale_signatures(sig, geo_weight=1.5)`                   | Standardise; up-weight latitude/longitude                       |
| `compare_clusterings(X, sig, k_range=3..8)`               | 4 algorithms × 6 K, with all scores and labels                 |
| `select_clustering(scores, min_size=2)`                   | Mean-rank winner                                                |
| `name_zones(sig, labels, weights)`                        | Names like `Z5_Uttar+Punjab`                                  |
| `ZONE_FEATURES`                                           | The 23 weather features aggregated per zone                     |
| `zone_daily_features(state_day, zone_of, weights, names)` | Demand-weighted zone averages + zone heat exposure (wide table) |

### 24.7 `features.py`

| Function                                                                            | What it does                                                                                                               |
| ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `indian_holidays(years)`                                                          | Gazetted Indian holidays from the `holidays` package                                                                     |
| `MAJOR_FESTIVALS`                                                                 | Pattern matching Holi, Diwali, Dussehra, Independence Day, Republic Day, Id-ul-Fitr, Id-ul-Zuha, Christmas, Gandhi Jayanti |
| `_calendar(dates)`                                                                | The 12 calendar features + `covid_lockdown`                                                                              |
| `_load_history(daily)`                                                            | `base_mw` + the 17 load-history features                                                                                 |
| `_weather_deltas(weather, cols)`                                                  | Weekly changes (`__d7`)                                                                                                  |
| `build_model_table(daily, national_heat, zone_wide, national_weather, psp_daily)` | Assembles target, features, flags, splits and weights                                                                      |
| `NON_FEATURES`, `feature_columns(tab)`                                          | Which columns are not inputs; returns the 309 feature names                                                                |

### 24.8 `losses.py`

| Item                                                           | What it does                                                 |
| -------------------------------------------------------------- | ------------------------------------------------------------ |
| `HALOObjective(a0, a_heat, a_scarcity, u_max=4, h_min=0.05)` | The loss (defaults in code: 0.1/0.6/0.3; selected: 0.2/12/6) |
| `.register(dataset, severity, scarcity, scale, weight)`      | Attaches per-day strictness and weight to a LightGBM dataset |
| `.asymmetry(severity, scarcity)`                             | a0 + a_heat·s + a_scarcity·g                               |
| `._exp_huber(u)`                                             | exp(u), continued as a straight line above u_max             |
| `.loss(y, y_hat, a, scale)`                                  | The loss value (including the Huber part)                    |
| `.grad_hess(y, y_hat, a, scale)`                             | Gradient and Hessian (with the floor)                        |
| `.objective(preds, train_data)`                              | What LightGBM calls before every tree                        |
| `.feval(preds, eval_data)`                                   | Average HALO loss on validation, for early stopping          |
| `PrototypeAsymmetricL2(alpha_heatwave=21, alpha_ordinary=4)` | The prototype objective (baseline)                           |
| `numerical_gradient_check(obj)`                              | Compares the formula with finite differences                 |

### 24.9 `models.py`

| Item                                                    | What it does                                                                 |
| ------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `BASE_PARAMS`, `MAX_ROUNDS`, `EARLY_STOP`         | Shared LightGBM settings (6.6)                                               |
| `SplitData`                                           | Container: X, z, base, weight, severity, scarcity, hw_flag, frame            |
| `add_loss_side_info(tab)`                             | Adds `loss_severity` and `loss_scarcity`; returns the severity reference |
| `make_split(tab, features, mask)`                     | Builds a `SplitData` for chosen dates                                      |
| `to_mw(base, z_hat)`                                  | % → MW                                                                      |
| `Trained`                                             | Holds a booster + offset; `predict_z`, `predict_mw`                      |
| `_offset(tr)`, `_scale(s, ref)`                     | Target centring; MW scale                                                    |
| `fit_builtin(name, objective, tr, va, extra, rounds)` | Trains L2 / quantile models                                                  |
| `fit_halo(name, halo, tr, va, rounds)`                | Trains HALO / LINEX                                                          |
| `fit_prototype(name, proto, tr, rounds)`              | Trains the prototype                                                         |

### 24.10 `metrics.py`

| Item                              | What it does                                                            |
| --------------------------------- | ----------------------------------------------------------------------- |
| `RESERVE_BREACH_PCT = 3.0`      | Breach threshold                                                        |
| `forecast_cost_crore(y, y_hat)` | Daily stylised cost                                                     |
| `regime_metrics(y, y_hat)`      | All metrics for one set of days                                         |
| `evaluate(df, pred_col)`        | Metrics for all / ordinary / heatwave / critical / heatwave_or_critical |
| `compare_models(df, pred_cols)` | Long table over models                                                  |

### 24.11 `experiment.py`

| Item                                             | What it does                                                                                          |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| `MODEL_ORDER`                                  | The 7 forecasters reported                                                                            |
| `HALO_GRID`, `MAX_ORDINARY_MAE_RATIO = 1.05` | Tuning grid and accuracy limit                                                                        |
| `load_table()`                                 | Reads `model_table.parquet`, adds loss side info, returns (table, 309 features, severity reference) |
| `_mask(tab, start, end)`                       | Date filter                                                                                           |
| `fit_all(tr, va, halo_params, rounds)`         | Trains all six LightGBM models                                                                        |
| `predict_frame(models, split)`                 | Prediction table (plus naive)                                                                         |
| `_eval_frame(df)`                              | Drops imputed-target days                                                                             |
| `tune_halo(tr, va)`                            | The 80-setting search and selection                                                                   |
| `run_full_experiment(verbose)`                 | Tuning → validation → refit → test → backtest; saves everything                                   |

### 24.12 `evidence.py`

| Item                                                                                                                                                                                                                                                                                             | What it does                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `FOLDS`, `OOS_FOLDS`                                                                                                                                                                                                                                                                         | (2022, 2023, 2024, 2025) and (2022, 2023, 2025)                                                                 |
| `stat_*` functions, `STATS`                                                                                                                                                                                                                                                                  | Under-rate, breach, expected shortfall (heat/critical, heatwave), cost (heat/critical, all), ordinary MAE, MAPE |
| `block_bootstrap_diff`                                                                                                                                                                                                                                                                         | Paired 7-day moving-block bootstrap                                                                             |
| `diebold_mariano`                                                                                                                                                                                                                                                                              | DM test with a 7-day correlation correction and small-sample correction                                         |
| `mcnemar_under`                                                                                                                                                                                                                                                                                | Exact McNemar test                                                                                              |
| `significance_table`                                                                                                                                                                                                                                                                           | All statistics for one comparison                                                                               |
| `load_rounds_and_params`                                                                                                                                                                                                                                                                       | Reads `experiment_summary.json`                                                                               |
| `backtest(tab, feats, specs, folds)`                                                                                                                                                                                                                                                           | Generic walk-forward runner                                                                                     |
| `spec_l2`, `spec_quantile`, `spec_halo`, `spec_raw_mw`                                                                                                                                                                                                                                   | Model "recipes" for the runner                                                                                  |
| `summarise`                                                                                                                                                                                                                                                                                    | Statistics per model                                                                                            |
| `feature_ablation_sets`                                                                                                                                                                                                                                                                        | Full / no-regime / no-accumulation / no-humidity feature lists                                                  |
| `table_with_spatial_mode(mode)`                                                                                                                                                                                                                                                                | Rebuilds the table with zones / grid regions / national only                                                    |
| `beyond_training_mask`, `level_shift_table`                                                                                                                                                                                                                                                  | E5                                                                                                              |
| `exp_significance`, `exp_frontier`, `iso_accuracy_gain`, `frontier_bootstrap`, `exp_frontier_bootstrap`, `exp_component_contrast`, `exp_feature_ablation`, `ablation_bootstrap`, `exp_spatial_ablation`, `exp_level_shift`, `exp_metric_disagreement`, `exp_lightweight` | The experiments E1–E7 (each saves `reports/metrics/evidence_*.csv`)                                          |

### 24.13 Scripts and tests

* `scripts/run_pipeline.py`: stages `download`, `load`, `weather`, `cluster`, `features`, `train`, `evidence`. Each stage calls the
  functions above and saves its outputs. Default (no arguments) = everything except download.
* `scripts/run_notebooks.py`: executes the notebooks in order and saves their outputs in place (`python scripts/run_notebooks.py 03 04` runs only those).
* `tests/test_losses.py`: the 5 loss tests of Chapter 8.11 (`python -m pytest tests -q`).
* `pyproject.toml` / `requirements.txt`: package definition and libraries (numpy, pandas, pyarrow, scikit-learn, lightgbm, matplotlib, seaborn,
  requests, openpyxl, imdlib, xarray, holidays, pypdf, Jupyter tools, pytest).

---

# Part H: Decisions, limits and running it

## 25. Decision log: what, why, how

| #  | Decision (what)                                                                                 | Why                                                                                                  | How                                             |
| -- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| 1  | Forecast the **All-India daily peak**                                                     | Only national hourly data exists 2019–2025 (regional hourly stops in Apr-2024); peaks are what fail | `build_daily_peaks`                           |
| 2  | **Stitch 3 load sources**                                                                 | No single free source covers 2019–2025; May–Oct 2024 exists only in Mendeley                       | `build_national_hourly`                       |
| 3  | **Calibrate** sources with median overlap ratios                                          | A 0.8 % level step = 1.6 GW of fake growth                                                           | `overlap_ratio`                               |
| 4  | **Two-condition glitch rule**                                                             | Keep real events (lights-off, Holi), drop sensor spikes                                              | `flag_spikes`                                 |
| 5  | Fill lost report days from the **official PSP maximum**, but never train or score on them | Lags stay defined; scores stay honest                                                                | `build_daily_peaks`                           |
| 6  | **Rolling P90** critical peaks                                                            | Growth-adjusted and uses only the past                                                               | `add_critical_peak_flag`                      |
| 7  | **Demand-weighted** weather                                                               | Weather matters where electricity is used                                                            | `state_demand_weights`                        |
| 8  | Weights from **training years only**                                                      | No leakage from 2024–25                                                                             | same                                            |
| 9  | **IMD official** heatwave rule                                                            | Government definition, defensible                                                                    | `imdlib.heatwave` + `add_heatwave_features` |
| 10 | Add **EHF** (IMD OR EHF)                                                                  | IMD alone: 0 national heatwave days in 2025; misses humid heat                                       | `add_heatwave_features`                       |
| 11 | National regime at **20 %** demand exposure                                               | Rare (8 % of days) yet testable (55 in 2024, 14 in 2025)                                             | `national_heat_exposure`                      |
| 12 | ERA5 at **load centres**                                                                  | Free quota allows 1 point per state; IMD covers the area                                             | `download_era5_hourly`                        |
| 13 | **Parquet** for intermediate data                                                         | 9× smaller, 3× faster, keeps data types                                                            | Chapter 10.3                                    |
| 14 | **Spatial Ward** clustering, K by mean rank                                               | Contiguous zones; robust selection                                                                   | `select_clustering`                           |
| 15 | Target = **% above last week**                                                            | Trees cannot extrapolate growth (raw MW: short 100 % of record days)                                 | `build_model_table`                           |
| 16 | **Observed weather** as the forecast                                                      | Standard ex-post set-up                                                                              | —                                              |
| 17 | **Festival** features (major vs minor, day-after, Diwali window)                          | First validation's biggest errors were Holi and Diwali                                               | `_calendar`                                   |
| 18 | **COVID weight 0**                                                                        | Demand fell for non-weather reasons                                                                  | `build_model_table`                           |
| 19 | **Time-based** splits + refit + backtest                                                  | No future peeking; robust conclusions                                                                | `run_full_experiment`                         |
| 20 | **LightGBM**                                                                              | Small tabular data, custom loss, fast, explainable                                                   | `models.py`                                   |
| 21 | **Identical settings** for all models                                                     | Isolates the effect of the loss                                                                      | `BASE_PARAMS`                                 |
| 22 | **LINEX** shape                                                                           | Normal for small errors, convex shortfall cost, safety margin ∝ uncertainty                         | `HALOObjective`                               |
| 23 | **Per-day strictness** from heat severity + scarcity                                      | The novelty: stricter only when it matters                                                           | `.asymmetry`                                  |
| 24 | Huber cap 4, Hessian floor 0.05                                                                 | Numerical stability                                                                                  | `.grad_hess`                                  |
| 25 | **MW scaling** and **offset**                                                       | Loss tracks megawatts; fair start for custom losses                                                  | `_scale`, `_offset`                         |
| 26 | Tuning rule: min heat/critical cost s.t. ordinary MAE ≤ 105 % of L2                            | The prototype's two criteria, as a rule                                                              | `tune_halo`                                   |
| 27 | Baselines: naive, L2, Q0.6, Q0.8, prototype, LINEX                                              | Cover symmetric, textbook asymmetric, own prior work, and an ablation                                | `fit_all`                                     |
| 28 | Metrics beyond MAPE (shortfall, breach, cost)                                                   | MAPE ranks models in reverse of risk (τ = −1)                                                      | `metrics.py`                                  |
| 29 | Block bootstrap, DM, McNemar; equal-accuracy test                                               | Real, attributable, fair claims                                                                      | `evidence.py`                                 |
| 30 | Keep the zone model despite the ablation                                                        | Do not change the model after seeing results                                                         | Chapter 19.5                                    |

## 26. Honest limitations and the revision history

### 26.1 Limitations

1. **Ordinary-day accuracy cost:** HALO's ordinary MAE is 5–10 % higher than L2's (2025 +10.3 %, backtest +5.2 %). The 5 % tolerance held only on 2024.
2. **Heat term alone is not robust out of sample;** scarcity + heat together are needed.
3. **Scarcity uses the realised peak** inside the training loss. That is legitimate (losses see labels), but the model must learn it through proxies.
4. **2025 has only 52 heat/critical days,** so several 2025-only differences are not significant. The backtest is the main evidence.
5. **The stylised cost** (₹10 / ₹2.5, 4 h) is for comparing models, not for settlement.
6. **Perfect-forecast weather;** real IMD forecasts would add error.
7. **One ERA5 point per state** (the area-wide signal comes from IMD).
8. **The tuning grid's upper edge was reached** (a_heat 12, a_scarcity 6).
9. **Clustering** does not improve the national forecast.
10. **No conformal/probabilistic comparison** and no climate-scenario test.
11. **Industrial demand excluded from the weights,** and the national target includes all demand.

### 26.2 What was changed after looking at results (full disclosure)

1. **Heatwave definition (before modelling).** IMD-only gave 26 national days in 7 years and 0 in 2025, so EHF was added. This was decided
   from event counts, not model results.
2. **Festival features.** Added after the first validation errors (Holi/Diwali); validation-based.
3. **Wider HALO grid.** The first grid (a ≤ 1.5) barely changed anything on validation; widened based on the LINEX margin formula.
4. **Selection rule (after seeing 2025).** Run 1 minimised **total** cost under the 105 % rule and chose a0 = 0.3, a_heat = 8,
   a_scarcity = 4. That pushed every day up (2025: ordinary MAE +8.7 %, bias +880 MW). The rule was changed to **heat/critical cost**,
   giving a0 = 0.2, a_heat = 12, a_scarcity = 6. Run 1's outputs are kept in `reports/metrics/run1_allday_cost_rule/`, and both runs reach
   the same conclusions. To limit the bias this could introduce, the novelty evidence is reported on folds **not** used for any tuning (OOS 2022, 2023, 2025).
5. **Ex-ante scarcity** (last week's maximum vs the record) was tested on 2022–2024 only. It was worse (more ordinary-day bias), so it was not adopted.

## 27. How to run everything

```bash
# 1. environment (Python 3.10+; this project used 3.14 on Windows)
python -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
pip install -e .                  # makes `import heatwave_peak` work everywhere

# 2. full pipeline (approximate times on a laptop)
python scripts/run_pipeline.py --stages download     # ~20 min (mostly ERA5, rate-limited)
python scripts/run_pipeline.py --stages load weather cluster features   # ~35 s
python scripts/run_pipeline.py --stages train        # ~6-7 min
python scripts/run_pipeline.py --stages evidence     # ~9 min

# 3. or run the notebooks (they call the same functions and save figures)
python scripts/run_notebooks.py                      # 01 ... 08 (06 ~7 min, 08 ~9 min)

# 4. tests
python -m pytest tests -q
```

Results are deterministic (fixed seeds; LightGBM `deterministic=True`), so re-running reproduces the numbers in this guide.

## 28. Frequently asked questions

**Why not a deep neural network (LSTM, Transformer)?**
~1,700 training days is small for deep networks, which would over-fit. LightGBM is more accurate on small tables, trains in seconds,
accepts our custom loss easily, and is explainable (gap G9).

**Why forecast daily peaks, not all 24 hours?**
The peak is what determines whether enough generation is online. A 24-hour version would need hourly weather forecasts and is a natural extension.

**Why not forecast each state or region?**
Hourly regional data stops in April 2024 (before the record heatwave), and hourly state data is not public. The zones and state weights are
ready for a regional extension.

**Is using observed weather cheating?**
No, it is the standard way to evaluate the model itself without mixing in weather-forecast errors (this is how the GEFCom competitions
work). In operation, IMD's forecast would replace it, and errors would grow somewhat.

**Why is 2025 only 355 days?**
10 days had no hourly data (lost NLDC reports). Their peaks are filled from the official report only to keep "yesterday" defined, and they are not scored.

**How do I change the cost ratio or thresholds?**
Edit `config.py` (`COST_UNDER_RS_PER_KWH`, `NATIONAL_HW_EXPOSURE_THRESHOLD`, …) and re-run the stages from the one that uses the setting.

**How would I add 2026?**
When IMD publishes the 2026 yearly files, extend `STUDY_END`, re-run `download` (which skips existing files) and all later stages. The
SCADA data in dataset C already reaches September 2026.

**Why is HALO less accurate on normal days if it is "better"?**
Because "better" here means **safer where it matters**, at a small price elsewhere. Chapter 19.3 shows the price is lower than with any
fixed-bias method; Chapter 26 states the price honestly.

**What exactly is the novelty, in one line?**
The safety margin is *conditioned* on heat severity and grid scarcity inside a LINEX boosting loss. At equal everyday accuracy, this is
significantly safer than a fixed margin (Chapters 19.3 and 20).
