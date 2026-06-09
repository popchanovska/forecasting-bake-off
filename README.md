# forecasting bake-off: foundation models vs classical baselines

---

## reason

People have a saying in forecasting: "simple models are best."
For many years, this was true. Simple models like ARIMA and seasonal naïve beat complex neural networks.

But something changed recently. New AI models called **foundation models** are pre-trained on millions of time series. They learn general patterns from huge amounts of data. Then they can forecast a new series without any training - this is called **zero-shot**.

A recent study on agricultural prices found that foundation models break the old rule.
They beat simple models. This project tests if the same is true on different datasets.

---

## goal

Compare a zero-shot foundation model (Chronos-T5-tiny by Amazon) against three classical methods:

- **seasonal naïve** — repeat last year's values
- **SARIMA** — a statistical model for trend and seasonality
- **ETS (Holt-Winters)** — exponential smoothing with trend and seasons

We measure MAE, RMSE, and MAPE (lower = better).

---

## datasets

| notebook | dataset | source | frequency | horizon |
|---|---|---|---|---|
| 01 | airline passengers 1949–1960 | statsmodels built-in | monthly | 24 months |
| 02 | AEP electricity demand | kaggle: `robikscube/hourly-energy-consumption` | daily (from hourly) | 30 days |
| 03 | US avocado prices | kaggle: `neuromusic/avocado-prices` | weekly | 12 weeks |

These three datasets are different.
Air passengers is very regular and clean.
Energy data is long with weekly patterns.
Avocado prices are volatile and have irregular spikes.
This range tests the models in easy and hard conditions.

---

## flow

Each notebook follows the same steps:

1. load the dataset
2. plot the data to understand the pattern
3. split into train and test sets
4. run seasonal naïve forecast
5. run SARIMA forecast
6. run ETS forecast
7. run Chronos-T5-tiny in zero-shot mode (no training on this data)
8. compare errors with a table and bar chart
9. write a short summary

---

## how to run

**step 1** — install dependencies:
```
pip install chronos-forecasting statsmodels scikit-learn matplotlib pandas torch nbformat
```

**step 2** — open and run each notebook in order:
```
01_air_passengers.ipynb
02_energy_consumption.ipynb
03_avocado_prices.ipynb
```

---

## results

### notebook 1 · air passengers

| model | MAPE |
|---|---|
| **ETS holt-winters** | **2.81%** ← winner |
| SARIMA | 14.70% |
| seasonal naïve | 15.52% |
| Chronos-tiny | 15.68% |

ETS wins by a very large margin.
Air passengers has a very clean pattern — regular trend and stable seasonality.
ETS was designed exactly for this shape, so it wins easily.
Chronos, SARIMA, and seasonal naïve all finish around 15% error.
Chronos adds no advantage over simply repeating last year's values here.

### notebook 2 · AEP energy consumption

| model | MAPE |
|---|---|
| **SARIMA** | **4.61%** ← winner |
| Chronos-tiny | 5.28% |
| seasonal naïve | 9.25% |
| ETS | 9.82% |

SARIMA wins, but Chronos is very close — only 0.67% behind.
This is impressive because Chronos never trained on this data at all.
ETS and seasonal naïve both fail at around 9–10%.
The test period (July–August 2018) had unusual demand, and these simpler models could not adapt.

### notebook 3 · avocado prices

| model | MAPE |
|---|---|
| **ETS holt-winters** | **5.76%** ← winner |
| SARIMA | 6.65% |
| Chronos-tiny | 7.82% |
| seasonal naïve | 12.44% |

ETS wins again.
After the big 2017 price spike, avocado prices returned to a stable low level.
ETS adapts to this new level well.
Chronos comes 3rd — it beats seasonal naïve but does not beat the classical models.
Seasonal naïve is worst because it repeats the spike values from late 2017.

---

## main finding

Classical models (ETS or SARIMA) won every single notebook.
Chronos did not beat them outright on any dataset.

However, Chronos is competitive.
On energy data it finished only 0.67% behind the winner — with zero training on the series.
On avocado prices it beat seasonal naïve by a large margin.

The old rule "simple models are best" still holds in these experiments.
The paper's claim that modern foundation models break this rule was not confirmed here.
A possible reason: all three datasets are relatively clean and structured.
The foundation model advantage may appear more clearly on shorter, messier, or more irregular series.

---

## model notes

**Chronos-T5-tiny** is the smallest version of Amazon's Chronos model.
It uses app. 8M parameters and runs on CPU.
It was trained on a large public collection of time series from many domains.
It needs no fine-tuning - just give it the history and it predicts the future.
