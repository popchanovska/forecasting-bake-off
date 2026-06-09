# forecasting bake-off: foundation models vs classical baselines

---

## reason

People have a saying in forecasting: "simple models are best."
For many years, this was true.
Simple models like ARIMA and seasonal naïve beat complex neural networks.

But something changed recently.
New AI models called **foundation models** are pre-trained on millions of time series.
They learn general patterns from huge amounts of data.
Then they can forecast a new series without any training — this is called **zero-shot**.

A recent study on agricultural prices found that foundation models break the old rule.
They beat simple models.
This project tests if the same is true on different datasets.

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

**step 2** — for notebooks 2 and 3, set up kaggle credentials:
```
pip install kaggle
# put your kaggle.json file in ~/.kaggle/
```
You can get `kaggle.json` from your Kaggle account page → API → Create New Token.

**step 3** — open and run each notebook in order:
```
01_air_passengers.ipynb
02_energy_consumption.ipynb
03_avocado_prices.ipynb
```

---

## insights

**on clean, regular data (air passengers):**
Classical models win.
ETS and SARIMA were designed for data with a clear trend and stable seasonality.
They do very well.
Chronos still gives a good forecast but it is not always the best.

**on noisy, irregular data (energy, avocado prices):**
Classical models struggle more.
They rely on patterns in the recent history.
If the history has unusual events (like the 2017 avocado price spike), they can be confused.
Chronos has seen many different series in training.
It can recognize unusual patterns and forecast more carefully.

**main finding:**
The old rule "simple models are best" still holds for very clean, textbook data.
But on real-world noisy data, Chronos zero-shot is competitive or better.
This supports the finding from the agricultural-prices paper.
Modern foundation models change the rules.

---

## model notes

**Chronos-T5-tiny** is the smallest version of Amazon's Chronos model.
It uses ~8M parameters and runs on CPU.
It was trained on a large public collection of time series from many domains.
It needs no fine-tuning — just give it the history and it predicts the future.
