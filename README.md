# Predicting Load Type from Power Consumption Data

I built this project to predict whether a factory's electricity load at any given moment is **Light**, **Medium** or **Maximum**, using 15-minute readings from 2018.

Everything lives in one Jupyter notebook, from loading the raw CSV to comparing models. This README explains what I did, what I found, and where I'm still unsure.

---

## The problem

The dataset (`load_data.csv`) has **35,041 rows**, one every 15 minutes for the whole of 2018. Each row has the energy used (kWh), reactive power, CO2 emissions, power factors, the time of day in seconds (NSM), and the label I need to predict: `Load_Type`.

| Class | Rows | Share |
|---|---|---|
| Light_Load | 18,073 | ~52% |
| Medium_Load | 9,696 | ~28% |
| Maximum_Load | 7,272 | ~21% |

The classes are imbalanced, so I didn't trust accuracy on its own. I mostly looked at **macro F1**, which treats all three classes equally.

---

## How I approached it

### 1. Looking at the data first
Before touching any model I plotted things. A few things stood out:

- Between midnight and 8 AM, the load is almost always Light.
- Sundays are mostly Light, Saturdays partly.
- Usage_kWh separates Light from the others well, but Medium and Maximum overlap a lot, so usage alone isn't enough.

### 2. Cleaning
- **Missing values** (roughly 1 to 4% in seven columns) were filled with the median, **calculated on the training data only**, so nothing from the test month leaked in.
- **Impossible values**: some power factors were above 100 (up to ~300), which can't be right. I turned them into missing values and filled them like the rest.
- **NSM** (seconds since midnight) also had impossible values, so I rebuilt it directly from the timestamp.
- I did **not** remove large Usage_kWh outliers. They could be genuine peak readings, and tree models handle them fine.

### 3. Splitting by time, not randomly
This is time-ordered data, so a random split would let the model peek at the future. I trained on **January to November (32,064 rows)** and tested on **December (2,977 rows)**.

### 4. Features I added
- `hour`, `minute`, `dayofweek`, `is_weekend`, `is_sunday`
- `total_reactive` and `usage_per_co2` (simple power ratios)
- `is_winter` (November to February)
- `day_mean_usage` and `day_max_usage` (average and peak usage for that calendar day)

---

## Two things that changed my results the most

### The season matters (and I got this wrong at first)
My first models scored only about 0.63 to 0.65 macro F1. Early on I had dropped the month, because the month-by-month class counts looked flat and I assumed it carried no information.

That was a mistake. When I looked at hour and month **together**, I found the schedule flips with the season:

| Hour (weekdays) | Nov to Feb | Mar to Oct |
|---|---|---|
| 3 PM | Medium_Load | Maximum_Load |
| 6 PM | Maximum_Load | Medium_Load |

Peak load comes in the evening in winter and in the afternoon the rest of the year. Adding a single `is_winter` flag took macro F1 from roughly 0.63 to about 0.95.

I used a season flag rather than the raw month number because December never appears in the training data, and a flag lets the model reuse what it learned from November, January and February.

### Knowing whether the factory was running that day
After the season fix, almost all remaining errors were Light vs not-Light, and they clustered on specific days (for example Dec 25, Dec 31 and a few weekends). The model had no way to know whether the plant was working that day.

The daily average and peak usage fixed that without needing a holiday calendar. If a day's average usage is tiny, it's a day off. This pushed macro F1 to around 0.98.

---

## Models compared

I trained four models, all with class balancing:

1. **Logistic Regression**: a deliberately simple baseline
2. **Random Forest**
3. **LightGBM**
4. **XGBoost**

### Results on the December test set

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 0.687 | 0.559 |
| Random Forest | 0.987 | 0.985 |
| LightGBM | 0.999 | 0.998 |
| XGBoost | 1.000 | 1.000 |

Logistic Regression struggles because the rules here are things like "winter + 6 PM = Maximum", which are not straight-line relationships. Tree-based models capture those easily.

### Is it overfitting? (my honest answer)

A perfect score on XGBoost made me suspicious, so I checked:

- The test month was never seen during training, so this is not the usual train-versus-test overfitting.
- The feature importance plot shows CO2, time of day (NSM), `is_winter` and `day_mean_usage` doing most of the work, with no strange column dominating. That makes accidental label leakage unlikely.
- My best explanation is that `Load_Type` in this dataset follows a fairly rigid pattern based on time of day, season and whether it's a working day, which trees can learn almost completely.

Month-by-month validation (training on all earlier months and testing on each month from June to December):

| Test month | Macro F1 |
|---|---|
| June | _fill in_ |
| July | _fill in_ |
| August | _fill in_ |
| September | _fill in_ |
| October | _fill in_ |
| November | _fill in_ |
| December | _fill in_ |

Train vs test macro F1 gap: _fill in_

Final model chosen: _fill in_

---

## Limitations

- **`day_mean_usage` and `day_max_usage` use the whole day's readings**, including readings later in the day than the row being predicted. That's fine for this dataset, which arrives as one table, but it wouldn't be available in a live system predicting in real time. Without these two features, my results were around 0.95 macro F1.
- **I tested on one month.** Scores that high on a single month deserve caution, which is why I added the month-by-month check.
- **The data is from a single year and a single site.** I haven't checked how this would generalise to other years or other factories. The winter/summer pattern, in particular, might not hold elsewhere.
- Missing values were filled with medians, which is simple but not perfect.

---

## How to run it

1. Put `load_data.csv` in the same folder as the notebook.
2. Install the libraries:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost
   ```
3. Open the notebook and choose **Kernel → Restart & Run All**.

Run the cells in order. The train/test split depends on the cleaning steps above it, and the feature steps depend on the split.

## Tech used

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, LightGBM, XGBoost, Jupyter.

---

## What I'd do next

- Try a proper holiday calendar instead of inferring holidays from daily usage.
- Rebuild the model without the whole-day features, so it could work in real time.
- Test on more than one year of data.
