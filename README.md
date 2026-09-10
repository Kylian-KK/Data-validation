# Walmart Sales Data Validation and Forecasting

## Objective

This project builds an end-to-end data validation, quality-control and forecasting pipeline for weekly Walmart sales. It combines four related datasets, checks their structural consistency, profiles data-quality issues, engineers time-series features and trains a forecasting model for store and department sales.

The project uses the Walmart Sales Forecast dataset from Kaggle and covers **45 stores over approximately three years**. Its main business objective is to support inventory planning, promotional planning and logistics through more reliable weekly sales forecasts.

> This project is an academic data-science and machine-learning exercise. The results should be interpreted as analytical estimates rather than official Walmart forecasts.

## Data Used

The project loads four CSV files:

| File | Role | Main key |
| --- | --- | --- |
| `train.csv` | Historical weekly sales by store and department | `Store + Dept + Date` |
| `features.csv` | Weather, economic and promotional variables | `Store + Date` |
| `stores.csv` | Store metadata, type and size | `Store` |
| `test.csv` | Future observations for which sales must be predicted | `Store + Dept + Date` |

The notebook reports the following dataset sizes:

| Dataset | Rows | Columns |
| --- | --- | --- |
| `train.csv` | 421,570 | 5 |
| `features.csv` | 8,190 | 12 |
| `stores.csv` | 45 | 3 |
| `test.csv` | 115,064 | 4 |

The training data covers the period from **February 5, 2010 to October 26, 2012**. The test data covers the period from **November 2, 2012 to July 26, 2013**.

## Methodology

### 1. Data loading and structural understanding

The four CSV files are loaded with Pandas. Date fields are parsed as datetime values at import time. The notebook then examines dimensions, data types, date ranges, store counts, department counts, holiday observations and descriptive statistics for `Weekly_Sales`.

The training set contains 45 stores and 81 departments. Its mean weekly sales are approximately **$15,981**, while the median is approximately **$7,612**. The difference between these two values indicates a strongly right-skewed sales distribution.

### 2. Data integration

The historical sales data is enriched through two left joins:

```
train
  └── LEFT JOIN features ON Store + Date
        └── LEFT JOIN stores ON Store
```

The same integration process is applied to the test data so that the forecasting features are available for future observations.

Before joining, the notebook checks that all stores exist in the three relevant source files and that the `features` table contains no duplicate `Store + Date` keys. After joining, the number of rows is compared with the original training set to detect a possible Cartesian explosion.

The validation shows that all 45 stores are present, there are no duplicate feature keys and the merge preserves all **421,570** training rows. The unified training dataset contains **16 columns**.

### 3. Data profiling and quality control

The notebook implements two reusable scikit-learn-compatible transformers:

- `Pre_Profiling` performs a rapid profile of the raw datasets, including dimensions, data types, missing values and duplicate rows.

- `Data_Profiling` profiles the unified dataset and adds descriptive statistics and IQR-based outlier detection.

The main data-quality findings are:

| Finding | Variables | Treatment or interpretation |
| --- | --- | --- |
| Missing promotional values | `MarkDown1` to `MarkDown5` | Replace missing values with zero when absence means no active promotion |
| Missing contextual values | `CPI`, `Unemployment` | Investigate and impute where required |
| Negative sales | `Weekly_Sales` | Treat as possible returns or anomalies rather than deleting automatically |
| Extreme sales values | `Weekly_Sales` | Preserve legitimate large-department behavior and treat carefully |
| Duplicate rows | Raw and merged datasets | None reported in the validation results |
| Join integrity | `Store`, `Date` and store metadata | No row explosion detected |

The notebook correctly distinguishes between a statistical outlier and a data error. Large sales values can be legitimate because departments and stores differ substantially in scale.

### 4. Exploratory analysis

The visual analysis follows a storytelling sequence:

1. Global sales evolution over time;

2. Sales comparison by store type;

3. Sales distributions and boxplots;

4. Holiday versus non-holiday sales;

5. Weekly and monthly seasonality;

6. Promotional Markdown analysis;

7. Correlation analysis;

8. Detection and interpretation of unusual observations.

The notebook reports an upward sales trend of approximately **6%** over the three-year period. It also identifies strong seasonal peaks near the end of the year, especially around Thanksgiving and Christmas.

The project reports that Type A stores generate the largest share of sales and that holiday weeks have, on average, approximately **7.1% higher sales** than non-holiday weeks. These results are descriptive associations and do not prove that holidays or promotions independently cause the increase.

### 5. Feature engineering

The feature-engineering stage produces **24 model features**. They include:

| Feature group | Examples |
| --- | --- |
| Store structure | `Store`, `Dept`, `Type_enc`, `Size_norm` |
| Calendar features | `Week`, `Month`, `Quarter`, `Year` |
| Cyclical time features | `Week_sin`, `Week_cos` |
| Holiday information | `IsHoliday` |
| Economic and environmental variables | `Temperature`, `Fuel_Price`, `CPI`, `Unemployment` |
| Promotion features | `MarkDown_total`, `MarkDown_active`, `is_promo_intensive` |
| Historical sales features | `lag_1`, `lag_4`, `lag_52` |
| Rolling statistics | `roll_mean_4`, `roll_mean_12`, `roll_std_4` |

The cyclical week variables represent the calendar cycle using sine and cosine transformations. Lag and rolling features represent recent and seasonal sales history, which is essential for time-series forecasting.

### 6. Modeling and evaluation

The project uses a `RandomForestRegressor` with 100 trees, a maximum depth of 15 and a minimum leaf size of 10. The data is split chronologically rather than randomly to prevent information from the future leaking into the training set.

The first 80% of the ordered observations are used for training and the most recent 20% are used for testing. The evaluation uses:

| Metric | Purpose |
| --- | --- |
| WMAE | Weighted Mean Absolute Error used for the Kaggle-style evaluation, with holiday weeks weighted five times more heavily |
| R² | Proportion of variance explained by the model |
| MAE | Mean absolute prediction error in dollars |
| MAPE | Relative error indicator, interpreted cautiously when actual sales are close to zero |

The notebook reports the following run-specific results:

| Metric | Result |
| --- | --- |
| WMAE | 1,261.39 |
| R² | 0.9864 |
| MAE | $1,229.91 |
| MAPE | 223.93% |

The high MAPE should not be interpreted in isolation. Percentage errors can become extremely large when actual sales values are small or negative. For this reason, WMAE, MAE, R² and residual analysis provide more useful complementary information for this dataset.

### 7. Time-series cross-validation

The notebook also uses `TimeSeriesSplit` with five folds. Each fold trains on earlier observations and evaluates on a later period, preserving the temporal order.

The reported fold MAEs are approximately:

```
$1,815, $2,548, $1,345, $1,705 and $1,675
```

The mean is approximately **$1,817**, with a standard deviation of approximately **$397** and a coefficient of variation of **21.9%**. The notebook classifies this level of variability as acceptable but indicates some instability between periods.

## Results and Strategic Interpretation

The analysis produces several practical findings:

- Recent sales history, especially `lag_1` and `roll_mean_4`, is among the most informative groups of features.

- Store and department identifiers capture meaningful differences in sales behavior.

- Store size and store type contribute to the explanation of sales levels.

- Markdown promotion variables have a moderate but relevant contribution.

- Holiday periods create strong seasonal effects, but their impact varies by department and store.

- Type A stores require particular attention because they account for a large share of total sales.

- The test pipeline generates **115,064 forecast rows** for the requested future period.

The project also produces a Kaggle-style submission file and visual diagnostics, including actual-versus-predicted values, residual distributions, feature importance and cross-validation errors by fold.

## Technologies

| Technology | Use in the project |
| --- | --- |
| Python | Main programming language |
| Pandas | Data loading, joining, cleaning and feature engineering |
| NumPy | Numerical calculations and cyclical features |
| Matplotlib | Time-series and model visualizations |
| Seaborn | Statistical plots and distribution analysis |
| Scikit-learn | Transformers, Random Forest, metrics and time-series validation |
| Jupyter Notebook | Interactive analysis and documentation |

## Project Structure

```
Data-validation/
├── Data_Validation_complete.ipynb
├── train.csv
├── features.csv
├── stores.csv
├── test.csv
├── walmart-data-validation.jpg
└── README.md
```

The exact output filenames depend on the execution environment and the cells selected for export.

## How to Run the Project

1. Download the Walmart Sales Forecast dataset from Kaggle.

2. Install the Python dependencies.

3. Open `Data_Validation_complete.ipynb` with Jupyter Notebook or JupyterLab.

4. Run the cells from top to bottom.

5. Inspect the validation reports, visualizations, metrics and generated forecast file.

Install the main dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```


## Limitations and Possible Improvements

The dataset covers a historical period and may not represent current retail conditions, consumer behavior or economic conditions. The model should therefore not be assumed to generalize automatically to a different period or retailer.

The use of a Random Forest provides a strong baseline, but the project could be extended with gradient-boosting methods such as XGBoost or LightGBM, hierarchical forecasting, separate models by department or store type and systematic hyperparameter optimization.

The treatment of missing Markdown values as zero is reasonable when a missing value means that no promotion was active. This assumption should be verified against the data documentation before use in another dataset.

Negative weekly sales may represent returns, corrections or other accounting effects. Replacing them with the median can be useful for a baseline model, but a production pipeline should document the business meaning of these values and consider a dedicated treatment.

The reported MAPE is very high because percentage metrics are unstable for small or negative sales. Future versions should report additional robust metrics, consider a log-scale target transformation and evaluate performance separately for high-volume and low-volume departments.



## Author

**Kylian Kouda Kuete**

This project was developed as part of an academic portfolio in data validation, forecasting and applied data science.
