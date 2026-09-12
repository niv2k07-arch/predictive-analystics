# Predictive Analytics Using Historical Data — Project Report

## 1. Objective
Build a predictive model to forecast future sales trends using historical data,
applying regression-based time-series forecasting techniques.

## 2. Dataset
- 48 months of historical sales data (Jan 2023 – Dec 2026)
- Columns: Date, Sales
- Contained 2 missing values and 1 duplicate row (realistic messiness)

## 3. Data Cleaning
- Removed duplicate rows
- Sorted chronologically
- Filled missing values using linear interpolation
- Set Date as the time-series index

## 4. Exploratory Data Analysis
- Identified an overall upward trend in sales over the 4-year period
- Detected seasonal fluctuations across months (sine-wave pattern)
- Visualized in: outputs/figures/sales_trend.png

## 5. Feature Engineering
- Sales_Lag1, Sales_Lag2: previous 1-2 months' sales
- Rolling_Mean_3: 3-month moving average (using only past data, no leakage)
- Time_Index: sequential counter to capture overall trend
- Month: captures seasonality

Note: An initial version of Rolling_Mean_3 accidentally included the current
month's value, causing data leakage (MAE=0.00, R²=1.0000 — an unrealistic
perfect score). This was identified and fixed by shifting the window to
only use prior months' data.

## 6. Model
- Algorithm: Linear Regression (scikit-learn)
- Train/test split: 80/20, chronological (no shuffling, to respect time order)

## 7. Evaluation Metrics
- MAE: 90.82
- RMSE: 113.98
- R² Score: 0.6822
- Average error: 4.52% of average sales

## 8. Forecast
6-month forward forecast generated using recursive prediction
(each month's forecast feeds into the next month's lag features).
See: outputs/figures/forecast.png

## 9. Conclusion
The model explains approximately 68% of the variance in monthly sales,
with an average prediction error of about 4.5%. This demonstrates
that regression-based forecasting can effectively capture trend and seasonal
patterns in historical sales data for short-term forecasting.
