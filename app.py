import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Predictive Analytics — Sales Forecast", layout="wide")
st.title("📈 Predictive Analytics Using Historical Data")
st.caption("Thiranex Project #3 — Regression-based sales forecasting")

# --- Generate / load data ---
@st.cache_data
def load_data():
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=48, freq='MS')
    trend = np.linspace(1000, 2200, len(dates))
    seasonality = 300 * np.sin(2 * np.pi * dates.month / 12)
    noise = np.random.normal(0, 80, len(dates))
    sales = np.round(trend + seasonality + noise, 2)
    df = pd.DataFrame({'Date': dates, 'Sales': sales})
    df.loc[5, 'Sales'] = np.nan
    df.loc[20, 'Sales'] = np.nan
    df = pd.concat([df, df.iloc[[10]]], ignore_index=True)
    df = df.drop_duplicates().sort_values('Date').reset_index(drop=True)
    df['Sales'] = df['Sales'].interpolate(method='linear')
    return df

df = load_data()

st.subheader("Historical Sales Data")
st.dataframe(df.tail(10), use_container_width=True)

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df['Date'], df['Sales'], marker='o', color='#2563eb')
ax1.set_title("Monthly Sales Trend")
ax1.grid(alpha=0.3)
st.pyplot(fig1)

# --- Feature engineering ---
df['Sales_Lag1'] = df['Sales'].shift(1)
df['Sales_Lag2'] = df['Sales'].shift(2)
df['Rolling_Mean_3'] = df['Sales'].shift(1).rolling(window=3).mean()
df['Month'] = df['Date'].dt.month
df['Time_Index'] = np.arange(len(df))
df_model = df.dropna().reset_index(drop=True)

features = ['Time_Index', 'Month', 'Sales_Lag1', 'Sales_Lag2', 'Rolling_Mean_3']
X = df_model[features]
y = df_model['Sales']
split = int(len(df_model) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")
col3.metric("R² Score", f"{r2:.4f}")

fig2, ax2 = plt.subplots(figsize=(10, 4))
test_dates = df_model['Date'].iloc[split:]
ax2.plot(test_dates, y_test.values, marker='o', label='Actual', color='#2563eb')
ax2.plot(test_dates, y_pred, marker='s', label='Predicted', color='#dc2626', linestyle='--')
ax2.set_title("Actual vs Predicted (Test Set)")
ax2.legend(); ax2.grid(alpha=0.3)
st.pyplot(fig2)

# --- Forecast ---
st.subheader("6-Month Forward Forecast")
n_months = st.slider("Months to forecast", 1, 12, 6)

last_sales = list(df_model['Sales'].iloc[-3:])
future_dates = pd.date_range(start=df_model['Date'].max() + pd.DateOffset(months=1), periods=n_months, freq='MS')
future_preds = []
for i in range(n_months):
    lag1, lag2 = last_sales[-1], last_sales[-2]
    roll3 = np.mean(last_sales[-3:])
    Xf = pd.DataFrame([[split + len(y_train) + i, future_dates[i].month, lag1, lag2, roll3]], columns=features)
    pred = model.predict(Xf)[0]
    future_preds.append(pred)
    last_sales.append(pred)

future_df = pd.DataFrame({'Date': future_dates, 'Forecast': future_preds})
st.dataframe(future_df, use_container_width=True)

fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(df_model['Date'], df_model['Sales'], label='Historical', color='#2563eb')
ax3.plot(future_df['Date'], future_df['Forecast'], label='Forecast', color='#16a34a', linestyle='--', marker='o')
ax3.legend(); ax3.grid(alpha=0.3)
ax3.set_title(f"Sales Forecast — Next {n_months} Months")
st.pyplot(fig3)

st.caption("Built with scikit-learn Linear Regression | Chronological 80/20 train-test split")

