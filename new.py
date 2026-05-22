import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Stock Forecast Dashboard",
    layout="wide"
)

st.title("Stock Price Forecast Dashboard ")

stocks = [
    "MSFT", "GOOGL", "AMZN", "TSLA", "META",
    "NVDA", "NFLX", "INTC", "IBM"
]

ticker = st.sidebar.selectbox("Select Company", stocks)
target_date = st.sidebar.date_input("Select Forecast Date ")
lag_days = st.sidebar.slider("Lag Features (ML)", 3, 20, 5)
forecast_days_ml = st.sidebar.slider("Forecast Horizon (ML)", 1, 30, 7)
show_candles = st.sidebar.checkbox("Show Candlestick Chart", True)

@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, start="2000-01-01", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.astype(float)
    return df

df = load_data(ticker)

if df.empty or "Close" not in df.columns:
    st.error("No valid data found")
    st.stop()

close_prices = df["Close"].dropna()
open_prices = df["Open"].dropna()
high_prices = df["High"].dropna()
low_prices = df["Low"].dropna()

last_close = float(close_prices.iloc[-1])
prev_close = float(close_prices.iloc[-2])
daily_change = last_close - prev_close
daily_change_pct = (daily_change / prev_close) * 100
high_52 = float(df["High"].tail(252).max())
low_52 = float(df["Low"].tail(252).min())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Last Close", f"${last_close:.2f}")
m2.metric("Daily Change", f"${daily_change:.2f}", f"{daily_change_pct:.2f}%")
m3.metric("52 Week High", f"${high_52:.2f}")
m4.metric("52 Week Low", f"${low_52:.2f}")

# Prophet Forecast
df_prophet = df.reset_index()[["Date","Close"]]
df_prophet.columns = ["ds","y"]

model_prophet = Prophet(daily_seasonality=True)
model_prophet.fit(df_prophet)

days_ahead = (pd.to_datetime(target_date) - df_prophet["ds"].max()).days
if days_ahead > 0:
    future = model_prophet.make_future_dataframe(periods=days_ahead)
    forecast = model_prophet.predict(future)
    forecast_value = forecast.loc[forecast["ds"] == pd.to_datetime(target_date), "yhat"]
    if not forecast_value.empty:
        future_price_prophet = float(forecast_value.values[0])
        st.subheader("Prophet Forecast")
        st.metric(f"Forecast for {target_date}", f"${future_price_prophet:.2f}")
else:
    st.warning("Prophet requires a future date after the last available trading day.")

# ML Forecast
ml_df = pd.DataFrame({"Close": close_prices})
for i in range(1, lag_days + 1):
    ml_df[f"lag_{i}"] = close_prices.shift(i)
ml_df["Target"] = close_prices.shift(-forecast_days_ml)
ml_df.dropna(inplace=True)

if not ml_df.empty:
    features = [f"lag_{i}" for i in range(1, lag_days + 1)]
    X, y = ml_df[features], ml_df["Target"]

    split_index = int(len(ml_df) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    if len(X_train) > 0:
        model_ml = LinearRegression()
        model_ml.fit(X_train, y_train)
        predictions = model_ml.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        st.subheader("ML Model Performance")
        p1, p2, p3 = st.columns(3)
        p1.metric("MAE", f"{mae:.2f}")
        p2.metric("MSE", f"{mse:.2f}")
        p3.metric("RMSE", f"{rmse:.2f}")

        latest_values = np.array(close_prices.tail(lag_days), dtype=float).reshape(1, -1)
        future_price_ml = float(model_ml.predict(latest_values)[0])

        st.subheader("ML Forecast")
        st.metric(f"{forecast_days_ml}-Day Forecast", f"${future_price_ml:.2f}")
    else:
        st.warning("Not enough data to train ML model with current settings.")
else:
    st.warning("ML dataset is empty. Try reducing forecast horizon or lag features.")

# Candlestick Chart
if show_candles:
    st.subheader("Candlestick Chart")
    candle_chart = go.Figure()
    candle_chart.add_trace(go.Candlestick(
        x=df.index,
        open=open_prices,
        high=high_prices,
        low=low_prices,
        close=close_prices,
        name="Candlestick"
    ))
    st.plotly_chart(candle_chart, use_container_width=True)

# Plot Comparison
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_prophet["ds"], y=df_prophet["y"], mode="lines", name="Historical"
))
if days_ahead > 0:
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Prophet Forecast"
))
if 'future_price_ml' in locals():
    future_date_ml = df.index[-1] + pd.Timedelta(days=forecast_days_ml)
    fig.add_trace(go.Scatter(
        x=[future_date_ml], y=[future_price_ml],
        mode="markers+text", text=[f"${future_price_ml:.2f}"],
        textposition="top center", name="ML Forecast"
))
st.subheader("Historical vs Forecast Comparison")
st.plotly_chart(fig, use_container_width=True)

st.caption("Data sourced from Yahoo Finance. Forecasts generated using Prophet and Linear Regression (lag features).")
st.markdown("Created by Bhavya S Solanki")
