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

st.title("Stock Price Forecast Dashboard")

stocks = [
    "MSFT", "GOOGL", "AMZN", "TSLA", "META",
    "NVDA", "NFLX", "INTC", "IBM"
]

ticker = st.sidebar.selectbox("Select Company", stocks)

forecast_days = st.sidebar.slider(
    "Forecast Days",
    1,
    30,
    7
)

lag_days = st.sidebar.slider(
    "Lag Features (ML)",
    3,
    20,
    5
)

show_candles = st.sidebar.checkbox(
    "Show Candlestick Chart",
    True
)

@st.cache_data
def load_data(symbol):
    df = yf.download(
        symbol,
        start="2015-01-01",
        progress=False,
        auto_adjust=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    return df

df = load_data(ticker)

if df.empty:
    st.error("No stock data found.")
    st.stop()

date_col = df.columns[0]

required_cols = ["Open", "High", "Low", "Close"]

for col in required_cols:
    if col not in df.columns:
        st.error(f"Missing column: {col}")
        st.stop()

last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2])

daily_change = last_close - prev_close
daily_change_pct = (daily_change / prev_close) * 100

high_52 = float(df["High"].tail(252).max())
low_52 = float(df["Low"].tail(252).min())

m1, m2, m3, m4 = st.columns(4)

m1.metric("Last Close", f"${last_close:.2f}")

m2.metric(
    "Daily Change",
    f"${daily_change:.2f}",
    f"{daily_change_pct:.2f}%"
)

m3.metric("52 Week High", f"${high_52:.2f}")

m4.metric("52 Week Low", f"${low_52:.2f}")

st.subheader("Future Stock Forecast")

df_prophet = df[[date_col, "Close"]].copy()

df_prophet.columns = ["ds", "y"]

df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])
df_prophet["y"] = pd.to_numeric(df_prophet["y"])

model_prophet = Prophet(
    daily_seasonality=True
)

model_prophet.fit(df_prophet)

future = model_prophet.make_future_dataframe(
    periods=forecast_days
)

forecast = model_prophet.predict(future)

future_data = forecast.tail(forecast_days)

future_high = future_data["yhat_upper"].max()
future_low = future_data["yhat_lower"].min()

high_date = future_data.loc[
    future_data["yhat_upper"].idxmax(),
    "ds"
]

low_date = future_data.loc[
    future_data["yhat_lower"].idxmin(),
    "ds"
]

c1, c2 = st.columns(2)

c1.metric(
    "Future Highest Price",
    f"${future_high:.2f}"
)

c1.success(
    f"The stock price may reach a HIGH near ${future_high:.2f} around {high_date.date()}"
)

c2.metric(
    "Future Lowest Price",
    f"${future_low:.2f}"
)

c2.warning(
    f"The stock price may fall to a LOW near ${future_low:.2f} around {low_date.date()}"
)

st.subheader("ML Forecast")

ml_df = pd.DataFrame()

ml_df["Close"] = df["Close"]

for i in range(1, lag_days + 1):
    ml_df[f"lag_{i}"] = ml_df["Close"].shift(i)

ml_df["Target"] = ml_df["Close"].shift(-forecast_days)

ml_df.dropna(inplace=True)

features = [f"lag_{i}" for i in range(1, lag_days + 1)]

X = ml_df[features]
y = ml_df["Target"]

split_index = int(len(ml_df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

model_ml = LinearRegression()

model_ml.fit(X_train, y_train)

predictions = model_ml.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

p1, p2, p3 = st.columns(3)

p1.metric("MAE", f"{mae:.2f}")
p2.metric("MSE", f"{mse:.2f}")
p3.metric("RMSE", f"{rmse:.2f}")

latest_values = np.array(
    df["Close"].tail(lag_days)
).reshape(1, -1)

future_price_ml = model_ml.predict(latest_values)[0]

st.metric(
    f"{forecast_days}-Day ML Forecast",
    f"${future_price_ml:.2f}"
)

if show_candles:

    st.subheader("Candlestick Chart")

    fig_candle = go.Figure()

    fig_candle.add_trace(
        go.Candlestick(
            x=df[date_col],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candlestick"
        )
    )

    fig_candle.update_layout(
        xaxis_rangeslider_visible=False,
        height=600
    )

    st.plotly_chart(
        fig_candle,
        use_container_width=True
    )

st.subheader("Historical vs Forecast")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df_prophet["ds"],
        y=df_prophet["y"],
        mode="lines",
        name="Historical"
    )
)

fig.add_trace(
    go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat"],
        mode="lines",
        name="Forecast"
    )
)

fig.add_trace(
    go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_upper"],
        mode="lines",
        name="Upper Range"
    )
)

fig.add_trace(
    go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_lower"],
        mode="lines",
        name="Lower Range"
    )
)

fig.update_layout(
    height=600
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.caption(
    "Data sourced from Yahoo Finance | Forecasts generated using Prophet and Linear Regression"
)

st.markdown("Created by Bhavya S Solanki")
