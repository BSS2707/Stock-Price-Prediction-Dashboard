import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    layout="wide"
)


st.title("Stock Price Prediction Dashboard")


stocks = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "META",
    "NVDA",
    "NFLX",
    "INTC",
    "IBM"
]


ticker = st.sidebar.selectbox(
    "Select Company",
    stocks
)


forecast_days = st.sidebar.slider(
    "Forecast Horizon",
    1,
    30,
    7
)


lag_days = st.sidebar.slider(
    "Lag Features",
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
        period="2y",
        auto_adjust=False,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.astype(float)

    return df


df = load_data(ticker)


if df.empty:
    st.error("No data found")
    st.stop()


close_prices = df["Close"]
open_prices = df["Open"]
high_prices = df["High"]
low_prices = df["Low"]


last_close = float(close_prices.iloc[-1])
prev_close = float(close_prices.iloc[-2])

daily_change = last_close - prev_close
daily_change_pct = (daily_change / prev_close) * 100

high_52 = float(high_prices.tail(252).max())
low_52 = float(low_prices.tail(252).min())


m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Last Close",
    f"${last_close:.2f}"
)

m2.metric(
    "Daily Change",
    f"${daily_change:.2f}",
    f"{daily_change_pct:.2f}%"
)

m3.metric(
    "52 Week High",
    f"${high_52:.2f}"
)

m4.metric(
    "52 Week Low",
    f"${low_52:.2f}"
)


df["MA20"] = close_prices.rolling(20).mean()
df["MA50"] = close_prices.rolling(50).mean()


st.subheader("Price Trend")


line_chart = go.Figure()

line_chart.add_trace(
    go.Scatter(
        x=df.index,
        y=close_prices,
        mode="lines",
        name="Close"
    )
)

line_chart.add_trace(
    go.Scatter(
        x=df.index,
        y=df["MA20"],
        mode="lines",
        name="MA20"
    )
)

line_chart.add_trace(
    go.Scatter(
        x=df.index,
        y=df["MA50"],
        mode="lines",
        name="MA50"
    )
)

st.plotly_chart(
    line_chart,
    use_container_width=True
)


if show_candles:

    st.subheader("Candlestick Chart")

    candle_chart = go.Figure()

    candle_chart.add_trace(
        go.Candlestick(
            x=df.index,
            open=open_prices,
            high=high_prices,
            low=low_prices,
            close=close_prices,
            name="Price"
        )
    )

    st.plotly_chart(
        candle_chart,
        use_container_width=True
    )


ml_df = pd.DataFrame()

ml_df["Close"] = close_prices


for i in range(1, lag_days + 1):

    ml_df[f"lag_{i}"] = close_prices.shift(i)


ml_df["Target"] = close_prices.shift(
    -forecast_days
)


ml_df.dropna(inplace=True)


features = [
    f"lag_{i}"
    for i in range(1, lag_days + 1)
]


X = ml_df[features]
y = ml_df["Target"]


split_index = int(
    len(ml_df) * 0.8
)


X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]


model = LinearRegression()

model.fit(
    X_train,
    y_train
)


predictions = model.predict(
    X_test
)


mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mse
)


st.subheader("Model Performance")


p1, p2, p3 = st.columns(3)

p1.metric(
    "MAE",
    f"{mae:.2f}"
)

p2.metric(
    "MSE",
    f"{mse:.2f}"
)

p3.metric(
    "RMSE",
    f"{rmse:.2f}"
)


latest_values = np.array(
    close_prices.tail(lag_days)
).reshape(1, -1)


future_price = float(
    model.predict(
        latest_values
    )[0]
)


st.subheader("Forecast")


st.metric(
    f"{forecast_days} Day Forecast",
    f"${future_price:.2f}"
)


recent = df.tail(60)


future_date = (
    recent.index[-1] +
    pd.Timedelta(days=forecast_days)
)


forecast_chart = go.Figure()


forecast_chart.add_trace(
    go.Scatter(
        x=recent.index,
        y=recent["Close"],
        mode="lines+markers",
        name="Historical"
    )
)


forecast_chart.add_trace(
    go.Scatter(
        x=[future_date],
        y=[future_price],
        mode="markers+text",
        text=[f"${future_price:.2f}"],
        textposition="top center",
        name="Forecast"
    )
)


st.subheader("Historical vs Forecast")


st.plotly_chart(
    forecast_chart,
    use_container_width=True
)
