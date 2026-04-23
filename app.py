import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils.data_loader import load_stock_data
from utils.prophet_model import prophet_forecast

st.set_page_config(page_title="Stock Prediction App", layout="wide")

st.title("📈 Stock Price Prediction App")

st.sidebar.header("User Input")

Stock_df = pd.read_csv("Stock.csv")

stock_symbol = st.sidebar.selectbox(
    "Select Stock",
    Stock_df["Symbol"]
)

if st.sidebar.button("Predict"):

    st.subheader(f"Stock Data for {stock_symbol}")

    data = load_stock_data(stock_symbol)

    # Check empty data
    if data is None or data.empty:
        st.error("No stock data found.")
        st.stop()

    # Remove missing values
    data = data.dropna()

    # Minimum rows check
    if len(data) < 10:
        st.error("Not enough stock data available for prediction.")
        st.stop()

    st.write(data.tail())

    # Prophet Forecast
    forecast, prophet_model = prophet_forecast(data, 30)

    st.subheader("Future 30-Day Prediction")

    future_forecast = forecast[['ds', 'yhat']].tail(30)

    st.write(future_forecast)

    # Plot
    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(data['Date'], data['Close'], label='Historical Price')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast')

    ax.set_title(f"{stock_symbol} Future Stock Prediction")
    ax.legend()

    st.pyplot(fig)