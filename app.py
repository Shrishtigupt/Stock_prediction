import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils.data_loader import load_stock_data
from utils.preprocess import split_data
from utils.prophet_model import prophet_forecast

st.set_page_config(page_title="Stock Prediction App", layout="wide")

st.title("📈 Stock Price Prediction App")

st.sidebar.header("User Input")

stock_symbol = st.sidebar.text_input("Enter Stock Name", "AAPL")

if st.sidebar.button("Predict"):

    st.subheader(f"Stock Data for {stock_symbol}")

    data = load_stock_data(stock_symbol)

    # ---- FIX ADDED HERE ----
    if data is None or data.empty:
        st.error("No stock data found. Please enter a valid stock symbol.")
        st.stop()

    # Remove missing values
    data = data.dropna()

    # Check minimum rows
    if len(data) < 10:
        st.error("Not enough stock data available for prediction.")
        st.stop()
    # ------------------------

    st.write(data.tail())

    train, test, train_size = split_data(data)

    prophet_pred, prophet_model = prophet_forecast(train, test)

    st.subheader("Forecast Results")

    prophet_dates = test['Date'].values

    forecast_df = pd.DataFrame({
        'Date': prophet_dates,
        'Actual Price': test['Close'].values,
        'Predicted Price': prophet_pred
    })

    st.write(forecast_df.tail())

    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(test['Date'], test['Close'], label='Actual Price')
    ax.plot(test['Date'], prophet_pred, label='Predicted Price')

    ax.set_title(f"{stock_symbol} Stock Prediction")
    ax.legend()

    st.pyplot(fig)