import yfinance as yf
import pandas as pd

def load_stock_data(symbol):

    stock = yf.download(symbol, start="2018-01-01", end="2024-01-01")

    if stock.empty:
        return pd.DataFrame()

    data = stock[['Adj Close']].reset_index()

    data.dropna(inplace=True)

    data.columns = ['Date', 'Close']

    data['Date'] = pd.to_datetime(data['Date'])

    return data