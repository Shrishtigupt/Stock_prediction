from prophet import Prophet
import pandas as pd

def prophet_forecast(train, future_days=30):

    prophet_train = train[['Date', 'Close']].copy()

    prophet_train.columns = ['ds', 'y']

    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True
    )

    model.fit(prophet_train)

    future = model.make_future_dataframe(periods=future_days)

    forecast = model.predict(future)

    return forecast, model