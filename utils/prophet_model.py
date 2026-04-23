from prophet import Prophet
import pandas as pd

def prophet_forecast(train, test):

    prophet_train = train[['Date', 'Close']].copy()

    prophet_train.columns = ['ds', 'y']

    prophet_train['ds'] = pd.to_datetime(prophet_train['ds'])

    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True
    )

    model.fit(prophet_train)

    future = model.make_future_dataframe(periods=len(test))

    forecast = model.predict(future)

    predicted = forecast['yhat'].tail(len(test)).values

    return predicted, model