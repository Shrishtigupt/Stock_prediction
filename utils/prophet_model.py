from prophet import Prophet

def prophet_forecast(train, test):

    prophet_train = train.rename(columns={"Date": "ds", "Close": "y"})

    model_prophet = Prophet()

    model_prophet.fit(prophet_train)

    future = model_prophet.make_future_dataframe(periods=len(test))

    forecast = model_prophet.predict(future)

    prophet_pred = forecast['yhat'][-len(test):].values

    return prophet_pred, model_prophet