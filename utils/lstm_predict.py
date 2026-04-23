from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


def build_lstm_model():

    model = Sequential()

    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(50))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model