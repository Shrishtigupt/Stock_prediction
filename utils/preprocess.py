import numpy as np
from sklearn.preprocessing import MinMaxScaler

def split_data(data):

    train_size = int(len(data) * 0.8)

    train = data[:train_size]
    test = data[train_size:]

    return train, test, train_size


def scale_data(data):

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(data[['Close']])

    return scaled_data, scaler


def create_dataset(dataset, time_step=60):

    X, y = [], []

    for i in range(len(dataset) - time_step):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])

    return np.array(X), np.array(y)