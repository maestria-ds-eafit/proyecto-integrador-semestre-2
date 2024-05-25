import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class InputData:
    def __init__(
        self,
        df: pd.DataFrame,
        sequence_length: int,
        train_size_proportion=0.8,
        scaler: MinMaxScaler | StandardScaler | None = None,
    ):
        self.sequence_length = sequence_length
        self.train_size_proportion = train_size_proportion
        self.scaler = scaler
        self.df = df[["gap"]]
        self.read_data()
        self.scale_data()

    def read_data(self):
        self.data = self.df["gap"].to_numpy().reshape(-1, 1)

    def scale_data(self):
        if not self.scaler:
            return

        self.data = self.scaler.fit_transform(self.df)

    def create_training_and_test_sets(self):
        X, y = [], []
        for i in range(len(self.data) - self.sequence_length):
            X.append(self.data[i : i + self.sequence_length])
            y.append(self.data[i + self.sequence_length])
        X, y = np.array(X), np.array(y)
        self.train_size = int(self.train_size_proportion * len(X))
        X_train, X_test = X[: self.train_size], X[self.train_size :]
        y_train, y_test = y[: self.train_size], y[self.train_size :]
        return X_train, X_test, y_train, y_test
