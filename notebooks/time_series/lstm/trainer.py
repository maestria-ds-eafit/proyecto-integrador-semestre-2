import numpy as np
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Trainer:
    def __init__(
        self,
        num_epochs: int,
        optimizer,
        criterion,
        model,
        train_loader,
        scaler: MinMaxScaler | StandardScaler | None,
    ):
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.model = model
        self.train_loader = train_loader
        self.scaler = scaler

    def train(self, print_epochs=True):
        for epoch in range(self.num_epochs):
            for inputs, targets in self.train_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if print_epochs:
                print(
                    f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {round(loss.item(), 4)}"
                )

    def evaluate(self, X_test, y_test, print_metrics=True):
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_pred = self.model(X_test_tensor).numpy()
            if self.scaler:
                y_pred = self.scaler.inverse_transform(y_pred)
                y_test = self.scaler.inverse_transform(y_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        if print_metrics:
            print(f"Root Mean Squared Error (RMSE): {rmse}")
            print(f"Mean Absolute Error (MAE): {mae:.2f}")
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        self.model.train()
        return {
            "y_pred": y_pred,
            "y_test": y_test,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
        }
