import logging
from concurrent.futures import ProcessPoolExecutor
from math import sqrt

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def naive_model(train_df, test_df, column_name="FP_mean"):
    """
    Performs walk-forward validation on time series data.

    Parameters:
    - train_df: DataFrame containing the training data.
    - test_df: DataFrame containing the test data.
    - column_name: The name of the column to predict.

    Returns:
    - rmse: Root Mean Squared Error of the predictions.
    - mape: Mean Absolute Percentage Error of the predictions.
    """
    # Initialize history with training data
    history = [x for x in train_df[column_name]]
    predictions = []

    # Walk-forward validation
    for i in range(len(test_df)):
        # Predict using the last item in history
        yhat = history[-1]
        predictions.append(yhat)
        # Add actual observation to history for the next loop
        obs = test_df.iloc[i][column_name]
        history.append(obs)

    # Calculate performance metrics
    values = test_df[column_name].values
    rmse = sqrt(mean_squared_error(values, predictions))
    mape = mean_absolute_percentage_error(values, predictions)
    mae = mean_absolute_error(values, predictions)

    print("Naive approach:")
    print(f"RMSE: {rmse}")
    print(f"MAPE: {round(mape*100, 2)}%")
    print(f"MAE: {mae}")

    return rmse, mape


def arima_rolling_training_rolling_prediction(
    train_df, test_df, arima_order, column="FP_mean"
):
    """
    Evaluates an ARIMA model with a specified order on the given training and testing datasets.

    Parameters:
    - train_df: DataFrame containing the training data.
    - test_df: DataFrame containing the test data.
    - arima_order: A tuple (p, d, q) specifying the order of the ARIMA model.
    - column: The name of the column to predict. Defaults to "FP_mean".

    Returns:
    - mape: Mean Absolute Percentage Error of the predictions.
    - rmse: Root Mean Squared Error of the predictions.
    - predictions: A list of predicted values.
    """
    # Log transform the historical data for stability
    history = list(train_df[column])
    predictions = []

    # Predict and update model for each point in test set
    for i in range(len(test_df)):
        # Predict
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        # Observation
        obs = test_df.iloc[i][column]
        history.append(obs)

    # Reverse log transform and calculate performance metrics
    values = test_df[column].values
    mape = mean_absolute_percentage_error(values, predictions)
    rmse = sqrt(mean_squared_error(values, predictions))
    mae = mean_absolute_error(values, predictions)

    # Reporting performance
    print(f"ARIMA Order: {arima_order}")
    print(f"RMSE: {rmse}")
    print(f"MAPE: {round(mape*100, 2)}%")
    print(f"MAE: {mae}")

    return mape, rmse, [predictions, values]


def arima_train_once_predict_once(train_df, test_df, arima_order, column="FP_mean"):
    """
    Trains an ARIMA model once with the specified order and forecasts the next 'test_size' points.

    Parameters:
    - train_df: DataFrame containing the training data.
    - test_df: DataFrame containing the test data to evaluate the forecasts.
    - arima_order: A tuple (p, d, q) specifying the order of the ARIMA model.
    - test_size: The number of steps to forecast.

    Returns:
    - mape: Mean Absolute Percentage Error of the predictions.
    - rmse: Root Mean Squared Error of the predictions.
    - predictions_and_values: A list containing the predictions and actual values.
    """
    test_size = len(test_df)
    # Preparing historical data
    history = np.array(train_df[column])
    history = list(history)

    # Train model and forecast
    model = ARIMA(history, order=arima_order)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=test_size)

    # Flatten test data values for comparison
    values = test_df[column].values

    # Calculate performance metrics
    mape = mean_absolute_percentage_error(values, predictions)
    rmse = sqrt(mean_squared_error(values, predictions))
    mae = mean_absolute_error(values, predictions)

    # Report performance
    print(f"ARIMA Order: {arima_order}")
    print(f"RMSE: {rmse}")
    print(f"MAPE: {round(mape*100, 2)}%")
    print(f"MAE: {mae}")

    return mape, rmse, [predictions, values]


def evaluate_arima_model(
    train_df,
    test_df,
    arima_order,
    column,
    model=arima_rolling_training_rolling_prediction,
):
    """
    Evaluates an ARIMA model and returns its order, MAPE, and RMSE.
    If an error occurs, returns the order with None for MAPE and inf for RMSE.
    """
    try:
        mape, rmse, _ = model(train_df, test_df, arima_order, column)
        return (arima_order, mape, rmse)
    except Exception as e:
        # Optionally, you could log the exception if needed for debugging
        return (arima_order, None, float("inf"))


def find_best_arima_order_parallel(
    train_df,
    test_df,
    model,
    column="FP_mean",
    p_values=range(0, 3),
    d_values=range(0, 3),
    q_values=range(0, 3),
    logs=True,
):
    """
    Finds the best ARIMA model (p,d,q) order in parallel, based on the lowest RMSE.
    """
    orders = [(p, d, q) for p in p_values for d in d_values for q in q_values]
    best_score, best_order = float("inf"), None

    # Use ProcessPoolExecutor to parallelize the model evaluation
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                evaluate_arima_model, train_df, test_df, order, column, model
            )
            for order in orders
        ]
        for future in futures:
            order, mape, rmse = future.result()
            if rmse < best_score:
                best_score, best_order = rmse, order
                print(f"New best ARIMA: {order} with RMSE: {rmse}")
    if logs:
        logging.basicConfig(level=logging.INFO, filename="arima_log.log", filemode="w")
        logging.info(
            f"Final Best ARIMA: {best_order} --- Best RMSE: {round(best_score, 2)}"
        )
    else:
        print(f"Final Best ARIMA: {best_order} --- Best RMSE: {round(best_score, 2)}")

    return best_order, best_score


def sarima_rolling_training_rolling_prediction(
    train_df, test_df, arima_order, seasonal_order, column="FP_mean", exog=None
):
    """
    Evaluates an ARIMA model with a specified order on the given training and testing datasets.

    Parameters:
    - train_df: DataFrame containing the training data.
    - test_df: DataFrame containing the test data.
    - arima_order: A tuple (p, d, q) specifying the order of the ARIMA model.
    - column: The name of the column to predict. Defaults to "FP_mean".

    Returns:
    - mape: Mean Absolute Percentage Error of the predictions.
    - rmse: Root Mean Squared Error of the predictions.
    - predictions: A list of predicted values.
    """
    # Log transform the historical data for stability
    history = list(train_df[column])
    predictions = []

    # Predict and update model for each point in test set
    for i in range(len(test_df)):
        # Predict
        model = SARIMAX(
            history, order=arima_order, seasonal_order=seasonal_order, exog=exog
        )
        model_fit = model.fit(disp=False)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        # Observation
        obs = test_df.iloc[i][column]
        history.append(obs)

    # Reverse log transform and calculate performance metrics
    values = test_df[column].values
    mape = mean_absolute_percentage_error(values, predictions)
    rmse = sqrt(mean_squared_error(values, predictions))
    mae = mean_absolute_error(values, predictions)

    # Reporting performance
    print(f"SARIMA Order: {arima_order}")
    print(f"Seasonal Order: {seasonal_order}")
    print(f"RMSE: {rmse}")
    print(f"MAPE: {round(mape*100, 2)}%")
    print(f"MAE: {mae}")

    return mape, rmse, [predictions, values]


def plot_arima(
    test_df,
    values_arima_rolling,
    predictions_arima_rolling,
    title="Comparación de los valores reales con las predicciones del modelo ARIMA usando ROLLING TRAINING - ROLLING PREDICTIONS",
):
    # Crear un DataFrame para el plot
    df_plot = pd.DataFrame(
        {
            "Fecha": test_df.index,
            "Valor Real": values_arima_rolling,
            "Predicción": predictions_arima_rolling,
        }
    )
    # Plotting
    fig = px.line(
        df_plot,
        x="Fecha",
        y=["Valor Real", "Predicción"],
        markers=True,
        labels={"value": "Gap", "variable": "Series"},
        title=title,
    )

    fig.update_layout(legend_title_text="")
    fig.show()
