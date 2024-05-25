import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.stattools import adfuller, kpss
import plotly.express as px
import pandas as pd
import skforecast


def plot_acf_pacf(df, variable, lags=40):
    if variable == None:
        plt.rcParams["figure.figsize"] = 18, 5
        _, axes = plt.subplots(1, 2)
        sgt.plot_acf(df, zero=False, lags=50, ax=axes[0])
        sgt.plot_pacf(df, zero=False, lags=50, ax=axes[1])
        plt.show()
    else:
        plt.rcParams["figure.figsize"] = 18, 5
        _, axes = plt.subplots(1, 2)
        sgt.plot_acf(df[variable], zero=False, lags=lags, ax=axes[0])
        sgt.plot_pacf(df[variable], zero=False, lags=lags, ax=axes[1])
        plt.show()


def add_stl_plot(fig, res, legend):
    """Add 3 plots from a second STL fit"""
    axs = fig.get_axes()
    comps = ["trend", "seasonal", "resid"]
    for ax, comp in zip(axs[1:], comps):
        series = getattr(res, comp)
        if comp == "resid":
            ax.plot(series, marker="o", linestyle="none")
        else:
            ax.plot(series)
            if comp == "trend":
                ax.legend(legend, frameon=False)


def plot_time_series(df_positive, df_negative, df_neutral, column):
    """
    Crea y muestra una gráfica de líneas con tres series temporales: Positive, Negative y Neutral.

    Parámetros:
    df_positive (DataFrame): DataFrame con los datos de la serie temporal 'Positive'.
    df_negative (DataFrame): DataFrame con los datos de la serie temporal 'Negative'.
    df_neutral (DataFrame): DataFrame con los datos de la serie temporal 'Neutral'.
    """
    # Crear la figura inicial con la primera serie temporal
    fig_line = px.line(title="Review count time series")
    fig_line.add_scatter(
        x=df_positive.index,
        y=df_positive[column],
        mode="lines",
        name="Positive",
        line=dict(color="green"),
    )

    # Agregar la segunda serie temporal
    fig_line.add_scatter(
        x=df_negative.index,
        y=df_negative[column],
        mode="lines",
        name="Negative",
        line=dict(color="darkorange"),
    )

    # Agregar la tercera serie temporal
    fig_line.add_scatter(
        x=df_neutral.index,
        y=df_neutral[column],
        mode="lines",
        name="Neutral",
        line=dict(color="silver"),
    )

    # Mostrar el gráfico
    fig_line.show()


def split_data_into_train_test(data, test_size=30) -> tuple[pd.Series, pd.Series]:

    df_length = len(data)
    train_df = data.iloc[: (df_length - test_size)]
    test_df = data.iloc[(df_length - test_size) :]

    return train_df, test_df


def test_stationarity_and_plot(train_df, column_name):
    """
    Test the stationarity of the time series and plot the original and differenced series.

    Parameters:
    train_df (DataFrame): The dataframe containing the time series data.
    column_name (str): The name of the column containing the time series to be tested and plotted.
    """
    print("Skforecast version: ", skforecast.__version__)

    data_diff_1 = train_df[column_name].diff().dropna()
    data_diff_2 = data_diff_1.diff().dropna()

    print("Test stationarity for original series")
    print("-------------------------------------")
    adfuller_result = adfuller(train_df[column_name])
    kpss_result = kpss(train_df[column_name])
    print(f"ADF Statistic: {adfuller_result[0]}, p-value: {adfuller_result[1]}")
    print(f"KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}")

    print("\nTest stationarity for differenced series (order=1)")
    print("--------------------------------------------------")
    adfuller_result = adfuller(data_diff_1)
    kpss_result = kpss(data_diff_1)
    print(f"ADF Statistic: {adfuller_result[0]}, p-value: {adfuller_result[1]}")
    print(f"KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}")

    print("\nTest stationarity for differenced series (order=2)")
    print("--------------------------------------------------")
    adfuller_result = adfuller(data_diff_2)
    kpss_result = kpss(data_diff_2)
    print(f"ADF Statistic: {adfuller_result[0]}, p-value: {adfuller_result[1]}")
    print(f"KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}")

    # Plot series
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(7, 5), sharex=True)
    train_df[column_name].plot(ax=axs[0], title="Original time series")
    data_diff_1.plot(ax=axs[1], title="Differenced order 1")
    data_diff_2.plot(ax=axs[2], title="Differenced order 2")
    plt.tight_layout()
    plt.show()
