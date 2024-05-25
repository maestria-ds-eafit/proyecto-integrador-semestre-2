import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.stattools import adfuller
import plotly.express as px


def plot_acf_pacf(df, variable, lags=40):
    plt.rcParams["figure.figsize"] = 18, 5

    _, axes = plt.subplots(1, 2)

    sgt.plot_acf(df[variable], zero=False, lags=lags, ax=axes[0])
    sgt.plot_pacf(df[variable], zero=False, lags=lags, ax=axes[1])

    plt.show()


def calculate_adf_test(df, column_name):
    result = adfuller(df[column_name])
    # Mostrar los resultados con sus respectivos nombres
    print("ADF Statistic: %f" % result[0])
    print("p-value: %f" % result[1])
    print("Critical Values:")
    for key, value in result[4].items():
        print("\t%s: %.3f" % (key, value))


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
