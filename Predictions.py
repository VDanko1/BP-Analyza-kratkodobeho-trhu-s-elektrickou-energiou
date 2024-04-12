import itertools
import pickle
import time
import PySimpleGUI as sg
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import inv_boxcox
from sklearn.preprocessing import PowerTransformer
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pmdarima as pmd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox, stats, yeojohnson
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stl._stl import STL
import scipy.stats as stats
import matplotlib.pyplot as plt
import pickle
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import bartlett


def SarimaModel():
    df_dam = data_preparing()

    train_df = df_dam.iloc[:-23, :]
    test_size = df_dam.iloc[-31:, :]

    model = pmd.auto_arima(train_df['boxcox_seasonal_diff_price'], start_p=1, start_q=1, m=24, seasonal=True,
                           trace=True, stepwise=True, max_time=50,
                           d=1, D=1, )

    predikcie = model.predict(n_periods=len(test_size))

    print(predikcie)


def SarimaNaTvrdo():
    df_dam, lambda_value = data_preparing()
    print(lambda_value)
    print(df_dam.columns)

    train_df = df_dam.iloc[:-48, :]  # Trénovacia množina obsahuje všetky údaje až po posledných 720 hodinách
    test_size = df_dam.iloc[-48:, :]  # Testovacia množina zahŕňa posledných 720 hodín

    model = pmd.auto_arima(train_df['price'], start_p=1, start_q=0, max_p=1, max_q=0,
                           start_P=2, start_Q=0, max_P=2, max_Q=0, m=24, seasonal=True,
                           trace=True, stepwise=False, max_time=50,
                           d=1, D=1, suppress_warnings=True)

    predikcie_original = model.predict(n_periods=len(test_size))

    print(predikcie_original)
    plt.figure(figsize=(10, 6))
    plt.plot(test_size.index, test_size['price'], label='Testovací set', color='blue')
    # Vykreslenie grafu s predikciami
    plt.plot(test_size.index, predikcie_original, label='Predikcie', color='red')
    plt.title('Porovnanie predikcií s testovacím setom')
    plt.xlabel('Dátum')
    plt.ylabel('Rozdiel ceny')
    plt.legend()
    plt.show()

    print(predikcie_original)


def data_preparing():
    # with open("Data/DAM_results_2023.pkl", "rb") as file_dam:
    #    data_dam_2023 = pickle.load(file_dam)

    with open("Data/DAM_results_2024-JAN-APR.pkl", "rb") as file_dam:
        data_dam_2024 = pickle.load(file_dam)

    # df_dam2023 = pd.DataFrame(data_dam_2023)
    df_dam2024 = pd.DataFrame(data_dam_2024)

    # df_dam2023['deliveryStart'] = pd.to_datetime(df_dam2023['deliveryStart'])
    df_dam2024['deliveryStart'] = pd.to_datetime(df_dam2024['deliveryStart'])

    # df_dam2023 = df_dam2023[['deliveryEnd', 'price']]
    df_dam2024 = df_dam2024[['deliveryEnd', 'price']]

    # df_dam2023.set_index('deliveryEnd', inplace=True)
    df_dam2024.set_index('deliveryEnd', inplace=True)

    # df_dam2023.index.freq = 'H'
    df_dam2024.index.freq = 'H'

    # df_dam2023['price'] = pd.to_numeric(df_dam2023['price'], errors='coerce')
    df_dam2024['price'] = pd.to_numeric(df_dam2024['price'], errors='coerce')

    # df_dam2023.dropna(inplace=True)
    df_dam2024.dropna(inplace=True)

    # df_dam2023['price_diff'] = df_dam2023['price'].diff()
    df_dam2024['price_diff'] = df_dam2024['price'].diff()

    # transformed_price1, lambda_value1 = yeojohnson(df_dam2023['price'])
    pt = PowerTransformer(method='yeo-johnson', standardize=False)

    transformed_price = pt.fit_transform(df_dam2024[['price']])

    # Priradenie transformovaných hodnôt späť do DataFrame df_dam2024
    df_dam2024['yeo_johnson_price'] = transformed_price

    # Získanie hodnoty lambda z transformátora pt
    lambda_value = pt.lambdas_

    # df_dam2023['boxcox_seasonal_diff_price'] = df_dam2023['boxcox_price'].diff(periods=24)
    # df_dam2024['boxcox_seasonal_diff_price'] = df_dam2024['boxcox_price'].diff(periods=24)

    # merged_df = pd.concat([df_dam2023, df_dam2024])
    # merged_df.dropna(inplace=True)
    df_dam2024.dropna(inplace=True)

    return df_dam2024, lambda_value


SarimaNaTvrdo()


# data_preparing()

def preparingData():
    with open("Data/DAM_results_2023.pkl", "rb") as file_dam:
        data_dam = pickle.load(file_dam)

    df_dam = pd.DataFrame(data_dam)

    df_dam['deliveryEnd'] = pd.to_datetime(df_dam['deliveryEnd'])

    df_dam = df_dam[['deliveryEnd', 'price']]

    # Ensure 'price' column is numerical
    df_dam['price'] = pd.to_numeric(df_dam['price'], errors='coerce')
    prices = df_dam['price']

    # Remove any rows with missing values
    df_dam.dropna(inplace=True)
    df_dam = df_dam[df_dam['price'] > 0]

    transformed_price, lambda_value = boxcox(df_dam['price'])

    df_dam['boxcox_price'] = transformed_price
    df_dam['diff_price'] = df_dam['price'].diff()

    df_dam['boxcox_diff_price'] = df_dam['boxcox_price'].diff()
    df_dam['boxcox_seasonal_diff_price'] = df_dam['boxcox_price'].diff(periods=24)
    df_dam['seasonal_diff_price'] = df_dam['price'].diff(periods=24)

    plt.figure(figsize=(20, 6))
    plt.title("Porovnanie sezónnej diferencie")
    plt.plot(df_dam['deliveryEnd'], df_dam['price'], label='Original', color='red', alpha=0.3)
    # plt.plot(df_dam['deliveryEnd'], df_dam['diff_price'], label='Differencovane', color='green',alpha=0.5)
    plt.plot(df_dam['deliveryEnd'], df_dam['boxcox_seasonal_diff_price'], label='Box cox seasonal diff', color='green')
    plt.plot(df_dam['deliveryEnd'], df_dam['boxcox_diff_price'], label='Box cox diff', color='red')
    # plt.plot(df_dam['deliveryEnd'], df_dam['seasonal_diff_price'], label='Seasonal diff', color='aqua', alpha=0.5)
    plt.xlabel("Datum")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def SarimaModelWithoutGrid():
    with open("Data/DAM_results_2023.pkl", "rb") as file_dam:
        data_dam2023 = pickle.load(file_dam)

    with open("Data/DAM_results_2022.pkl", "rb") as file_dam:
        data_dam2022 = pickle.load(file_dam)

    with open("Data/DAM_results_2021.pkl", "rb") as file_dam:
        data_dam2021 = pickle.load(file_dam)

    with open("Data/DAM_results_2020.pkl", "rb") as file_dam:
        data_dam2020 = pickle.load(file_dam)

    df_dam2023 = pd.DataFrame(data_dam2023)
    df_dam2022 = pd.DataFrame(data_dam2022)
    df_dam2021 = pd.DataFrame(data_dam2021)
    df_dam2020 = pd.DataFrame(data_dam2020)

    # Vytvorte zoznam s jednotlivými DataFrame objektami
    dataframes = [df_dam2020, df_dam2021, df_dam2022, df_dam2023]

    # Použite concat na spojenie DataFrame objektov
    # df_dam2023 = pd.concat(dataframes)

    # Ak chcete resetovať index, môžete to urobiť nasledovne:
    # df_dam2023.reset_index(drop=True, inplace=True)

    df_dam2023 = df_dam2023[['deliveryEnd', 'price']]
    df_dam2023['deliveryEnd'] = pd.to_datetime(df_dam2023['deliveryEnd'])
    df_dam2023['price'] = pd.to_numeric(df_dam2023['price'], errors='coerce')
    df_dam2023 = df_dam2023[df_dam2023['price'] > 0]

    transformed_price, lambda_value = boxcox(df_dam2023['price'])

    df_dam2023['boxcox_price'] = transformed_price
    df_dam2023['diff_price'] = df_dam2023['price'].diff()
    # df_dam2020_2023['boxcox_diff_price'] = df_dam2020_2023['boxcox_price'].diff()
    df_dam2023['seasonal_diff_price'] = df_dam2023['price'].diff(periods=24)

    plt.figure(figsize=(20, 6))
    plt.plot(df_dam2023['deliveryEnd'], df_dam2023['price'], label='Original', color='red', alpha=0.3)
    plt.plot(df_dam2023['deliveryEnd'], df_dam2023['diff_price'], label='Differencovane', color='blue', alpha=0.7)
    # plt.plot(df_dam2023['deliveryEnd'], df_dam2023['boxcox_price'], label='Box cox', color='green',alpha=0.7)
    plt.plot(df_dam2023['deliveryEnd'], df_dam2023['seasonal_diff_price'], label='Seasonal diff', color='aqua')
    plt.legend()
    plt.show()

    """
    # Split data into train and test sets
    train_size = int(len(df_dam2020_2023) * 0.8)
    train, test = df_dam2020_2023.iloc[:train_size], df_dam2020_2023.iloc[train_size:]

    # Define SARIMA model parameters
    order = (1, 1, 1)  # Example SARIMA parameters (p, d, q)
    seasonal_order = (1, 1, 1, 31)  # Example seasonal SARIMA parameters (P, D, Q, s)

    # Fit SARIMA model
    model = SARIMAX(train['diff_price'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False,
                    enforce_invertibility=False)
    sarima_model = model.fit()

    # Validate model
    predictions_diff = sarima_model.predict(start=test.index[0], end=test.index[-1], dynamic=False)
    mse = mean_squared_error(test['diff_price'], predictions_diff)
    rmse = round((mse ** 0.5), 2)
    print(f"Root Mean Squared Error (RMSE) after differencing: {rmse}")

    # Invert differencing to get predictions in the original scale
    predictions = predictions_diff.cumsum()  # Cumulative sum to invert first difference

    # Forecast future prices
    forecast_horizon = len(test)
    forecast_diff = sarima_model.forecast(steps=forecast_horizon)
    forecast = forecast_diff.cumsum()  # Cumulative sum to invert first difference

    # Plot predictions and forecast
    plt.figure(figsize=(20, 6))
    plt.plot(df_dam2020_2023['deliveryEnd'], df_dam2020_2023['price'], label='Actual', color='blue')
    plt.plot(test['deliveryEnd'], test['price'], label='Test', alpha=0.2, color='orange')
    plt.plot(test['deliveryEnd'], predictions, label='Predictions', linestyle='dashed', color='green')
    plt.plot(test['deliveryEnd'], forecast, label='Forecast', linestyle='dashed', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('SARIMA Model Forecast after Differencing')
    plt.legend()
    plt.show()
    """


def ArimaModel():
    merged_df = data_preparing()

    # stepwise_fit = auto_arima(merged_df['price_diff'], trace=True, suppress_warnings=True)
    # print(stepwise_fit.summary())

    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    total_mse = 0

    for train_index, test_index in tscv.split(merged_df):
        warnings.filterwarnings("ignore")

        train_data = merged_df.iloc[train_index]
        test_data = merged_df.iloc[test_index]

        # Natrénovanie modelu ARIMA
        model = ARIMA(train_data['price_diff'], order=(2, 1, 2))
        model_fit = model.fit()

        predictions = model_fit.forecast(steps=len(test_data))
        predictions_df = pd.DataFrame(predictions, index=test_data.index, columns=['Predicted'])

        # Predikcia na testovacích dátach
        predictions = model_fit.forecast(steps=len(test_data))
        predictions_df = pd.DataFrame(predictions, index=test_data.index, columns=['Predicted'])
        # predictions_df.dropna(inplace=True)
        print(predictions.head())
        print(predictions_df.head())
        # inverzna transformacia
        predictions_df['Predicted_original'] = merged_df['price'].iloc[0] + predictions_df['Predicted'].cumsum()

        mse = mean_squared_error(test_data['price'], predictions_df['Predicted_original'])
        total_mse += mse
        print(f"MSE for fold: {mse:.2f}")

        print("Priemerne MSE " + str(total_mse / n_splits))


def plot_predictions(train_data, test_data, predictions):
    plt.figure(figsize=(10, 6))
    # plt.plot(train_data.index, train_data['price'], label='Train', color='blue')
    plt.plot(test_data.index, test_data['price'], label='Test', color='green')
    plt.plot(test_data.index, predictions, label='Predictions', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('ETS Predictions vs Actual')
    plt.legend()
    plt.show()


def ETSModel():
    merged_df = data_preparing()

    # Počet skladieb pre TimeSeriesSplit
    n_splits = 25
    tscv = TimeSeriesSplit(n_splits=n_splits)

    total_mse = 0
    i = 0
    # 2023-01-01 po 2024-03-01

    for train_index, test_index in tscv.split(merged_df):
        # Rozdelenie dát na trénovaciu a testovaciu sadu
        train_data = merged_df.iloc[train_index]
        test_data = merged_df.iloc[test_index]
        warnings.filterwarnings("ignore")

        # Natrénovanie ETS modelu
        model = ExponentialSmoothing(train_data['price_diff'], trend='add', seasonal='add', seasonal_periods=24)
        model_fit = model.fit()

        # Predikcia na testovacích dátach
        predictions_diff = model_fit.forecast(steps=len(test_data))

        # Spätná transformácia diferencovaných predikcií na pôvodné hodnoty
        last_price = train_data['price'].iloc[-1]  # Posledná známa hodnota
        predictions = pd.Series(predictions_diff, index=test_data.index).cumsum() + last_price

        # print(predictions.tail())

        mse = mean_squared_error(test_data['price'], predictions)
        total_mse += mse
        print(f"MSE for fold: {mse:.2f}")

        if i % 10 == 0:  # Kontrola, či je i delitelné 10
            plot_predictions(train_data, test_data, predictions)
        i += 1  # Inkrementácia i

        time.sleep(0.5)

        # Priemerne MSE cez všetky testovacie sady
    avg_mse = total_mse / n_splits
    print(f"Average MSE across all folds: {avg_mse:.2f}")
