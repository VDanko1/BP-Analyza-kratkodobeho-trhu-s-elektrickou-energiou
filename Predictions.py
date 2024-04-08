import itertools
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox, stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from sklearn.metrics import mean_squared_error
import pandas as pd
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
    with open("Data/DAM_results_2023.pkl", "rb") as file_dam:
        data_dam = pickle.load(file_dam)
        # Load data
    with open("Data/DAM_results_2023.pkl", "rb") as file_dam:
        data_dam_2023 = pickle.load(file_dam)

    with open("Data/DAM_results_2024-JAN-APR.pkl", "rb") as file_dam:
        data_dam_2024 = pickle.load(file_dam)

    df_dam2023 = pd.DataFrame(data_dam_2023)
    df_dam2024 = pd.DataFrame(data_dam_2024)

    df_dam2023['deliveryEnd'] = pd.to_datetime(df_dam2023['deliveryEnd'])
    df_dam2024['deliveryEnd'] = pd.to_datetime(df_dam2024['deliveryEnd'])

    df_dam2023 = df_dam2023[['deliveryEnd', 'price']]
    df_dam2024 = df_dam2024[['deliveryEnd', 'price']]

    df_dam2023['price'] = pd.to_numeric(df_dam2023['price'], errors='coerce')
    df_dam2024['price'] = pd.to_numeric(df_dam2024['price'], errors='coerce')

    df_dam2023.dropna(inplace=True)
    df_dam2024.dropna(inplace=True)

    df_dam2023['price_diff'] = df_dam2023['price'].diff()
    df_dam2024['price_diff'] = df_dam2024['price'].diff()

    merged_df = pd.concat([df_dam2023, df_dam2024], ignore_index=True)
    merged_df.dropna(inplace=True)

    df_dam = pd.DataFrame(data_dam)

    df_dam['deliveryEnd'] = pd.to_datetime(df_dam['deliveryEnd'])

    df_dam = df_dam[['deliveryEnd', 'price']]

    # Ensure 'price' column is numerical
    df_dam['price'] = pd.to_numeric(df_dam['price'], errors='coerce')

    # Remove any rows with missing values
    df_dam.dropna(inplace=True)

    # Split data into train and test sets
    train_size = int(len(df_dam) * 0.8)
    train, test = df_dam.iloc[:train_size], df_dam.iloc[train_size:]

    # Grid search pro SARIMA parametry
    auto_model = auto_arima(train['price'], seasonal=True, m=24, max_order=None,
                            max_p=3, max_d=3, max_q=3, max_P=3, max_D=3, max_Q=3,
                            stepwise=True, suppress_warnings=True, error_action="ignore",
                            trace=True)

    # Získání nejlepších SARIMA parametrů
    order = auto_model.order
    seasonal_order = auto_model.seasonal_order

    print("Best SARIMA parameters (p, d, q):", order)
    print("Best seasonal SARIMA parameters (P, D, Q, s):", seasonal_order)

    # Vytvoření a trénování SARIMA modelu s nejlepšími parametry
    model = SARIMAX(train['price'], order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    sarima_model = model.fit()

    # Validace modelu, predikce budoucích cen a zobrazení grafu
    predictions = sarima_model.predict(start=test.index[0], end=test.index[-1], dynamic=False)
    mse = mean_squared_error(test['price'], predictions)
    rmse = round((mse ** 0.5), 2)
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Predikce budoucích cen pro celou délku testovací sady
    forecast_horizon = len(test)
    forecast = sarima_model.forecast(steps=forecast_horizon)

    # Vykreslení predikcí a předpovědi
    plt.figure(figsize=(20, 6))
    plt.plot(train.index, train['price'], label='Train')
    plt.plot(test.index, test['price'], label='Test', alpha=0.2)
    plt.plot(test.index, predictions, label='Predictions')
    plt.plot(test.index, forecast, label='Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('SARIMA Model Forecast')
    plt.legend()
    plt.savefig("forecast_plot.png")
    plt.show()



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
    df_dam['seasonal_diff_price'] = df_dam['price'].diff(periods=24)

    plt.figure(figsize=(20, 6))
    plt.title("Porovnanie sezónnej diferencie")
    plt.plot(df_dam['deliveryEnd'], df_dam['price'], label='Original', color='red', alpha = 0.3)
    #plt.plot(df_dam['deliveryEnd'], df_dam['diff_price'], label='Differencovane', color='green',alpha=0.5)
    plt.plot(df_dam['deliveryEnd'], df_dam['boxcox_diff_price'], label='Box cox diff', color='green')
    #plt.plot(df_dam['deliveryEnd'], df_dam['seasonal_diff_price'], label='Seasonal diff', color='aqua', alpha=0.5)

    #plt.plot(df_dam['deliveryEnd'], df_dam['boxcox_diff_price'], label='Logaritmizovane_differencovane', color='brown')
    plt.legend()

    plt.figure(figsize=(8, 6))
    stats.probplot(df_dam['boxcox_price'], dist="norm", plot=plt)  # Using probplot from scipy.stats directly
    plt.title('Q-Q Plot of DAM differenced prices of year 2023 - granularity 1 hour')
    plt.xlabel('Theoretical quantiles')
    plt.ylabel('Ordered values')
    plt.grid(True)

    plt.show()
    # Difference the data to remove trend and seasonality
    #df_dam['diff_price'] = df_dam['price'].diff()  # First difference for trend

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
    #df_dam2023 = pd.concat(dataframes)

    # Ak chcete resetovať index, môžete to urobiť nasledovne:
    #df_dam2023.reset_index(drop=True, inplace=True)

    df_dam2023 = df_dam2023[['deliveryEnd', 'price']]
    df_dam2023['deliveryEnd'] = pd.to_datetime(df_dam2023['deliveryEnd'])
    df_dam2023['price'] = pd.to_numeric(df_dam2023['price'], errors='coerce')
    df_dam2023 = df_dam2023[df_dam2023['price'] > 0]

    transformed_price, lambda_value = boxcox(df_dam2023['price'])

    df_dam2023['boxcox_price'] = transformed_price
    df_dam2023['diff_price'] = df_dam2023['price'].diff()
    #df_dam2020_2023['boxcox_diff_price'] = df_dam2020_2023['boxcox_price'].diff()
    df_dam2023['seasonal_diff_price'] = df_dam2023['price'].diff(periods=24)

    plt.figure(figsize=(20, 6))
    plt.plot(df_dam2023['deliveryEnd'], df_dam2023['price'], label='Original', color='red', alpha = 0.3)
    plt.plot(df_dam2023['deliveryEnd'], df_dam2023['diff_price'], label='Differencovane', color='blue',alpha=0.7)
    #plt.plot(df_dam2023['deliveryEnd'], df_dam2023['boxcox_price'], label='Box cox', color='green',alpha=0.7)
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

def data_preparing():
    # Load data
    with open("Data/DAM_results_2023.pkl", "rb") as file_dam:
        data_dam_2023 = pickle.load(file_dam)

    with open("Data/DAM_results_2024-JAN-APR.pkl", "rb") as file_dam:
        data_dam_2024 = pickle.load(file_dam)

    df_dam2023 = pd.DataFrame(data_dam_2023)
    df_dam2024 = pd.DataFrame(data_dam_2024)

    df_dam2023['deliveryEnd'] = pd.to_datetime(df_dam2023['deliveryEnd'])
    df_dam2024['deliveryEnd'] = pd.to_datetime(df_dam2024['deliveryEnd'])


    df_dam2023 = df_dam2023[['deliveryEnd', 'price']]
    df_dam2024 = df_dam2024[['deliveryEnd', 'price']]

    df_dam2023.set_index('deliveryEnd', inplace=True)
    df_dam2024.set_index('deliveryEnd', inplace=True)
    df_dam2023.index.freq = 'H'
    df_dam2024.index.freq = 'H'

    df_dam2023['price'] = pd.to_numeric(df_dam2023['price'], errors='coerce')
    df_dam2024['price'] = pd.to_numeric(df_dam2024['price'], errors='coerce')

    df_dam2023.dropna(inplace=True)
    df_dam2024.dropna(inplace=True)

    df_dam2023['price_diff'] = df_dam2023['price'].diff()
    df_dam2024['price_diff'] = df_dam2024['price'].diff()

    merged_df = pd.concat([df_dam2023, df_dam2024])
    merged_df.dropna(inplace=True)
    return merged_df

def ArimaModel():
    merged_df = data_preparing()

    #stepwise_fit = auto_arima(merged_df['price_diff'], trace=True, suppress_warnings=True)
    #print(stepwise_fit.summary())

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

        print(predictions_df.head())
        print(train_data)
        #print(test_data)
        """
        # Predikcia na testovacích dátach
        predictions = model_fit.forecast(steps=len(test_data))
        predictions_df = pd.DataFrame(predictions, index=test_data.index, columns=['Predicted'])
        #predictions_df.dropna(inplace=True)
        print(predictions.head())
        print(predictions_df.head())
        #inverzna transformacia
        predictions_df['Predicted_original'] = merged_df['price'].iloc[0] + predictions_df['Predicted'].cumsum()

        mse = mean_squared_error(test_data['price'], predictions_df['Predicted_original'])
        total_mse+=mse
        print(f"MSE for fold: {mse:.2f}")

        print("Priemerne MSE " + str(total_mse/n_splits))
        """


ArimaModel()