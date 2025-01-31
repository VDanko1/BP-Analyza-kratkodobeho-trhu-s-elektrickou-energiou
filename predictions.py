import pickle
import numpy as np
import requests
import pmdarima as pmd
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg


def sarima_model(periods_to_predict, market_type):
    df_dam = data_prep(market_type)

    model = pmd.auto_arima(df_dam['price'], start_p=1, start_q=0, max_p=1, max_q=0,
                           start_P=2, start_Q=0, max_P=2, max_Q=0, m=24, seasonal=True,
                           trace=True, stepwise=False, max_time=50,
                           d=1, D=1, suppress_warnings=True)

    predikcie_original = model.predict(periods_to_predict)

    predikcie_df = pd.DataFrame({
        'deliveryEnd': predikcie_original.index,
        'price': predikcie_original.values
    })

    predikcie_df['deliveryEnd'] = pd.to_datetime(predikcie_df['deliveryEnd'])
    predikcie_df['deliveryEnd'] = predikcie_df['deliveryEnd'].dt.strftime('%Y-%m-%d %H:%M')
    predikcie_df['price'] = predikcie_df['price'].round(2)

    print(predikcie_df)

    plt.figure(figsize=(10, 6))
    plt.plot(predikcie_df['deliveryEnd'], predikcie_df['price'], label='Predikcie', color='blue')
    if market_type == "IDM":
        plt.title('Predikcie cien vnutrodenného trhu - model SARIMA')
    if market_type == "DAM":
        plt.title('Predikcie cien denného trhu - model SARIMA')

    plt.xlabel('Dátum')
    plt.ylabel('Cena €/MWh')

    n = return_n(periods_to_predict)
    plt.xticks(predikcie_df['deliveryEnd'][::n], rotation=0)
    plt.legend()
    plt.savefig("Graphs/SARIMA_from_to")
    plt.show()

    return predikcie_df


def auto_regressive_model(periods_to_predict, market_type):
    df_dam = data_prep(market_type)

    model = AutoReg(df_dam['price'], lags=168)
    model_fit = model.fit()

    predictions = model_fit.predict(start=len(df_dam), end=len(df_dam) + periods_to_predict)

    predikcie_df = pd.DataFrame({
        'deliveryEnd': predictions.index,
        'price': predictions.values
    })

    predikcie_df['deliveryEnd'] = pd.to_datetime(predikcie_df['deliveryEnd'])
    predikcie_df['deliveryEnd'] = predikcie_df['deliveryEnd'].dt.strftime('%Y-%m-%d %H:%M')
    predikcie_df['price'] = predikcie_df['price'].round(2)

    plt.figure(figsize=(10, 6))
    plt.plot(predikcie_df['deliveryEnd'], predikcie_df['price'], label='Predikcie', color='blue')
    if market_type == "IDM":
        plt.title('Predikcie cien vnutrodenného trhu - model AR')
    if market_type == "DAM":
        plt.title('Predikcie cien denného trhu - model AR')

    plt.xlabel('Dátum')
    plt.ylabel('Cena €/MWh')

    n = return_n(periods_to_predict)

    plt.xticks(predikcie_df['deliveryEnd'][::n], rotation=0)
    plt.legend()
    plt.savefig("Graphs/AR_from_to")
    plt.show()

    return predikcie_df


def sarimax_model(periods_to_predict, market_type):
    df_dam = data_prep(market_type)

    df_weather_data = sarimax_exogenous_data_prep()

    exogenous_columns = ['wind_speed_10m', 'ghi', 'dhi']
    exogenous_data = df_weather_data[exogenous_columns]

    SARIMAX_model = pmd.auto_arima(df_dam[['price']], exogenous=exogenous_data,
                                   start_p=1, start_q=0, max_p=1, max_q=0,
                                   start_P=2, start_Q=0, max_P=2, max_Q=0, m=24, seasonal=True,
                                   trace=True, stepwise=False, max_time=50,
                                   d=1, D=1, suppress_warnings=True)

    exog_weights = SARIMAX_model.get_params()['exog']
    predikcie_or = SARIMAX_model.predict(periods_to_predict)

    predikcie_df = pd.DataFrame({
        'deliveryEnd': predikcie_or.index,
        'price': predikcie_or.values
    })

    predikcie_df['deliveryEnd'] = pd.to_datetime(predikcie_df['deliveryEnd'])
    predikcie_df['deliveryEnd'] = predikcie_df['deliveryEnd'].dt.strftime('%Y-%m-%d %H:%M')
    predikcie_df['price'] = predikcie_df['price'].round(2)

    print(predikcie_df)

    plt.figure(figsize=(10, 6))
    plt.plot(predikcie_df['deliveryEnd'], predikcie_df['price'], label='Predikcie', color='blue')
    if market_type == "IDM":
        plt.title('Predikcie cien vnutrodenného trhu - model SARIMAX')
    if market_type == "DAM":
        plt.title('Predikcie cien denného trhu - model SARIMAX')

    plt.xlabel('Dátum')
    plt.ylabel('Cena €/MWh')

    n = return_n(periods_to_predict)

    plt.xticks(predikcie_df['deliveryEnd'][::n], rotation=0)
    plt.legend()
    plt.savefig("Graphs/SARIMAX_from_to")
    plt.show()

    return predikcie_df


def sarimax_exogenous_data_prep():
    file_path = "Data/weather_2023-10.4.2024.csv"
    df = pd.read_csv(file_path)
    df = df.drop(columns=['period'])
    df.dropna(inplace=True)

    df["period_end"] = pd.to_datetime(df["period_end"])
    df["period_end"] = pd.to_datetime(df["period_end"], format='%Y-%m-%d %H:%M')
    print(df)

    return df


def SarimaTrainTest(number_of_days_to_predict):
    df_dam = data_prep_tt("IDM")

    train_df = df_dam.iloc[:-96, :]
    test_df = df_dam.iloc[-96:, :]

    model = pmd.auto_arima(train_df['price'], start_p=1, start_q=0, max_p=1, max_q=0,
                           start_P=2, start_Q=0, max_P=2, max_Q=0, m=24, seasonal=True,
                           trace=True, stepwise=False, max_time=50,
                           d=1, D=1, suppress_warnings=True)

    predikcie_original = model.predict(n_periods=number_of_days_to_predict)

    predikcie_df = pd.DataFrame({
        'deliveryEnd': predikcie_original.index,
        'price': predikcie_original.values
    })

    print(predikcie_df)

    test_dates = test_df.index
    predikcie = predikcie_df['price']
    test_ceny = test_df['price']

    predikcie_df = predikcie_df[predikcie_df['deliveryEnd'].isin(test_dates)]

    mse = mean_squared_error(test_ceny, predikcie)
    rmse = np.sqrt(mse)

    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)

    plt.figure(figsize=(12, 9))
    plt.plot(predikcie_df['deliveryEnd'], predikcie_df['price'], label='Predikované ceny', color='red')
    plt.plot(test_df.index, test_df['price'], label='Testovacie ceny', color='blue')
    plt.title('Predikované ceny - model SARIMA', fontsize=16)
    plt.xlabel('Dátum', fontsize=14)
    plt.ylabel('Cena €/MWh', fontsize=14)
    plt.legend()
    plt.show()


def data_prep(market_type):
    today = datetime.date.today()

    if market_type == "DAM":
        today = datetime.date.today()

        if datetime.datetime.now().hour >= 13:
            today += datetime.timedelta(days=1)

        api_url = f"https://isot.okte.sk/api/v1/dam/results?deliveryDayFrom=2024-01-01&deliveryDayTo={today}"

        with open('Data/DAM_results_2023.pkl', "rb") as file_2023:
            data_DAM2023 = pickle.load(file_2023)

        df_dam2023 = pd.DataFrame(data_DAM2023)
        df_dam2023['deliveryEnd'] = pd.to_datetime(df_dam2023['deliveryEnd'])
        df_dam2023 = df_dam2023[['deliveryEnd', 'price']]
        df_dam2023.set_index('deliveryEnd', inplace=True)
        df_dam2023['price'] = pd.to_numeric(df_dam2023['price'], errors='coerce')
        df_dam2023.dropna(inplace=True)


    elif market_type == "IDM":
        with open('Data/IDM_results_2023.pkl', "rb") as file_2023:
            data_IDM2023 = pickle.load(file_2023)
        api_url = f"https://isot.okte.sk/api/v1/idm/results?deliveryDayFrom=2024-01-01&deliveryDayTo={today}&productType=60"

        df_idm2023 = pd.DataFrame(data_IDM2023)
        df_idm2023['deliveryEnd'] = pd.to_datetime(df_idm2023['deliveryEnd'])
        df_idm2023.rename(columns={'priceWeightedAverage': 'price'}, inplace=True)
        df_idm2023 = df_idm2023[['deliveryEnd', 'price']]
        df_idm2023.set_index('deliveryEnd', inplace=True)
        df_idm2023['price'] = pd.to_numeric(df_idm2023['price'], errors='coerce')
        df_idm2023.dropna(inplace=True)

    else:
        raise ValueError("Neplatný typ marketu.")

    response = requests.get(api_url)
    data = response.json()

    response_df = pd.DataFrame(data)
    response_df['deliveryEnd'] = pd.to_datetime(response_df['deliveryEnd'])

    if market_type == "IDM":
        response_df.rename(columns={'priceWeightedAverage': 'price'}, inplace=True)

    response_df = response_df[['deliveryEnd', 'price']]
    response_df.set_index('deliveryEnd', inplace=True)
    response_df['price'] = pd.to_numeric(response_df['price'], errors='coerce')

    response_df.dropna(inplace=True)

    if market_type == "IDM":
        merged_df = pd.concat([df_idm2023, response_df])
        return merged_df

    if market_type == "DAM":
        merged_df = pd.concat([df_dam2023, response_df])
        return merged_df


def data_prep_tt(market_type):
    today = datetime.date.today()

    if market_type == "DAM":
        api_url = f"https://isot.okte.sk/api/v1/dam/results?deliveryDayFrom=2024-01-01&deliveryDayTo={today}"
    elif market_type == "IDM":
        api_url = f"https://isot.okte.sk/api/v1/idm/results?deliveryDayFrom=2024-01-01&deliveryDayTo={today}&productType=60"
    else:
        raise ValueError("Neplatný typ marketu.")

    response = requests.get(api_url)
    data = response.json()

    response_df = pd.DataFrame(data)

    response_df['deliveryEnd'] = pd.to_datetime(response_df['deliveryEnd'])

    if market_type == "IDM":
        response_df.rename(columns={'priceWeightedAverage': 'price'}, inplace=True)

    response_df = response_df[['deliveryEnd', 'price']]

    response_df.set_index('deliveryEnd', inplace=True)

    response_df['price'] = pd.to_numeric(response_df['price'], errors='coerce')

    response_df.dropna(inplace=True)

    return response_df

def return_n(periods_to_predict):
    n = 6

    if periods_to_predict <= 48:
        n = 6

    if periods_to_predict >= 48:
        n = 15

    if periods_to_predict >= 120:
        n = 30

    return n