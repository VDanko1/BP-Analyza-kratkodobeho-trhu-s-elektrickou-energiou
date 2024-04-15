import pickle
import requests
import pmdarima as pmd
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg


def SarimaPredikcie(number_of_days_to_predict, market_type):
    df_dam = data_preparing(market_type)

    model = pmd.auto_arima(df_dam['price'], start_p=1, start_q=0, max_p=1, max_q=0,
                           start_P=2, start_Q=0, max_P=2, max_Q=0, m=24, seasonal=True,
                           trace=True, stepwise=False, max_time=50,
                           d=1, D=1, suppress_warnings=True)

    predikcie_original = model.predict(number_of_days_to_predict)

    predikcie_df = pd.DataFrame({
        'deliveryEnd': predikcie_original.index,
        'price': predikcie_original.values
    })

    predikcie_df['deliveryEnd'] = pd.to_datetime(predikcie_df['deliveryEnd'])
    predikcie_df['deliveryEnd'] = predikcie_df['deliveryEnd'].dt.strftime('%Y-%m-%d %H:%M')
    predikcie_df['price'] = predikcie_df['price'].round(2)

    print(predikcie_df)

    plt.figure(figsize=(10, 6))  # Zväčšenie veľkosti grafu
    plt.plot(predikcie_df['deliveryEnd'], predikcie_df['price'], label='Predikcie', color='blue')
    if market_type == "IDM":
        plt.title('Predikcie cien vnutrodenného trhu - model SARIMA')
    if market_type == "DAM":
        plt.title('Predikcie cien denného trhu - model SARIMA')

    plt.xlabel('Dátum')
    plt.ylabel('Cena €/MWh')

    n = 6

    if number_of_days_to_predict <= 48:
        n = 6

    if number_of_days_to_predict >= 48:
        n = 15

    if number_of_days_to_predict >= 120:
        n = 30

    plt.xticks(predikcie_df['deliveryEnd'][::n], rotation=0)
    plt.legend()
    plt.savefig("Graphs/SARIMA_from_to")
    plt.show()

    return predikcie_df

def SARIMAX(number_of_days_to_predict, market_type):

    df_dam = data_preparing(market_type)

    df_dam = preparing_sarimax_data(df_dam)

    print(df_dam.columns)
    print(df_dam.tail())
    print(df_dam.head())

    """
    
    model = pmd.auto_arima(df_dam['price'], start_p=1, start_q=0, max_p=1, max_q=0,
                           start_P=2, start_Q=0, max_P=2, max_Q=0, m=24, seasonal=True,
                           trace=True, stepwise=False, max_time=50,
                           d=1, D=1, suppress_warnings=True)



    predikcie_original = model.predict(number_of_days_to_predict)

    predikcie_df = pd.DataFrame({
        'deliveryEnd': predikcie_original.index,
        'price': predikcie_original.values
    })

    predikcie_df['deliveryEnd'] = pd.to_datetime(predikcie_df['deliveryEnd'])
    predikcie_df['deliveryEnd'] = predikcie_df['deliveryEnd'].dt.strftime('%Y-%m-%d %H:%M')
    predikcie_df['price'] = predikcie_df['price'].round(2)

    print(predikcie_df)

    plt.figure(figsize=(10, 6))  # Zväčšenie veľkosti grafu
    plt.plot(predikcie_df['deliveryEnd'], predikcie_df['price'], label='Predikcie', color='blue')
    if market_type == "IDM":
        plt.title('Predikcie cien vnutrodenného trhu - model SARIMA')
    if market_type == "DAM":
        plt.title('Predikcie cien denného trhu - model SARIMA')

    plt.xlabel('Dátum')
    plt.ylabel('Cena €/MWh')

    n = 6

    if number_of_days_to_predict <= 48:
        n = 6

    if number_of_days_to_predict >= 48:
        n = 15

    if number_of_days_to_predict >= 120:
        n = 30

    plt.xticks(predikcie_df['deliveryEnd'][::n], rotation=0)
    plt.legend()
    plt.savefig("Graphs/SARIMA_from_to")
    plt.show()

    return predikcie_df
    """

def SarimaTrainTest(number_of_days_to_predict):
    df_dam = data_preparing_working("DAM")

    train_df = df_dam.iloc[:-24, :]  # Trénovacia množina obsahuje všetky údaje až po posledných 720 hodinách
    test_size = df_dam.iloc[-24:, :]  # Testovacia množina zahŕňa posledných 720 hodín

    model = pmd.auto_arima(df_dam['price'], start_p=1, start_q=0, max_p=1, max_q=0,
                           start_P=2, start_Q=0, max_P=2, max_Q=0, m=24, seasonal=True,
                           trace=True, stepwise=False, max_time=50,
                           d=1, D=1, suppress_warnings=True)

    predikcie_original = model.predict(n_periods=number_of_days_to_predict)

    predikcie_df = pd.DataFrame({
        'deliveryEnd': predikcie_original.index,  # Prvý stĺpec bude dátumy
        'price': predikcie_original.values  # Druhý stĺpec bude ceny
    })

    print(predikcie_original)

    plt.figure(figsize=(12, 9))  # Zväčšenie veľkosti grafu
    plt.plot(predikcie_df['deliveryEnd'], predikcie_df['price'], label='Predikované ceny', color='red')
    plt.title('Predikované ceny - model SARIMA', fontsize=16)
    plt.xlabel('Dátum', fontsize=14)
    plt.ylabel('Cena €/MWh', fontsize=14)
    plt.legend()
    plt.show()

    # Formátovanie dátumov na x-ovej osi
    # n = 4  # Každý n-tý dátum sa zobrazí
    # formatted_dates = pd.to_datetime(predikcie_df['deliveryEnd'][::n]).strftime('%Y-%m-%d %H:%M:%S')
    # plt.xticks(predikcie_df['deliveryEnd'][::n], formatted_dates, rotation=20)


def preparing_sarimax_data(df):
    df['previous_year_date'] = df.index - pd.DateOffset(years=1)

    # Pridanie minuloročných cien na základe dátumov
    df['previous_year_price'] = df['previous_year_date'].map(df['price'].shift(freq='365D'))

    return df

def data_preparing(market_type):
    today = datetime.date.today()
    api_url = ""

    if market_type == "DAM":
        api_url = f"https://isot.okte.sk/api/v1/dam/results?deliveryDayFrom=2024-01-01&deliveryDayTo={today}"
        with open('Data/DAM_results_2023.pkl', "rb") as file_2022:
            data_DAM2023 = pickle.load(file_2022)

        df_dam2023 = pd.DataFrame(data_DAM2023)
        df_dam2023['deliveryEnd'] = pd.to_datetime(df_dam2023['deliveryEnd'])
        df_dam2023 = df_dam2023[['deliveryEnd', 'price']]
        df_dam2023.set_index('deliveryEnd', inplace=True)
        df_dam2023['price'] = pd.to_numeric(df_dam2023['price'], errors='coerce')
        df_dam2023.dropna(inplace=True)

    elif market_type == "IDM":
        with open('Data/IDM_results_2023.pkl', "rb") as file_2022:
            data_IDM2023 = pickle.load(file_2022)
        api_url = f"https://isot.okte.sk/api/v1/idm/results?deliveryDayFrom=2024-01-01&deliveryDayTo={today}&productType=60"

        df_idm2023 = pd.DataFrame(data_IDM2023)
        df_idm2023['deliveryEnd'] = pd.to_datetime(df_idm2023['deliveryEnd'])
        df_idm2023.rename(columns={'priceWeightedAverage': 'price'}, inplace=True)
        df_idm2023 = df_idm2023[['deliveryEnd', 'price']]
        df_idm2023.set_index('deliveryEnd', inplace=True)
        df_idm2023['price'] = pd.to_numeric(df_idm2023['price'], errors='coerce')
        df_idm2023.dropna(inplace=True)

    else:
        print("Neplatný market_type.")

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


def data_preparing_working(market_type):
    today = datetime.date.today()
    api_url = ""

    if market_type == "DAM":
        api_url = f"https://isot.okte.sk/api/v1/dam/results?deliveryDayFrom=2024-01-01&deliveryDayTo={today}"

    elif market_type == "IDM":
        api_url = f"https://isot.okte.sk/api/v1/idm/results?deliveryDayFrom=2024-01-01&deliveryDayTo={today}&productType=60"
    else:
        print("Neplatný market_type.")

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


#SarimaPredikcie(24, "IDM")
SARIMAX(7,"DAM")

