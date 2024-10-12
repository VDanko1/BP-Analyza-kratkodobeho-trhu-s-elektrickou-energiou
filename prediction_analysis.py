from scipy.stats import stats, norm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stl._stl import STL
import scipy.stats as stats
from datetime import datetime
import pickle
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import bartlett


def adfuller_test():
    with open("Data/DAM_results_2023.pkl", "rb") as file_dam:
        data_dam = pickle.load(file_dam)

    df_dam = pd.DataFrame(data_dam)
    df_dam = df_dam[df_dam['price'] > 0]

    df_dam['diff_price'] = df_dam['price'].diff()
    transformed_price, _ = boxcox(df_dam['price'])

    df_dam['boxcox_price'] = transformed_price
    df_dam['seasonal_diff_price'] = df_dam['price'].diff(periods=24)

    price_series = df_dam['diff_price']

    bartlett_result = bartlett(df_dam['diff_price'])
    print(f"Bartlettov test - P-hodnota: {bartlett_result.pvalue}")

    result = adfuller(price_series.dropna())

    p_value = result[1]
    print(f'ADF test - P-hodnota: {p_value}')

    if p_value <= 0.05:
        print('Časový rad je stacionárny.')
    else:
        print('Časový rad nie je stacionárny.')

def data_prep_idm15(market_type, date_from, date_to):
    years = ["2024-JAN-APR", 2023]
    dataframes = []

    for year in years:
        with open(f"Data/{market_type}_results_{year}.pkl", "rb") as file_idm:
            data_idm = pickle.load(file_idm)
        df = pd.DataFrame(data_idm)
        df['deliveryEnd'] = pd.to_datetime(df['deliveryEnd']).dt.tz_localize(None)
        if market_type == "IDM15":
            df.rename(columns={'priceWeightedAverage': 'price'}, inplace=True)
        df = df[['deliveryEnd', 'price']]
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        dataframes.append(df)

    merged_df = pd.concat(dataframes)
    date_from = datetime.strptime(date_from, "%Y-%m-%d")
    date_to = datetime.strptime(date_to, "%Y-%m-%d")
    merged_df = merged_df[(merged_df['deliveryEnd'] >= date_from) & (merged_df['deliveryEnd'] <= date_to)]

    return merged_df

def data_prep_dam_idm60(market_type, date_from, date_to):
    years = [2024, 2023, 2022, 2021, 2020]
    dataframes = []

    for year in years:
        with open(f"Data/{market_type}_results_{year}.pkl", "rb") as file_dam:
            data_dam = pickle.load(file_dam)
        df = pd.DataFrame(data_dam)
        df['deliveryEnd'] = pd.to_datetime(df['deliveryEnd']).dt.tz_localize(None)
        if market_type == "IDM":
            df.rename(columns={'priceWeightedAverage': 'price'}, inplace=True)
        df = df[['deliveryEnd', 'price']]
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        dataframes.append(df)

    merged_df = pd.concat(dataframes)
    date_from = datetime.strptime(date_from, "%Y-%m-%d")
    date_to = datetime.strptime(date_to, "%Y-%m-%d")
    merged_df = merged_df[(merged_df['deliveryEnd'] >= date_from) & (merged_df['deliveryEnd'] <= date_to)]

    return merged_df


def histogram(market_type, date_from, date_to):
    merged_df = data_prep_idm15(market_type, date_from, date_to) if market_type == "IDM15" else data_prep_dam_idm60(market_type, date_from, date_to)

    plt.figure(figsize=(10, 6))
    plt.hist(merged_df['price'], bins=15, color='skyblue', edgecolor='black', density=True, alpha=0.7, label='Histogram')

    titles = {
        "IDM": 'Histogram pre ceny vnútrodenného trhu s 60 minútovou periódou',
        "DAM": 'Histogram pre ceny denného trhu',
        "IDM15": 'Histogram pre ceny vnútrodenného trhu s 15 minútovou periódou'
    }
    plt.title(titles.get(market_type, 'Histogram pre ceny trhu'), fontsize=14)

    plt.xlabel('Cena €/MWh')
    plt.ylabel('Hustota pravdepodobnosti')
    plt.legend()
    plt.savefig("Graphs/Histogram_from_to")

def decomposition():
    with open("Data/DAM_results_2023DEC.pkl", "rb") as file_dam:
        data_dam = pickle.load(file_dam)

    df_dam = pd.DataFrame(data_dam)

    df_dam['diff_price'] = df_dam['price'].diff()

    price_series = df_dam['price']

    df_dam['deliveryEnd'] = pd.to_datetime(df_dam['deliveryEnd'])
    # df_dam.dropna()

    result = seasonal_decompose(price_series, model='additive') # 24 p

    plt.figure(figsize=(12, 12))

    plt.subplot(4, 1, 1)
    plt.plot(df_dam['deliveryEnd'], df_dam['price'], label='Pôvodné ceny')
    plt.xlabel('Dátum')
    plt.xticks(rotation=0)
    plt.ylabel('Cena')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(df_dam['deliveryEnd'], result.trend, label='Trend')
    plt.xlabel('Dátum')
    plt.xticks(rotation=0)
    plt.ylabel('Cena')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(df_dam['deliveryEnd'], result.seasonal, label='Sezónnosť')
    plt.xlabel('Dátum')
    plt.ylabel('Cena')
    plt.xticks(rotation=0)
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(df_dam['deliveryEnd'], result.resid, label='Reziduá')
    plt.xlabel('Dátum')
    plt.ylabel('Cena')
    plt.xticks(rotation=0)
    plt.legend()

    plt.tight_layout()
    # plt.savefig("Graphs/Dekompozicia_DAM_2023")


def acf_plot(market_type, date_from, date_to):
    merged_df = data_prep_idm15(market_type, date_from, date_to) if market_type == "IDM15" else data_prep_dam_idm60(market_type, date_from, date_to)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_acf(merged_df['price'], lags=24, ax=ax)

    titles = {
        "IDM": 'ACF graf pre ceny vnútrodenného trhu s 60 minútovou periódou - granularita 1 hodina',
        "DAM": 'ACF graf pre ceny denného trhu - granularita 1 hodina',
        "IDM15": 'ACF graf pre ceny vnútrodenného trhu s 15 minútovou periódou - granularita 1 hodina'
    }
    ax.set_title(titles.get(market_type, 'ACF graf pre ceny trhu'), fontsize=12)

    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    plt.savefig("Graphs/ACF_from_to", bbox_inches='tight')


def pacf_plot(market_type, date_from, date_to):
    merged_df = data_prep_idm15(market_type, date_from, date_to) if market_type == "IDM15" else data_prep_dam_idm60(market_type, date_from, date_to)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_pacf(merged_df['price'], lags=24, ax=ax)

    titles = {
        "IDM": 'PACF graf pre ceny vnútrodenného trhu s 60 minútovou periódou - granularita 1 hodina',
        "DAM": 'PACF graf pre ceny denného trhu - granularita 1 hodina',
        "IDM15": 'PACF graf pre ceny vnútrodenného trhu s 15 minútovou periódou - granularita 1 hodina'
    }
    ax.set_title(titles.get(market_type, 'PACF graf pre ceny trhu'), fontsize=12)

    ax.set_xlabel('Lag')
    ax.set_ylabel('PACF')
    plt.savefig("Graphs/PACF_from_to")

def qq_plot(market_type, date_from, date_to):
    merged_df = data_prep_idm15(market_type, date_from, date_to) if market_type == "IDM15" else data_prep_dam_idm60(market_type, date_from, date_to)

    plt.figure(figsize=(10, 6))
    stats.probplot(merged_df['price'], dist="norm", plot=plt)

    titles = {
        "IDM": 'Q-Q graf pre ceny vnutrodenného trhu s 60 minútovou periódou - granularita 1 hodina',
        "DAM": 'Q-Q graf pre ceny denného trhu - granularita 1 hodina',
        "IDM15": 'Q-Q graf pre ceny vnutrodenného trhu s 15 minútovou periódou - granularita 1 hodina'
    }
    plt.title(titles.get(market_type, 'Q-Q graf pre ceny trhu'), fontsize=12)

    plt.xlabel('Teoretické kvantily')
    plt.ylabel('Usporiadané hodnoty')
    plt.savefig("Graphs/QQ_plot_from_to")

