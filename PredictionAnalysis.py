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


def AdFullerTestDAM():
    with open("Data/DAM_results_2023.pkl", "rb") as file_dam:
        data_dam = pickle.load(file_dam)

    df_dam = pd.DataFrame(data_dam)
    df_dam = df_dam[df_dam['price'] > 0]

    df_dam['diff_price'] = df_dam['price'].diff()
    transformed_price, lambda_value = boxcox(df_dam['price'])

    df_dam['boxcox_price'] = transformed_price
    df_dam['seasonal_diff_price'] = df_dam['price'].diff(periods=24)

    price_series = df_dam['diff_price']

    bartlett_result = bartlett(df_dam['diff_price'])
    print("Bartlettův test - Statistika:", bartlett_result.statistic)
    print("Bartlettův test - P-hodnota:", bartlett_result.pvalue)

    if bartlett_result.pvalue < 0.05:
        print(
            "P-hodnota je nižší než 0.05, což naznačuje, že existují statisticky významné rozdíly ve variabilitě mezi skupinami dat.")
        print("Nulová hypotéza o stejné variabilitě mezi skupinami dat je zamítnuta.")
    else:
        print(
            "P-hodnota je vyšší než 0.05, což naznačuje, že neexistují statisticky významné rozdíly ve variabilitě mezi skupinami dat.")
        print("Nulová hypotéza o stejné variabilitě mezi skupinami dat není zamítnuta.")

    result = adfuller(price_series.dropna())

    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]
    print("")
    print("Results of Dickey–Fuller test of price stacionarity for DAM market - 2023")
    print("")
    print(f'ADF Statistic: {adf_statistic}')
    print(f'p-value: {p_value}')
    print('Critical Values:')
    for key, value in critical_values.items():
        print(f'   {key}: {value}')

    if p_value <= 0.05:
        print('Null hypothesis (inability to reject the unit root) is rejected, the time series is stationary.')
    else:
        print('Null hypothesis (inability to reject the unit root) is not rejected, the time series is non-stationary.')


def merging_data_and_preparation_IDM15(market_type, date_from, date_to):
    with open(f"Data/{market_type}_results_2024-JAN-APR.pkl", "rb") as file_dam:
        data_dam_2024 = pickle.load(file_dam)

    with open(f"Data/{market_type}_results_2023.pkl", "rb") as file_dam:
        data_dam_2023 = pickle.load(file_dam)

    date_from = datetime.strptime(date_from, "%Y-%m-%d")
    date_to = datetime.strptime(date_to, "%Y-%m-%d")

    df_dam2024 = pd.DataFrame(data_dam_2024)
    df_dam2023 = pd.DataFrame(data_dam_2023)

    for df in [df_dam2024, df_dam2023]:
        df['deliveryEnd'] = pd.to_datetime(df['deliveryEnd']).dt.tz_localize(None)

    if market_type == "IDM15":
        for df in [df_dam2024, df_dam2023]:
            df.rename(columns={'priceWeightedAverage': 'price'}, inplace=True)

    df_dam2024 = df_dam2024[['deliveryEnd', 'price']]
    df_dam2023 = df_dam2023[['deliveryEnd', 'price']]

    for df in [df_dam2024, df_dam2023]:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')

    merged_df = pd.concat([df_dam2024, df_dam2023])
    merged_df = merged_df[(merged_df['deliveryEnd'] >= date_from) & (merged_df['deliveryEnd'] <= date_to)]
    return merged_df


def merging_data_and_preparation_dam_idm(market_type, date_from, date_to):
    with open(f"Data/{market_type}_results_2024-JAN-APR.pkl", "rb") as file_dam:
        data_dam_2024 = pickle.load(file_dam)

    with open(f"Data/{market_type}_results_2023.pkl", "rb") as file_dam:
        data_dam_2023 = pickle.load(file_dam)

    with open(f"Data/{market_type}_results_2022.pkl", "rb") as file_dam:
        data_dam2022 = pickle.load(file_dam)

    with open(f"Data/{market_type}_results_2021.pkl", "rb") as file_dam:
        data_dam2021 = pickle.load(file_dam)

    with open(f"Data/{market_type}_results_2020.pkl", "rb") as file_dam:
        data_dam2020 = pickle.load(file_dam)

    date_from = datetime.strptime(date_from, "%Y-%m-%d")
    date_to = datetime.strptime(date_to, "%Y-%m-%d")

    df_dam2024 = pd.DataFrame(data_dam_2024)
    df_dam2023 = pd.DataFrame(data_dam_2023)
    df_dam2022 = pd.DataFrame(data_dam2022)
    df_dam2021 = pd.DataFrame(data_dam2021)
    df_dam2020 = pd.DataFrame(data_dam2020)

    for df in [df_dam2024, df_dam2023, df_dam2022, df_dam2021, df_dam2020]:
        df['deliveryEnd'] = pd.to_datetime(df['deliveryEnd']).dt.tz_localize(None)

    if market_type == "IDM":
        for df in [df_dam2024, df_dam2023, df_dam2022, df_dam2021, df_dam2020]:
            df.rename(columns={'priceWeightedAverage': 'price'}, inplace=True)

    df_dam2024 = df_dam2024[['deliveryEnd', 'price']]
    df_dam2023 = df_dam2023[['deliveryEnd', 'price']]
    df_dam2022 = df_dam2022[['deliveryEnd', 'price']]
    df_dam2021 = df_dam2021[['deliveryEnd', 'price']]
    df_dam2020 = df_dam2020[['deliveryEnd', 'price']]

    for df in [df_dam2024, df_dam2023, df_dam2022, df_dam2021, df_dam2020]:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')

    merged_df = pd.concat([df_dam2020, df_dam2021, df_dam2022, df_dam2023, df_dam2024])
    merged_df = merged_df[(merged_df['deliveryEnd'] >= date_from) & (merged_df['deliveryEnd'] <= date_to)]
    return merged_df


def Histogram(market_type, date_from, date_to):
    if market_type == "IDM15":
        merged_df = merging_data_and_preparation_IDM15(market_type, date_from, date_to)
    else:
        merged_df = merging_data_and_preparation_dam_idm(market_type, date_from, date_to)

    plt.figure(figsize=(10, 6))
    plt.hist(merged_df['price'], bins=15, color='skyblue', edgecolor='black', density=True, alpha=0.7,
             label='Histogram')

    if market_type == "IDM":
        plt.title('Histogram pre ceny vnútrodenného trhu s 60 minútovou periódou', fontsize=14)
    elif market_type == "DAM":
        plt.title('Histogram pre ceny denného trhu', fontsize=14)
    elif market_type == "IDM15":
        plt.title('Histogram pre ceny vnútrodenného trhu s 15 minútovou periódou', fontsize=14)

    plt.xlabel('Cena €/MWh')
    plt.ylabel('Hustota pravdepodobnosti')
    plt.legend()
    plt.savefig("Graphs/Histogram_from_to")
    plt.show()


def AdFullerTestIDM15():
    with open("Data/IDM_results_2023_15min.pkl", "rb") as file_dam:
        data_dam = pickle.load(file_dam)

    df_dam = pd.DataFrame(data_dam)

    price_series = df_dam['priceAverage']

    result = adfuller(price_series.dropna())

    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]
    print("Results of Dickey–Fuller test of price stacionarity for IDM market 15 minute interval - 2023")
    print("")
    print(f'ADF Statistic: {adf_statistic}')
    print(f'p-value: {p_value}')
    print('Critical Values:')
    for key, value in critical_values.items():
        print(f'   {key}: {value}')

    if p_value <= 0.05:
        print('Null hypothesis (inability to reject the unit root) is rejected, the time series is stationary.')
    else:
        print('Null hypothesis (inability to reject the unit root) is not rejected, the time series is non-stationary.')


def DecompositionOfTimeSeries():
    with open("Data/DAM_results_2023DEC.pkl", "rb") as file_dam:
        data_dam = pickle.load(file_dam)

    df_dam = pd.DataFrame(data_dam)

    df_dam['diff_price'] = df_dam['price'].diff()

    price_series = df_dam['price']

    df_dam['deliveryEnd'] = pd.to_datetime(df_dam['deliveryEnd'])
    # df_dam.dropna()

    result = seasonal_decompose(price_series, model='additive')  # Perioda 24 pre dennú sezónnosť

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
    # plt.savefig("Graphs/Dekompozicia_DAM_2023_Additive")
    plt.show()


def STLDecomposition():
    with open("Data/DAM_results_2023.pkl", "rb") as file_dam:
        data_dam2023 = pickle.load(file_dam)

    with open("Data/DAM_results_2020.pkl", "rb") as file_dam:
        data_dam2020 = pickle.load(file_dam)

    with open("Data/DAM_results_2021.pkl", "rb") as file_dam:
        data_dam2021 = pickle.load(file_dam)

    with open("Data/DAM_results_2022.pkl", "rb") as file_dam:
        data_dam2022 = pickle.load(file_dam)

    # Vytvorenie DataFrame z vašich dát
    df_dam2023 = pd.DataFrame(data_dam2023)
    df_dam2022 = pd.DataFrame(data_dam2022)
    df_dam2021 = pd.DataFrame(data_dam2021)
    df_dam2020 = pd.DataFrame(data_dam2020)

    # Prevod stĺpca 'deliveryEnd' na dátumový typ
    df_dam2023['deliveryEnd'] = pd.to_datetime(df_dam2023['deliveryEnd'])
    df_dam2022['deliveryEnd'] = pd.to_datetime(df_dam2022['deliveryEnd'])
    df_dam2021['deliveryEnd'] = pd.to_datetime(df_dam2021['deliveryEnd'])
    df_dam2020['deliveryEnd'] = pd.to_datetime(df_dam2020['deliveryEnd'])

    # Nastavenie 'deliveryEnd' ako indexu DataFrame
    df_dam2023.set_index('deliveryEnd', inplace=True)
    df_dam2022.set_index('deliveryEnd', inplace=True)
    df_dam2021.set_index('deliveryEnd', inplace=True)
    df_dam2020.set_index('deliveryEnd', inplace=True)

    combined_years = pd.concat([df_dam2020, df_dam2021, df_dam2022, df_dam2023], ignore_index=False)
    combined_years.dropna()

    combined_years['diff_price'] = combined_years['price'].diff()

    # STl dekompozícia
    stl = STL(combined_years['price'], seasonal=13)  # 13 je priemerná sezónna dĺžka v roku

    # Vykonanie dekompozície
    result = stl.fit()

    # Vykreslenie dekompozície
    plt.figure(figsize=(12, 14))

    plt.subplot(4, 1, 1)
    plt.plot(combined_years['price'], label='Pôvodné ceny')
    plt.xlabel('Dátum', fontsize=11)  # Veľkosť textu nastavená na 14
    plt.xticks(rotation=0)  # Nastavenie rotácie textu na x-ovej osi
    plt.ylabel('Cena', fontsize=11)  # Veľkosť textu nastavená na 14
    plt.legend(prop={'size': 16})

    plt.subplot(4, 1, 2)
    plt.plot(combined_years.index, result.trend, label='Trend')
    plt.xlabel('Dátum', fontsize=11)  # Veľkosť textu nastavená na 14
    plt.xticks(rotation=0)  # Nastavenie rotácie textu na x-ovej osi
    plt.ylabel('Cena', fontsize=11)  # Veľkosť textu nastavená na 14
    plt.legend(prop={'size': 16})

    plt.subplot(4, 1, 3)
    plt.plot(combined_years.index, result.seasonal, label='Sezónnosť')
    plt.xlabel('Dátum', fontsize=11)  # Veľkosť textu nastavená na 14
    plt.xticks(rotation=0)  # Nastavenie rotácie textu na x-ovej osi
    plt.ylabel('Cena', fontsize=11)  # Veľkosť textu nastavená na 14
    plt.legend(prop={'size': 16})

    plt.subplot(4, 1, 4)
    plt.plot(combined_years.index, result.resid, label='Reziduá')
    plt.xlabel('Dátum', fontsize=11)  # Veľkosť textu nastavená na 14
    plt.xticks(rotation=0)  # Nastavenie rotácie textu na x-ovej osi
    plt.ylabel('Cena', fontsize=11)  # Veľkosť textu nastavená na 14
    plt.legend(prop={'size': 16})

    plt.show()

    plt.savefig("Graphs/Dekompozícia_STL_2023_DEC_DAM")
    plt.tight_layout()
    plt.show()


def ACF(market_type, date_from, date_to):
    if (market_type == "IDM15"):
        merged_df = merging_data_and_preparation_IDM15(market_type, date_from, date_to)
    else:
        merged_df = merging_data_and_preparation_dam_idm(market_type, date_from, date_to)

    fig, ax = plt.subplots(figsize=(10, 6))  # Definícia veľkosti obrázka

    plot_acf(merged_df['price'], lags=24, ax=ax)

    if market_type == "IDM":
        ax.set_title(f'ACF graf pre ceny vnútrodenného trhu s 60 minútovou periódou - granularita 1 hodina',
                     fontsize=12)
    elif market_type == "DAM":
        ax.set_title(f'ACF graf pre ceny denného trhu - granularita 1 hodina', fontsize=12)
    elif market_type == "IDM15":
        ax.set_title(f'ACF graf pre ceny vnútrodenného trhu s 15 minútovou periódou - granularita 1 hodina',
                     fontsize=12)

    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    plt.savefig("Graphs/ACF_from_to", bbox_inches='tight')  # bbox_inches='tight' zabezpečí, že sa uloží celý obrázok
    plt.show()


def PACF(market_type, date_from, date_to):
    if market_type == "IDM15":
        merged_df = merging_data_and_preparation_IDM15(market_type, date_from, date_to)
    else:
        merged_df = merging_data_and_preparation_dam_idm(market_type, date_from, date_to)

    fig, ax = plt.subplots(figsize=(10, 6))  # Definícia veľkosti obrázka

    plot_pacf(merged_df['price'], lags=24, ax=ax)  # Použitie ax=ax pre definíciu osi

    if market_type == "IDM":
        ax.set_title(f'PACF graf pre ceny vnútrodenného trhu s 60 minútovou periódou - granularita 1 hodina',
                     fontsize=12)
    elif market_type == "DAM":
        ax.set_title(f'PACF graf pre ceny denného trhu - granularita 1 hodina', fontsize=12)
    elif market_type == "IDM15":
        ax.set_title(f'PACF graf pre ceny vnútrodenného trhu s 15 minútovou periódou - granularita 1 hodina',
                     fontsize=12)

    ax.set_xlabel('Lag')
    ax.set_ylabel('PACF')
    plt.savefig("Graphs/PACF_from_to")
    plt.show()


def qq_plot(market_type, date_from, date_to):
    if market_type == "IDM15":
        merged_df = merging_data_and_preparation_IDM15(market_type, date_from, date_to)
    else:
        merged_df = merging_data_and_preparation_dam_idm(market_type, date_from, date_to)

    # Vytvorenie Q-Q grafu
    plt.figure(figsize=(10, 6))
    stats.probplot(merged_df['price'], dist="norm", plot=plt)
    if (market_type == "IDM"):
        plt.title(f'Q-Q graf pre ceny vnutrodenného trhu s 60 minútovou periódou - granularita 1 hodina', fontsize=12)
    if (market_type == "DAM"):
        plt.title(f'Q-Q graf pre ceny denného trhu - granularita 1 hodina', fontsize=12)
    if (market_type == "IDM15"):
        plt.title(f'Q-Q graf pre ceny vnutrodenného trhu s 15 minútovou periódou - granularita 1 hodina', fontsize=12)

    plt.xlabel('Teoretické kvantily')
    plt.ylabel('Usporiadané hodnoty')
    plt.grid(True)
    plt.savefig("Graphs/QQ_plot_from_to")
    plt.show()


