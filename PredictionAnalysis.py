import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
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


def AdFullerTestDAM():
    with open("Data/DAM_results_2023.pkl", "rb") as file_dam:
        data_dam = pickle.load(file_dam)

    # Vytvorte DataFrame z vašich dát
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

    # Interpretace výsledků
    if bartlett_result.pvalue < 0.05:
        print(
            "P-hodnota je nižší než 0.05, což naznačuje, že existují statisticky významné rozdíly ve variabilitě mezi skupinami dat.")
        print("Nulová hypotéza o stejné variabilitě mezi skupinami dat je zamítnuta.")
    else:
        print(
            "P-hodnota je vyšší než 0.05, což naznačuje, že neexistují statisticky významné rozdíly ve variabilitě mezi skupinami dat.")
        print("Nulová hypotéza o stejné variabilitě mezi skupinami dat není zamítnuta.")


    # ADF test pre overenie stacionarity
    result = adfuller(price_series.dropna())  # Dropna na odstránenie prípadných chýbajúcich hodnôt

    # Výsledky testu
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

    # Interpretácia výsledkov
    if p_value <= 0.05:
        print('Null hypothesis (inability to reject the unit root) is rejected, the time series is stationary.')
    else:
        print('Null hypothesis (inability to reject the unit root) is not rejected, the time series is non-stationary.')


def Histrogram():
    with open("Data/DAM_results_2023.pkl", "rb") as file_dam:
        data_dam = pickle.load(file_dam)

    # Vytvorenie DataFrame z vašich dát
    df_dam = pd.DataFrame(data_dam)

    # Aplikácia sezónnej diferenciácie na stĺpec 'priceWeightedAverage'
    seasonal_diff_price = df_dam['price'].diff()

    # Odstránenie riadkov s chýbajúcimi hodnotami
    seasonal_diff_price = seasonal_diff_price.dropna()

    # Vykreslenie histogramu sezónnej diferencie
    plt.figure(figsize=(10, 6))
    plt.hist(seasonal_diff_price, bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram cien differencovane Slovenského denného trhu za rok 2023')
    plt.xlabel('Cena')
    plt.ylabel('Počet')
    plt.grid(True)
    plt.show()


def AdFullerTestIDM15():
    with open("Data/IDM_results_2023_15min.pkl", "rb") as file_dam:
        data_dam = pickle.load(file_dam)

    # Vytvorte DataFrame z vašich dát
    df_dam = pd.DataFrame(data_dam)

    price_series = df_dam['priceAverage']

    # ADF test pre overenie stacionarity
    result = adfuller(price_series.dropna())  # Dropna na odstránenie prípadných chýbajúcich hodnôt

    # Výsledky testu
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

    # Interpretácia výsledkov
    if p_value <= 0.05:
        print('Null hypothesis (inability to reject the unit root) is rejected, the time series is stationary.')
    else:
        print('Null hypothesis (inability to reject the unit root) is not rejected, the time series is non-stationary.')


def DecompositionOfTimeSeries():
    with open("Data/DAM_results_2023.pkl", "rb") as file_dam:
        data_dam = pickle.load(file_dam)

    # Vytvorte DataFrame z vašich dát
    df_dam = pd.DataFrame(data_dam)

    df_dam['diff_price'] = df_dam['price'].diff()

    price_series = df_dam['price']

    df_dam['deliveryEnd'] = pd.to_datetime(df_dam['deliveryEnd'])
    #df_dam.dropna()

    result = seasonal_decompose(price_series, model='additive')  # Perioda 24 pre dennú sezónnosť

    # Vykreslenie dekompozície
    plt.figure(figsize=(14, 9))

    plt.subplot(4, 1, 1)
    plt.plot(df_dam['deliveryEnd'], df_dam['price'], label='Original Series')
    plt.xlabel('Date')
    plt.xticks(rotation=10)
    plt.ylabel('Price')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(df_dam['deliveryEnd'], result.trend, label='Trend')
    plt.xlabel('Date')
    plt.xticks(rotation=10)
    plt.ylabel('Price')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(df_dam['deliveryEnd'], result.seasonal, label='Seasonal')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=10)
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(df_dam['deliveryEnd'], result.resid, label='Residuals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=10)
    plt.legend()

    plt.tight_layout()
    #plt.savefig("Graphs/Dekompozicia_DAM_2023_Additive")
    plt.show()

# Spustenie funkcie pre dekompozíciu
#DecompositionOfTimeSeries()

def STLDecomposition():
    with open("Data/DAM_results_2022.pkl", "rb") as file_dam:
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

    combined_years = pd.concat([df_dam2020,df_dam2021,df_dam2022,df_dam2023],ignore_index=False)
    combined_years.dropna()

    combined_years['diff_price'] = combined_years['price'].diff()

    # STl dekompozícia
    stl = STL(combined_years['diff_price'], seasonal=13)  # 13 je priemerná sezónna dĺžka v roku

    # Vykonanie dekompozície
    result = stl.fit()

    # Vykreslenie dekompozície
    plt.figure(figsize=(16, 12))

    plt.subplot(4, 1, 1)
    plt.plot(combined_years['price'], label='Original Series')
    plt.xlabel('Date')
    plt.xticks(rotation=10)  # Nastavenie rotácie textu na x-ovej osi
    plt.ylabel('Price')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(combined_years.index, result.trend, label='Trend')
    plt.xlabel('Date')
    plt.xticks(rotation=10)  # Nastavenie rotácie textu na x-ovej osi
    plt.ylabel('Price')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(combined_years.index, result.seasonal, label='Seasonal')
    plt.xlabel('Date')
    plt.xticks(rotation=10)
    plt.ylabel('Price')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(combined_years.index, result.resid, label='Residuals')
    plt.xlabel('Date')
    plt.xticks(rotation=10)
    plt.ylabel('Price')
    plt.legend()

    #plt.savefig("Graphs/Dekompozicia_IDM_2022_STL")
    plt.tight_layout()
    plt.show()


def ACF_PACF():
    with open("Data/IDM_results_2020.pkl", "rb") as file_dam:
        data_dam = pickle.load(file_dam)

    df_dam = pd.DataFrame(data_dam)

    df_dam['deliveryEnd'] = pd.to_datetime(df_dam['deliveryEnd'])

    df_dam = df_dam[['deliveryEnd', 'priceWeightedAverage']]

    # Remove missing values if any
    df_dam.dropna(inplace=True)

    df_dam['diff_price'] = df_dam['priceWeightedAverage'].diff()

    #plt.plot(df_dam['deliveryEnd'],df_dam['diff_price'])
    #plt.plot(df_dam['deliveryEnd'],df_dam['priceWeightedAverage'])
    #plt.figure(figsize=(20,6))

    # Plot ACF and PACF
    plot_acf(df_dam['priceWeightedAverage'], lags=24)
    plt.title('ACF for IDM prices of 2022- granularity 1 hour')
    plt.xlabel('Lag')
    plt.savefig("Graphs/ACF IDM 2020 lag 24")
    plt.ylabel('ACF')

    plot_pacf(df_dam['priceWeightedAverage'], lags=24)
    plt.title('PACF for IDM prices of 2022 - granularity 1 hour')
    plt.xlabel('Lag')
    plt.savefig("Graphs/PACF IDM 2020 lag 24")
    plt.ylabel('PACF')
    plt.show()


def qq_plot():
    with open("Data/IDM_results_2023_15min.pkl", "rb") as file_dam:
        data_dam = pickle.load(file_dam)

        # Vytvorenie DataFrame z vašich dát
    df_dam = pd.DataFrame(data_dam)

    df_dam.dropna(subset=['priceAverage'], inplace=True)

    prices = df_dam['priceAverage']

    # Vytvorenie Q-Q grafu
    plt.figure(figsize=(8, 6))
    stats.probplot(prices, dist="norm", plot=plt)
    plt.title('Q-Q Plot of IDM Prices with 15 minute period of year 2023 - granularity 1 hour')
    plt.xlabel('Theoretical quantiles')
    plt.ylabel('Ordered values')
    plt.grid(True)
    #plt.savefig("Graphs/QQ Plot IDM(15) 2023")
    plt.show()



# Zavolanie funkcie na vykreslenie Q-Q grafu
#Histrogram()
#qq_plot()
#ACF_PACF()
#STLDecomposition()
#AdFullerTestIDM15()
#AdFullerTestDAM()
DecompositionOfTimeSeries()