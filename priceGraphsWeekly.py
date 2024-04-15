import pickle
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def visualize_av_price_in_week(filename):
    try:
        # Load the pickle file
        with open(filename, "rb") as file:
            data = pickle.load(file)

        #columns_to_read = ['Date', 'Start', 'Poland', 'Austria', 'France', 'Denmark 1', 'Netherlands', 'Hungary',
        #                   'Czech Republic', 'Germany/Luxembourg', ]


        columns_to_read = ['Date', 'Start', 'Austria', 'Czech Republic', 'Germany/Luxembourg', 'Hungary']
        data_foreign_market = pd.read_csv('Data/Day-Ahead_GERLUX_AUS_POL_CZ_HU.csv', sep=';', usecols=columns_to_read)
        df_slovak_market = pd.DataFrame(data)

        df_slovak_market['deliveryStart'] = pd.to_datetime(df_slovak_market['deliveryStart'])
        data_foreign_market['Date'] = pd.to_datetime(data_foreign_market['Date'])
        data_foreign_market['Start'] = pd.to_datetime(data_foreign_market['Start'], format="%I:%M %p").dt.strftime(
            '%H:%M:%S')

        data_foreign_market['Date'] = data_foreign_market['Date'].astype(str) + ' ' + data_foreign_market[
            'Start'].astype(str)
        data_foreign_market['Date'] = pd.to_datetime(data_foreign_market['Date'], format="%Y-%m-%d %H:%M:%S")
        data_foreign_market['Date'] = data_foreign_market['Date'] - pd.Timedelta(hours=1)
        """
        correlationCzech = np.corrcoef(df_slovak_market['price'], data_foreign_market['Czech Republic'])[0, 1]
        correlationAustria = np.corrcoef(df_slovak_market['price'], data_foreign_market['Austria'])[0, 1]
        correlationGermany = np.corrcoef(df_slovak_market['price'], data_foreign_market['Germany/Luxembourg'])[0, 1]
        correlationHungary = np.corrcoef(df_slovak_market['price'], data_foreign_market['Hungary'])[0, 1]
        correlationPoland = np.corrcoef(df_slovak_market['price'], data_foreign_market['Poland'])[0, 1]
        correlationFrance = np.corrcoef(df_slovak_market['price'], data_foreign_market['France'])[0, 1]
        correlationNetherlands = np.corrcoef(df_slovak_market['price'], data_foreign_market['Netherlands'])[0, 1]
        correlationDenmark = np.corrcoef(df_slovak_market['price'], data_foreign_market['Denmark 1'])[0, 1]
       
        print("Czech Republic: %.2f" % correlationCzech)
        print("Austria: %.2f" % correlationAustria)
        print("Germany: %.2f" % correlationGermany)
        print("Hungary: %.2f" % correlationHungary)
        print("Poland: %.2f" % correlationPoland)
        print("Netherlands: %.2f" % correlationNetherlands)
        print("France: %.2f" % correlationFrance)
        print("Denmark: %.2f" % correlationDenmark)
        """

        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(df_slovak_market['deliveryStart'], df_slovak_market['price'], linestyle='-', color='b',
                 label='Slovensko')
        plt.plot(data_foreign_market['Date'], data_foreign_market['Austria'], linestyle='-', color='r', label='Rakúsko')
        plt.plot(data_foreign_market['Date'], data_foreign_market['Czech Republic'], linestyle='-', color='y',
                 label='Česká republika')
        plt.plot(data_foreign_market['Date'], data_foreign_market['Hungary'], linestyle='-', color='g', label='Maďarsko')
        plt.plot(data_foreign_market['Date'], data_foreign_market['Germany/Luxembourg'], linestyle='-',
              color='k', label='Nemecko/Luxembursko')

        plt.title('Porovnanie denných trhov s okolitými štátmi - granularita 1 hodina', fontsize = 16)
        plt.xlabel('Dátum')
        plt.ylabel('Cena €/MWh')
        plt.xticks(rotation=10)
        plt.tight_layout()
        plt.legend(loc='upper center')
        plt.savefig("Graphs/SADC price comparasions (7.7 - 17.7.2023)- CZ,GER,AU,SK,HU.jpg" )
        plt.show()


    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
    except Exception as e:
        print(f"Error: {e}")

def visualize_idm_dam():
    with open('Data/DAM_results_2023DEC.pkl', "rb") as file_2023:
        data_dam_2023 = pickle.load(file_2023)

    with open('Data/IDM_results_december_2023.pkl', "rb") as file_2023:
        data_idm_2023 = pickle.load(file_2023)

    with open('Data/IDM_results_december_15min.pkl', "rb") as file_2023:
        data_idm_2023_15min = pickle.load(file_2023)

    df_dam = pd.DataFrame(data_dam_2023)
    df_idm = pd.DataFrame(data_idm_2023)
    df_idm_15 = pd.DataFrame(data_idm_2023_15min)

    df_idm_15.dropna()

    df_dam['deliveryStart'] = pd.to_datetime(df_dam['deliveryStart']).dt.tz_localize(None)
    df_idm['deliveryStart'] = pd.to_datetime(df_idm['deliveryStart']).dt.tz_localize(None)
    df_idm_15['deliveryStart'] = pd.to_datetime(df_idm_15['deliveryStart']).dt.tz_localize(None)

    #df_dam_december = df_dam[(df_dam['deliveryStart'].dt.month == 12) & (df_dam['deliveryStart'].dt.year == 2023)]
    #df_idm_december = df_idm[(df_idm['deliveryStart'].dt.month == 12) & (df_idm['deliveryStart'].dt.year == 2023)]

    plt.figure(figsize=(12, 6))

    plt.plot(df_idm['deliveryStart'], df_idm['priceWeightedAverage'], linestyle='-', color='blue',
             label="Vnútrodenný trh")

    plt.plot(df_dam['deliveryStart'], df_dam['price'], linestyle='-', color='r',
             label="Denný trh")

    #plt.plot(df_idm_15['deliveryStart'], df_idm_15['priceAverage'], linestyle='-', color='g',alpha=0.2,
    #         label="IDM (15 min) electricity price €/MWh")


    plt.xticks(rotation=30)  # Adjust the rotation if needed
    plt.title("Porovnanie cien na dennom a vnútrodennom trhu za december 2023  - granularita 1 hodina", fontsize=16)
    plt.xlabel("Dátum", fontsize=12)
    plt.ylabel("Cena €/MWh", fontsize=12)
    plt.legend(fontsize='large')
    plt.tight_layout()
    #plt.savefig("DAM a IDM (60,15 min) porovnanie december 2023")
    # Show the plot
    plt.show()


def visualize_av_price_in_week_overlay():
    try:
        with open('Data/DAM_results_2023.pkl', "rb") as file_2023:
            data_dam_2023 = pickle.load(file_2023)

        with open('Data/DAM_results_2020.pkl', "rb") as file_2020:
            data_dam_2020 = pickle.load(file_2020)

        with open('Data/DAM_results_2021.pkl', "rb") as file_2021:
            data_dam_2021 = pickle.load(file_2021)

        with open('Data/DAM_results_2022.pkl', "rb") as file_2022:
            data_dam_2022 = pickle.load(file_2022)

        columns_to_read = ['Date', 'Price']
        data_plyn = pd.read_csv('Data/Plyn-2020-2024.csv', sep=',', usecols=columns_to_read)
        data_ropa = pd.read_csv('Data/Ropa-2020-2024.csv', sep=',', usecols=columns_to_read)
        data_uhlie = pd.read_csv('Data/Uhlie-2020-2024.csv', sep=',', usecols=columns_to_read)

        data_2022_plyn = data_plyn[data_plyn['Date'].str.contains('2022')]
        data_2022_ropa = data_ropa[data_ropa['Date'].str.contains('2022')]
        data_2022_uhlie = data_uhlie[data_uhlie['Date'].str.contains('2022')]

        # Prekonvertujte stĺpec 'Date' na formát datetime, ak nie je
        data_2022_plyn['Date'] = pd.to_datetime(data_2022_plyn['Date'])
        data_2022_ropa['Date'] = pd.to_datetime(data_2022_ropa['Date'])
        data_2022_uhlie['Date'] = pd.to_datetime(data_2022_uhlie['Date'])

        # Rozdelenie dát z roku 2022 na prvé a druhé polrok
        data_2022_plyn_prvy_polrok = data_2022_plyn[data_2022_plyn['Date'].dt.month < 6]
        data_2022_plyn_druhy_polrok = data_2022_plyn[data_2022_plyn['Date'].dt.month >= 6]

        data_2022_ropa_prvy_polrok = data_2022_ropa[data_2022_ropa['Date'].dt.month < 6]
        data_2022_ropa_druhy_polrok = data_2022_ropa[data_2022_ropa['Date'].dt.month >= 6]

        data_2022_uhlie_prvy_polrok = data_2022_uhlie[data_2022_uhlie['Date'].dt.month < 6]
        data_2022_uhlie_druhy_polrok = data_2022_uhlie[data_2022_uhlie['Date'].dt.month >= 6]

        data_ropa['Date'] = pd.to_datetime(data_ropa['Date'], format='%m/%d/%Y')
        data_plyn['Date'] = pd.to_datetime(data_plyn['Date'], format='%m/%d/%Y')
        data_uhlie['Date'] = pd.to_datetime(data_uhlie['Date'], format='%m/%d/%Y')


        date_2023 = [entry['deliveryDay'] for entry in data_dam_2023]
        price_2023 = [entry['price'] for entry in data_dam_2023]

        date_2020 = [entry['deliveryDay'] for entry in data_dam_2020]
        price_2020 = [entry['price'] for entry in data_dam_2020]

        date_2021 = [entry['deliveryDay'] for entry in data_dam_2021]
        price_2021 = [entry['price'] for entry in data_dam_2021]

        date_2022 = [entry['deliveryDay'] for entry in data_dam_2022]
        price_2022 = [entry['price'] for entry in data_dam_2022]

        dates_2023 = [datetime.strptime(date, '%Y-%m-%d') for date in date_2023]
        dates_2020 = [datetime.strptime(date, '%Y-%m-%d') for date in date_2020]
        dates_2021 = [datetime.strptime(date, '%Y-%m-%d') for date in date_2021]
        dates_2022 = [datetime.strptime(date, '%Y-%m-%d') for date in date_2022]

        df = pd.DataFrame(data_dam_2023)

        df_2022 = pd.DataFrame({'Date': dates_2022, 'Price': price_2022})

        # Rozdelenie dát na prvé a druhé polrok
        df_2022_prvy_polrok = df_2022[df_2022['Date'].dt.month < 6]
        df_2022_druhy_polrok = df_2022[df_2022['Date'].dt.month >= 6]


        print(df_2022_druhy_polrok.describe())
        # Create the plot
        plt.figure(figsize=(20, 6))
        plt.plot(data_2022_ropa_druhy_polrok['Date'], data_2022_ropa_druhy_polrok['Price'], linestyle='-', color='r',label="Brent Oil price per barrel")
        plt.plot(data_2022_plyn_druhy_polrok['Date'], data_2022_plyn_druhy_polrok['Price'], linestyle='-', color='g',label="TTF Natural Gas price per cubic metre")
        plt.plot(data_2022_uhlie_druhy_polrok['Date'], data_2022_uhlie_druhy_polrok['Price'] ,linestyle='-', color='orange',label="Newcastle Coal Price per tonne")
        #plt.plot(dates_2023, price_2023, linestyle='-', color='blue',alpha=0.3)
        #plt.plot(dates_2020, price_2020, linestyle='-', color='blue',alpha=0.3)
        #plt.plot(dates_2021, price_2021, linestyle='-', color='blue',alpha=0.3)
        plt.plot(df_2022_druhy_polrok['Date'], df_2022_druhy_polrok['Price'], linestyle='-', color='blue',label="DAM Electricity price €/MWh",alpha=0.3)

        average_prices_per_day = df.groupby('deliveryDay')['price'].mean()
        interpolated_price = np.interp(np.linspace(0, 1, 365), np.linspace(0, 1, len(data_plyn)),
                                       data_plyn['Price'])

        # Create a new DataFrame with the interpolated 'Price'
        interpolated_data_plyn = pd.DataFrame(
            {'Date': pd.date_range(start=data_plyn['Date'].min(), periods=365), 'Price': interpolated_price})

        # Now, calculate the correlation
        correlationPoland = np.corrcoef(average_prices_per_day, interpolated_data_plyn['Price'])[0, 1]

        #print(correlationPoland)

        plt.xticks(rotation=45, ha='right')  # Adjust the rotation for better readability
        plt.xticks()
        plt.title("DAM Slovak market prices and commodities Q3 and Q4 of 2022 - granularity 1 day", fontsize=16)
        plt.xlabel("Date", fontsize = 12)
        plt.ylabel("Price of commodity in €", fontsize = 12)
        plt.legend()
        plt.tight_layout()
        plt.savefig("DAM a komodity 2022 Q3 Q4 - final")
        # Show the plot
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error: {e}")


# Example usage with four filenames
#visualize_av_price_in_week('Data/DAM_results_2024-02-07_2024-02-17.pkl')
#visualize_av_price_in_week_overlay()
visualize_idm_dam()
#visualize_av_price_in_week()
