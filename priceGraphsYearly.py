import pickle
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


def load_and_store_data_okte():
    api_url = "https://isot.okte.sk/api/v1/dam/results?deliveryDayFrom=2024-01-01&deliveryDayTo=2024-04-01"

    response = requests.get(api_url)

    if response.status_code == 200:
        filename = "Data/DAM_results_2024-JAN-APR.pkl"
        with open(filename, "wb") as file:
            pickle.dump(response.json(), file)
    else:
        print(f"Error: {response.status_code}")

def load_and_store_data_borrowed():
    api_url = "https://markets.tradingeconomics.com/chart/eecxm:ind?span=1m&securify=new&url=/commodity/carbon&AUTH=K3G0OIIGcJ2ojK4wqITEnYx5jnqDefwNGP54u9Ty11T76niQsWqDvjhrbV%2Bmk0S0&ohlc=0"

    response = requests.get(api_url)

    if response.status_code == 200:
        filename = "Data/EU_povolenky_jan22_feb16"
        with open(filename, "wb") as file:
            pickle.dump(response.json(), file)
    else:
        print(f"Error: {response.status_code}")

def load_and_store_data_oil():
    from_date_str = "2023-01-01"
    to_date_str = "2023-12-31"

    api_url = f"https://api.oilpriceapi.com/v1/prices?past_week"#[from]={from_date_str}&by_period[to]={to_date_str}"

    headers = {
        'Authorization': 'Token c22a61b902e22643052cb26ad2b7c413',
        'Content-Type': 'application/json'
    }


    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        filename = "Data/Oil_price_past_year"
        with open(filename, "wb") as file:
            pickle.dump(response.json(), file)
            print(response.status_code)

    else:
        print(f"Error: {response.status_code}")

def print_data ():
    import pickle

    # Load the pickled data
    filename = "Data/Oil_price_2021"
    with open(filename, "rb") as file:
        oil_price_data = pickle.load(file)

    # Now you can work with the 'oil_price_data' variable, which contains the loaded data
    print(oil_price_data)


def visualize_av_prices_overlay():
    try:
        # Load the pickle files
        with open('Data/DAM_results_1.pkl', "rb") as file_2022:
            data_dam1 = pickle.load(file_2022)

        with open('Data/DAM_results_2.pkl', "rb") as file_2022:
            data_dam2 = pickle.load(file_2022)

        with open('Data/DAM_results_3.pkl', "rb") as file_2022:
            data_dam3 = pickle.load(file_2022)

        with open('Data/DAM_results_4.pkl', "rb") as file_2022:
            data_dam4 = pickle.load(file_2022)

        with open('Data/DAM_results_2023WD.pkl', "rb") as file_2022:
            data_idm = pickle.load(file_2022)

        with open('Data/DAM_results_2023DEC.pkl', "rb") as file_2022:
            data_dec = pickle.load(file_2022)

        df_dam = pd.DataFrame(data_dam1)
        df_dam2 = pd.DataFrame(data_dam2)
        df_dam3 = pd.DataFrame(data_dam3)
        df_dam4 = pd.DataFrame(data_dam4)
        df_damWD = pd.DataFrame(data_idm)
        df_damdec = pd.DataFrame(data_dec)

        description_dam = df_dam['price'].describe()
        description_dam2 = df_dam2['price'].describe()
        description_dam3 = df_dam3['price'].describe()
        description_dam4 = df_dam4['price'].describe()
        description_damWD = df_damWD['price'].describe()
        description_damdec = df_damdec['price'].describe()

        print("Description for df_dam:")
        print(description_dam)

        print("\nDescription for df_dam2:")
        print(description_dam2)

        print("\nDescription for df_dam3:")
        print(description_dam3)

        print("\nDescription for df_dam4:")
        print(description_dam4)

        print("\nDescription for df_damWD:")
        print(description_damWD)

        print("\nDescription for df_damDEC:")
        print(description_damdec)

        df_idm = pd.DataFrame(data_idm)

        # Extract relevant information for plotting
        date_dam = [entry['deliveryDay'] for entry in data_dam1]
        price_dam = [entry['price'] for entry in data_dam1]
        price_dam2 = [entry['price'] for entry in data_dam2]
        price_dam3 = [entry['price'] for entry in data_dam3]
        price_dam4 = [entry['price'] for entry in data_dam4]


        #dates_dam = [datetime.strptime(date, '%Y-%m-%d') for date in date_idm]
        dates_idm = [datetime.strptime(date, '%Y-%m-%d') for date in date_dam]

        # Create the plot with two lines
        plt.figure(figsize=(20, 6))
        plt.plot(price_dam, linestyle='-', color='g', label='DAM prices 1 week of december 2023')
        plt.plot(price_dam2, linestyle='-', color='r', label='DAM prices 2 week of december 2023')
        plt.plot(price_dam3, linestyle='-', color='b', label='DAM prices of Christmas week of december 2023')
        plt.plot(price_dam4, linestyle='-', color='olive', label='DAM prices 4 week of december 2023')

        plt.title('Weekly prices of december 2023 on DAM market - granularity 1 hour', fontsize = 16)
        plt.xlabel('Period', fontsize = 16)
        plt.ylabel('Price €/MWh', fontsize = 16)
        plt.legend()
        plt.savefig("Vyvoj cien DAM za december 2023 (tyzdne) - final")
        plt.tight_layout()

        # Show the plot
        plt.show()
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error: {e}")


# visualize_av_prices_two_years("IDM_results_2020.pkl", "IDM_results_2021.pkl")
#visualize_av_prices_overlay("Data/IDM_results_2021.pkl", "Data/Oil_price_2021")
load_and_store_data_okte()
#visualize_av_prices_overlay()
#load_and_store_data_borrowed()
#load_and_store_data_oil()
#print_data()
#visualize_av_prices_overlay()
#visualize_av_prices_not_overlay("Data/IDM_results_2022.pkl", "Data/IDM_results_2023.pkl")
# visualize_av_price_in_year("IDM_results_2020.pkl")
