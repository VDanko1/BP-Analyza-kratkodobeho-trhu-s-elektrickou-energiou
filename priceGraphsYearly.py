import pickle
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


def load_and_store_data_okte():
    api_url = "https://isot.okte.sk/api/v1/idm/results?deliveryDayFrom=2023-12-11&deliveryDayTo=2023-12-11&productType=60"

    response = requests.get(api_url)

    if response.status_code == 200:
        filename = "Data/IDM_results_2023-12-11.pkl"
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


def visualize_av_price_in_year(filename):
    try:
        # Load the pickle file
        with open(filename, "rb") as file:
            data = pickle.load(file)

        # Extract relevant information for plotting
        dates = [entry['deliveryDay'] for entry in data]
        prices = [entry['priceWeightedAverage'] for entry in data]

        # Convert string dates to datetime objects
        dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

        # Create the plot
        plt.figure(figsize=(20, 6))
        plt.plot(dates, prices, linestyle='-', color='b')
        plt.title('Price Trend from API Data')
        plt.xlabel('Date')
        plt.ylabel('Average price €/MWh')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Show the plot
        plt.show()
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
    except Exception as e:
        print(f"Error: {e}")


def visualize_av_prices_two_years(filename_2022, filename_2023):
    try:
        # Load the pickle files
        with open(filename_2022, "rb") as file_2022:
            data_2022 = pickle.load(file_2022)

        with open(filename_2023, "rb") as file_2023:
            data_2023 = pickle.load(file_2023)

        # Extract relevant information for plotting
        dates_2022 = [entry['deliveryDay'] for entry in data_2022]
        prices_2022 = [entry['priceWeightedAverage'] for entry in data_2022]

        dates_2023 = [entry['deliveryDay'] for entry in data_2023]
        prices_2023 = [entry['priceWeightedAverage'] for entry in data_2023]

        # Convert string dates to datetime objects
        dates_2022 = [datetime.strptime(date, '%Y-%m-%d') for date in dates_2022]
        dates_2023 = [datetime.strptime(date, '%Y-%m-%d') for date in dates_2023]

        # Create the plot with two lines
        plt.figure(figsize=(20, 6))
        plt.plot(dates_2022, prices_2022, linestyle='-', color='b', label='2022 Prices')
        plt.plot(dates_2023, prices_2023, linestyle='-', color='r', label='2023 Prices')
        plt.title('Price Trend from API Data')
        plt.xlabel('Date')
        plt.ylabel('Average price €/MWh')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        # Show the plot
        plt.show()
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error: {e}")


def visualize_av_prices_not_overlay(filename_2022, filename_2023):
    try:
        # Load the pickle files
        with open(filename_2022, "rb") as file_2022:
            data_2022 = pickle.load(file_2022)

        with open(filename_2023, "rb") as file_2023:
            data_2023 = pickle.load(file_2023)

        # Extract relevant information for plotting
        dates_2022 = [entry['deliveryDay'] for entry in data_2022]
        prices_2022 = [entry['priceWeightedAverage'] for entry in data_2022]

        dates_2023 = [entry['deliveryDay'] for entry in data_2023]
        prices_2023 = [entry['priceWeightedAverage'] for entry in data_2023]

        # Convert string dates to datetime objects
        dates_2022 = [datetime.strptime(date, '%Y-%m-%d') for date in dates_2022]
        dates_2023 = [datetime.strptime(date, '%Y-%m-%d') for date in dates_2023]

        # Create the plot with two lines
        plt.figure(figsize=(20, 6))
        plt.plot(dates_2022, prices_2022, linestyle='-', color='b', label='2022 Prices')
        plt.plot(dates_2023, prices_2023, linestyle='-', color='r', label='2023 Prices')
        plt.title('Price Trend from API Data')
        plt.xlabel('Date')
        plt.ylabel('Average price €/MWh')
        plt.xticks(rotation=45)
        plt.legend()  # Add a legend to differentiate between the lines
        plt.tight_layout()
        plt.savefig()

        # Show the plot
        plt.show()
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error: {e}")


from collections import defaultdict


def visualize_av_prices_overlay(filename_2022, filename_2023):
    try:
        # Load the pickle files
        with open(filename_2022, "rb") as file_2022:
            data = pickle.load(file_2022)

        with open(filename_2023, "rb") as file_2023:
            data_1 = pickle.load(file_2023)

        # Extract relevant information for plotting
        dates_2022 = [entry['deliveryDay'] for entry in data]
        price = [entry['priceWeightedAverage'] for entry in data]

        dates_2023 = [entry['deliveryDay'] for entry in data_1]
        price1 = [entry['price'] for entry in data_1]

        # Create the plot with two lines
        plt.figure(figsize=(20, 6))
        plt.plot(price, linestyle='-', color='g', label='Prices 2022')
        plt.plot(price1, linestyle='-', color='r', label='Prices 2023')
        plt.title('Price Trend from API Data')
        plt.xlabel('Months in a year')
        plt.ylabel('Average price €/MWh')
        plt.xticks(np.arange(0, len(price) + 1, len(price) // 12), range(0, 13))
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
#load_and_store_data_oil()
#print_data()
# visualize_av_prices_overlay("IDM_results_2022.pkl", "IDM_results_2023.pkl")
# visualize_av_prices_not_overlay("IDM_results_2022.pkl", "IDM_results_2023.pkl")
# visualize_av_price_in_year("IDM_results_2020.pkl")
