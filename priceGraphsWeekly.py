import pickle
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import datetime

def visualize_av_price_in_week(filename):
    try:
        # Load the pickle file
        with open(filename, "rb") as file:
            data = pickle.load(file)

        # Convert data to a Pandas DataFrame
        df = pd.DataFrame(data)

        # Convert 'deliveryDay' and 'deliveryStart' columns to datetime type
        df['deliveryDay'] = pd.to_datetime(df['deliveryDay'])
        df['deliveryStart'] = pd.to_datetime(df['deliveryStart'])

        # Set 'deliveryStart' as the index for time series plotting
        df.set_index('deliveryStart', inplace=True)

        # Create the plot
        plt.figure(figsize=(20, 6))
        plt.plot(df.index, df['priceWeightedAverage'], linestyle='-', marker="*", color='b')

        plt.title('Price Trend from 4.12.2023 up to 11.12.2023')
        plt.xlabel('Date')
        plt.ylabel('Average price €/MWh')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Add legend with date and day of the week
        legend_labels = [f"{date.date()} - {date.strftime('%A')}" for date in df.index]
        plt.legend(legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        # Save the plot
        plt.savefig("continuous_prices.png")

        # Show the plot
        plt.show()

    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
    except Exception as e:
        print(f"Error: {e}")

def visualize_av_price_in_week_overlay(filename, filename1, filename2, filename3):
    try:
        # Load the pickle files
        with open(filename, "rb") as file_2022:
            data = pickle.load(file_2022)

        with open(filename1, "rb") as file_2023:
            data1 = pickle.load(file_2023)

        with open(filename2, "rb") as file_2022:
            data2 = pickle.load(file_2022)

        with open(filename3, "rb") as file_2023:
            data3 = pickle.load(file_2023)


        # Extract relevant information for plotting
        dates = [entry['deliveryDay'] for entry in data]
        prices = [entry['priceWeightedAverage'] for entry in data]

        dates_1 = [entry['deliveryDay'] for entry in data1]
        prices_1 = [entry['priceWeightedAverage'] for entry in data1]

        dates_2 = [entry['deliveryDay'] for entry in data2]
        prices_2 = [entry['priceWeightedAverage'] for entry in data2]

        dates_3 = [entry['deliveryDay'] for entry in data3]
        prices_3 = [entry['priceWeightedAverage'] for entry in data3]

        # Convert 'deliveryDay' to datetime objects if needed

        # Create the plot
        plt.figure(figsize=(20, 6))
        plt.plot(prices, linestyle='-', marker="*", color='b', label='Week 1')
        plt.plot(prices_1, linestyle='-', marker="*", color='r', label='Week 2')
        plt.plot(prices_2, linestyle='-', marker="*", color='g', label='Week 3')
        plt.plot(prices_3, linestyle='-', marker="*", color='y', label='Week 4')

        plt.title('Weekly price trends between 4.12 up to 31.12')
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

# Example usage with four filenames
visualize_av_price_in_week_overlay(
    "Data/IDM_results_2023_04_12-10_12.pkl",
    "Data/IDM_results_2023_11_12-17_12.pkl",
    "Data/IDM_results_2023_18_12-24_12.pkl",
    "Data/IDM_results_2023_25_12-31_12.pkl"
)

file_list = [
    "IDM_results_2023_04_12-10_12.pkl",
    "IDM_results_2023_11_12-17_12.pkl",
    "IDM_results_2023_18_12-24_12.pkl",
    "IDM_results_2023_25_12-31_12.pkl"
]