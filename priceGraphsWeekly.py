import pickle
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import datetime

"""
api_url = "https://isot.okte.sk/api/v1/idm/results?deliveryDayFrom=2023-12-04&deliveryDayTo=2023-12-10&productType=60"

response = requests.get(api_url)

if response.status_code == 200:
    filename = "IDM_results_2023_04_12-10_12.pkl"
    with open(filename, "wb") as file:
        pickle.dump(response.json(), file)
else:
    print(f"Error: {response.status_code}")
"""

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
        plt.plot(df.index, df['priceWeightedAverage'], linestyle='-',marker="*", color='b')

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


visualize_av_price_in_week("Data/IDM_results_2023_04_12-10_12.pkl")