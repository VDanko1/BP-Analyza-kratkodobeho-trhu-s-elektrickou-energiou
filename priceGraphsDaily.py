import pickle
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def visualize_idm15_with_lines():
    with open("Data/IDM15_results_2024-JAN-APR.pkl", "rb") as file_dam:
        data_idm = pickle.load(file_dam)

    # Vytvorte DataFrame z vašich dát
    df_dam = pd.DataFrame(data_idm)

    df_dam['deliveryEnd'] = pd.to_datetime(df_dam['deliveryEnd'])

    for df in [df_dam]:
        df.rename(columns={'priceWeightedAverage': 'price'}, inplace=True)


    df_dam = df_dam[['deliveryEnd', 'price', 'period']]

    selected_day = '2024-03-16'
    df_selected_day = df_dam[df_dam['deliveryEnd'].dt.date == pd.to_datetime(selected_day).date()]

    #print(df_selected_day.tail(50))

    plt.figure(figsize=(12, 6))
    plt.plot(df_selected_day['deliveryEnd'], df_selected_day['price'], label='Cena')

    # Vykreslenie rovnej čiary pre každú 4. periodu
    for index, row in df_selected_day.iterrows():
        if row['period'] % 4 == 0:
            plt.axvline(x=row['deliveryEnd'], color='red', linestyle='--', linewidth=0.5)  # Vertikálna čiara

    plt.xlabel('Dátum', fontsize=14)  # Zväčšenie písma osi x
    plt.ylabel('Cena €/MWh', fontsize=14)  # Zväčšenie písma osi y
    plt.title(f'Ceny z VDT15 pre 16.3.2024 - granularita 1 hodina', fontsize=16)  # Zväčšenie písma titulu
    plt.legend(fontsize=14)
    plt.show()


def visualize_av_price_in_day(filename):
    try:
        # Load the pickle file
        with open(filename, "rb") as file:
            data = pickle.load(file)

        # Convert data to a Pandas DataFrame
        df = pd.DataFrame(data)

        # Convert 'deliveryDay' and 'deliveryStart' columns to datetime type
        prices = [entry['priceWeightedAverage'] for entry in data]

        # Set 'deliveryStart' as the index for time series plotting
        df.set_index('deliveryStart', inplace=True)

        # Create the plot
        plt.figure(figsize=(20, 6))
        plt.plot(prices, linestyle='-', marker=".", color='b', label='5.12.2023')

        plt.title('Price Trend from 4.12.2023 up to 11.12.2023')
        plt.xlabel('Date')
        plt.ylabel('Average price €/MWh')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot
        plt.savefig("continuous_prices.png")

        # Show the plot
        plt.show()

    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
    except Exception as e:
        print(f"Error: {e}")


def visualize_av_price_in_day_overlay(filename, filename1, filename2, filename3, filename4, filename5,filename6):
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

        with open(filename4, "rb") as file_2023:
            data4 = pickle.load(file_2023)

        with open(filename5, "rb") as file_2023:
            data5 = pickle.load(file_2023)

        with open(filename6, "rb") as file_2023:
            data6 = pickle.load(file_2023)


        # Extract relevant information for plotting
        dates = [entry['deliveryDay'] for entry in data]
        prices = [entry['priceWeightedAverage'] for entry in data]

        dates_1 = [entry['deliveryDay'] for entry in data1]
        prices_1 = [entry['priceWeightedAverage'] for entry in data1]

        dates_2 = [entry['deliveryDay'] for entry in data2]
        prices_2 = [entry['priceWeightedAverage'] for entry in data2]

        dates_3 = [entry['deliveryDay'] for entry in data3]
        prices_3 = [entry['priceWeightedAverage'] for entry in data3]
        prices_4 = [entry['priceWeightedAverage'] for entry in data4]
        prices_5 = [entry['priceWeightedAverage'] for entry in data5]
        prices_6 = [entry['priceWeightedAverage'] for entry in data6]

        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(prices, linestyle='-', color='b', label='5.12.2023')
        plt.plot(prices_1, linestyle='-',  color='r', label='6.12.2023')
        plt.plot(prices_2, linestyle='-',  color='g', label='7.12.2023')
        plt.plot(prices_3, linestyle='-',  color='y', label='8.12.2023')
        plt.plot(prices_4, linestyle='-',  color='c', label='9.12.2023')
        plt.plot(prices_5, linestyle='-', color='#FF5733', label='10.12.2023')
        plt.plot(prices_6, linestyle='-', color='purple', label='11.12.2023')

        plt.title('Vývoj cien na dennom trhu od 5.12.2023 do 11.12.2023 - granularita 1 hodina',fontsize=16)
        plt.xlabel('Perióda dňa',fontsize=14)
        plt.ylabel('Cena €/MWh',fontsize=14)
        plt.xticks()
        plt.legend()
        #plt.grid(alpha=0.3, zorder=-1)
        plt.tight_layout()
        plt.savefig("Graphs/Tyzdnovy graf overlay.png")
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error: {e}")


visualize_idm15_with_lines()
