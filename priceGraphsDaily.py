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


