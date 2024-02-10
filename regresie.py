import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def korelacia():
    with open("Data/DAM_results_2023-12-11.pkl", "rb") as file_dam:
        data_dam = pickle.load(file_dam)

    with open("Data/IDM_results_2023-12-11.pkl", "rb") as file_idm:
        data_idm = pickle.load(file_idm)

    # Vytvorte DataFrame z vašich dát
    df_dam = pd.DataFrame(data_dam)
    df_idm = pd.DataFrame(data_idm)

    # Vyberte relevantné premenné z oboch DataFrame
    filtered_data = pd.merge(df_dam[['price']], df_idm[['priceWeightedAverage']], left_index=True, right_index=True,
                             suffixes=('_DAM', '_IDM')).dropna()

    filtered_data = filtered_data[
        (filtered_data['price'] <= 300) & (filtered_data['priceWeightedAverage'] <= 300)]

    correlation = filtered_data['price'].corr(filtered_data['priceWeightedAverage'])
    plt.figure(figsize=(20, 6))
    sns.scatterplot(x='price', y='priceWeightedAverage', data=filtered_data)

    x = filtered_data['price']
    y = filtered_data['priceWeightedAverage']
    fit = np.polyfit(x, y, deg=1)
    plt.plot(x, fit[0] * x + fit[1], color='red', linewidth=1, alpha=0.3)

    plt.title('Correlation graph between DAM price and IDM price - 11.12.2023 - granularity 1 hour')
    plt.xlabel('Price DAM €/Mwh')
    plt.ylabel('Price IDM €/MWh')
    plt.grid(True, alpha=0.5)
    plt.savefig("Graphs/Korelacia_DAM_IDM_2023-12-11")
    plt.show()
    print(f"Correlation between DAM price and IDM priceWeightedAverage: {correlation}")
    print(f"Linear Regression Equation: IDM_Price = {fit[0]:.4f} * DAM_Price + {fit[1]:.4f}")

def korelacia_denna():
    with open("Data/DAM_results_2023.pkl", "rb") as file_dam:
        data_dam = pickle.load(file_dam)

    with open("Data/IDM_results_2023.pkl", "rb") as file_idm:
        data_idm = pickle.load(file_idm)

    df_dam = pd.DataFrame(data_dam)
    df_idm = pd.DataFrame(data_idm)

    df_dam['deliveryStart'] = pd.to_datetime(df_dam['deliveryStart']).dt.tz_localize(None)
    df_idm['deliveryStart'] = pd.to_datetime(df_idm['deliveryStart']).dt.tz_localize(None)


    # For each period, create correlation graphs
    merged_data = pd.merge(df_dam[['price', 'deliveryStart']], df_idm[['priceWeightedAverage', 'deliveryStart']],
                           on='deliveryStart', suffixes=('_DAM', '_IDM')).dropna()

    merged_data['hour'] = merged_data['deliveryStart'].dt.hour

    # Vytvorte korelačné grafy pre každú hodinu
    for hour in range(24):
        hour_data = merged_data[merged_data['deliveryStart'].dt.hour == hour]

        correlation = hour_data['price'].corr(hour_data['priceWeightedAverage'])

        plt.figure(figsize=(20, 6))
        sns.scatterplot(x='price', y='priceWeightedAverage', data=hour_data)

        x = hour_data['price']
        y = hour_data['priceWeightedAverage']
        fit = np.polyfit(x, y, deg=1)

        plt.plot(x, fit[0] * x + fit[1], color='red', linewidth=1, alpha=0.3)
        hodina = hour
        plt.title(f'Correlation graph for DAM price and IDM priceWeightedAverage each period of year 2023 - period {hodina+1} - granularity 1 hour')
        plt.xlabel('Price DAM €/MWh')
        plt.ylabel('Average price IDM €/MWh')
        plt.grid(True, alpha=0.5)
        plt.savefig(f"Graphs/Korelacia_DAM_IDM_Hour_{hour}")
        plt.show()

        print(f"Correlation between DAM price and IDM priceWeightedAverage (Hour {hour}): {correlation}")

        print(f"Linear Regression Equation for Hour {hour}: IDM_Price = {fit[0]:.4f} * DAM_Price + {fit[1]:.4f}")


# Spustite funkciu pre vytvorenie korelácie a vykreslenie scatter plotu
korelacia_denna()
def korelacia_denna_vykresli():
    with open("Data/DAM_results_2023.pkl", "rb") as file_dam:
        data_dam = pickle.load(file_dam)

    with open("Data/IDM_results_2023.pkl", "rb") as file_idm:
        data_idm = pickle.load(file_idm)

    df_dam = pd.DataFrame(data_dam)
    df_idm = pd.DataFrame(data_idm)

    df_dam['deliveryStart'] = pd.to_datetime(df_dam['deliveryStart']).dt.tz_localize(None)
    df_idm['deliveryStart'] = pd.to_datetime(df_idm['deliveryStart']).dt.tz_localize(None)

    # Pre každú hodinu dňa spojte údaje DAM a IDM
    merged_data = pd.merge(df_dam[['price', 'deliveryStart']], df_idm[['priceWeightedAverage', 'deliveryStart']],
                           on='deliveryStart', suffixes=('_DAM', '_IDM')).dropna()

    print(merged_data.head(50))

#korelacia_denna_vykresli()
def korelacia_nedenna():
    with open("Data/DAM_results_2023-12-11.pkl", "rb") as file_dam:
        data_dam = pickle.load(file_dam)

    with open("Data/IDM_results_2023-12-11.pkl", "rb") as file_idm:
        data_idm = pickle.load(file_idm)

    df_dam = pd.DataFrame(data_dam)
    df_idm = pd.DataFrame(data_idm)

    df_dam['deliveryStart'] = pd.to_datetime(df_dam['deliveryStart']).dt.tz_localize(None)
    df_idm['deliveryStart'] = pd.to_datetime(df_idm['deliveryStart']).dt.tz_localize(None)

    # Merge DAM and IDM data on deliveryStart
    merged_data = pd.merge(df_dam[['price', 'deliveryStart']], df_idm[['priceWeightedAverage', 'deliveryStart']],
                           on='deliveryStart', suffixes=('_DAM', '_IDM')).dropna()

    # Create a scatter plot for all data
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='price', y='priceWeightedAverage', data=merged_data)

    # Perform linear regression on all data
    x = merged_data['price']
    y = merged_data['priceWeightedAverage']
    fit = np.polyfit(x, y, deg=1)

    # toto generuje priamku
    plt.plot(x, fit[0] * x + fit[1], color='red', linewidth=1, alpha=0.3)

    plt.title('Correlation graph for DAM price and IDM priceWeightedAverage')
    plt.xlabel('Price DAM €/MWh')
    plt.ylabel('Average price IDM €/MWh')
    plt.grid(True, alpha=0.5)
    plt.savefig("Graphs/Korelacia_DAM_IDM_All_Hours")
    plt.show()

    print(
        f"Correlation between DAM price and IDM priceWeightedAverage (All Hours): {merged_data['price'].corr(merged_data['priceWeightedAverage'])}")
    print(f"Linear Regression Equation for All Hours: IDM_Price = {fit[0]:.4f} * DAM_Price + {fit[1]:.4f}")

