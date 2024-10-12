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

    df_dam = pd.DataFrame(data_dam)
    df_idm = pd.DataFrame(data_idm)

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


def korelacie_cudzie_markety():
    with open('Data/DAM_results_2023.pkl', "rb") as file:
        data = pickle.load(file)

    columns_to_read = ['Date', 'Start', 'Poland', 'Austria', 'France', 'Denmark 1', 'Netherlands', 'Hungary',
                       'Czech Republic', 'Germany/Luxembourg', ]

    data_foreign_market = pd.read_csv('Data/DAM_EU_Market_2023.csv', sep=';', usecols=columns_to_read)
    df_slovak_market = pd.DataFrame(data)

    df_slovak_market['deliveryStart'] = pd.to_datetime(df_slovak_market['deliveryStart'])
    data_foreign_market['Date'] = pd.to_datetime(data_foreign_market['Date'])
    data_foreign_market['Start'] = pd.to_datetime(data_foreign_market['Start'], format="%I:%M %p").dt.strftime(
        '%H:%M:%S')

    data_foreign_market['Date'] = data_foreign_market['Date'].astype(str) + ' ' + data_foreign_market[
        'Start'].astype(str)
    data_foreign_market['Date'] = pd.to_datetime(data_foreign_market['Date'], format="%Y-%m-%d %H:%M:%S")
    data_foreign_market['Date'] = data_foreign_market['Date'] - pd.Timedelta(hours=1)

    filtered_data = pd.concat([df_slovak_market['price'], data_foreign_market['Czech Republic']], axis=1)
    filtered_data = filtered_data[(filtered_data['price'] > -100) & (filtered_data['Czech Republic'] > -100)]

    x = filtered_data['price']
    y = filtered_data['Czech Republic']

    fit = np.polyfit(x, y, deg=1)

    plt.figure(figsize=(20, 6))
    sns.scatterplot(x='price', y='Czech Republic', data=filtered_data)
    plt.plot(x, fit[0] * x + fit[1], color='red', linewidth=1, alpha=0.3)

    plt.title('Correlation graph Slovak market prices and Czech Republic market prices - granularity 1 hour',
              fontsize=20)
    plt.xlabel('Slovak market prices €/MWh', fontsize = 12)
    plt.ylabel('Czech republic market rices €/MWh', fontsize = 12)
    plt.legend()
    plt.grid(True)
    plt.savefig('Graphs/Slovensko-Cesky trh korelacia - final')
    plt.show()


def korelacia_hodinova():
    with open("Data/DAM_results_2023.pkl", "rb") as file_dam:
        data_dam = pickle.load(file_dam)

    with open("Data/IDM_results_2023.pkl", "rb") as file_idm:
        data_idm = pickle.load(file_idm)

    df_dam = pd.DataFrame(data_dam)
    df_idm = pd.DataFrame(data_idm)

    df_dam['deliveryStart'] = pd.to_datetime(df_dam['deliveryStart']).dt.tz_localize(None)
    df_idm['deliveryStart'] = pd.to_datetime(df_idm['deliveryStart']).dt.tz_localize(None)

    merged_data = pd.merge(df_dam[['price', 'deliveryStart']], df_idm[['priceWeightedAverage', 'deliveryStart']],
                           on='deliveryStart', suffixes=('_DAM', '_IDM')).dropna()

    merged_data['hour'] = merged_data['deliveryStart'].dt.hour
    correlations=[]
    for hour in range(24):
        hour_data = merged_data[merged_data['deliveryStart'].dt.hour == hour]

        correlation = hour_data['price'].corr(hour_data['priceWeightedAverage'])
        correlations.append(correlation)


        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='price', y='priceWeightedAverage', data=hour_data)

        x = hour_data['price']
        y = hour_data['priceWeightedAverage']
        fit = np.polyfit(x, y, deg=1)

        plt.plot(x, fit[0] * x + fit[1], color='red', linewidth=1, alpha=0.3)
        hodina = hour
        plt.title(
            f'Lineárna regresia medzi cenami denného a vnútrodenného trhu pre periódu {hodina + 1} - granularita 1 hodina', fontsize=12.5)
        plt.xlabel('Cena na dennom trhu €/MWh', fontsize=12)
        plt.ylabel('Cena na vnútrodennom trhu €/MWh',fontsize=12)
        plt.grid(True, alpha=0.5)
        #plt.savefig(f"Graphs/Korelacia_DAM_IDM_Hour_{hour}")
        plt.show()

        #print(f"Correlation between DAM price and IDM priceWeightedAverage (Hour {hour}): {correlation}")
        #print(f"Linear Regression Equation for Hour {hour}: IDM_Price = {fit[0]:.4f} * DAM_Price + {fit[1]:.4f}")

    plt.figure(figsize=(10, 6))
    plt.bar(range(1,25), correlations, color='blue')
    plt.title('Graf korelácie medzi cenami vnutrúdenného a denného trhu - granularita 1 hodina', fontsize=16)
    plt.xlabel('Perióda', fontsize=12)
    plt.ylabel('Koeficient korelácie', fontsize=12)
    plt.xticks(range(1,25))
    plt.grid(axis='y', alpha=0.5)
    plt.show()

