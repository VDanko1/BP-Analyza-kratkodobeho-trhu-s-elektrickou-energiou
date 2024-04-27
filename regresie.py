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

# korelacia()

def korelacie_cudzie_markety():
    with open('Data/DAM_results_2023.pkl', "rb") as file:
        data = pickle.load(file)

    columns_to_read = ['Date', 'Start', 'Poland', 'Austria', 'France', 'Denmark 1', 'Netherlands', 'Hungary',
                       'Czech Republic', 'Germany/Luxembourg', ]

    data_foreign_market = pd.read_csv('Data/DAM_EU_Trhy_2023.csv', sep=';', usecols=columns_to_read)
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

korelacie_cudzie_markety()


def korelacia_():
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
    correlations=[]
    # Vytvorte korelačné grafy pre každú hodinu
    for hour in range(24):
        hour_data = merged_data[merged_data['deliveryStart'].dt.hour == hour]

        correlation = hour_data['price'].corr(hour_data['priceWeightedAverage'])
        correlations.append(correlation)

        """
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
        """

    plt.figure(figsize=(10, 6))
    plt.bar(range(1,25), correlations, color='blue')
    plt.title('Graf korelácie medzi cenami vnutrúdenného a denného trhu - granularita 1 hodina', fontsize=16)
    plt.xlabel('Perióda', fontsize=12)
    plt.ylabel('Koeficient korelácie', fontsize=12)
    plt.xticks(range(1,25))
    plt.grid(axis='y', alpha=0.5)
    plt.show()


korelacia_()


def novaMetoda():
    with open("Data/DAM_results_2023.pkl", "rb") as file_dam:
        data_dam = pickle.load(file_dam)

    with open("Data/IDM_results_2023.pkl", "rb") as file_idm:
        data_idm = pickle.load(file_idm)

    df_dam = pd.DataFrame(data_dam)
    df_idm = pd.DataFrame(data_idm)

    df_dam['deliveryStart'] = pd.to_datetime(df_dam['deliveryStart']).dt.tz_localize(None)
    df_idm['deliveryStart'] = pd.to_datetime(df_idm['deliveryStart']).dt.tz_localize(None)

    correlation_data = []

    for hour in range(24):
        hour_data = pd.merge(df_dam[['price', 'deliveryStart']],
                             df_idm[['priceWeightedAverage', 'deliveryStart']],
                             on='deliveryStart', suffixes=('_DAM', '_IDM')).dropna()

        hour_data['hour'] = hour_data['deliveryStart'].dt.hour
        hour_data = hour_data[hour_data['hour'] == hour]

        correlation = hour_data['price'].corr(hour_data['priceWeightedAverage'])
        correlation_data.append({'Hour': hour, 'Correlation': correlation})

    # Vytvorte korelačnú tabuľku
    correlation_table = pd.DataFrame(correlation_data)

    # Vykreslite korelačnú tabuľku
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Hour', y='Correlation', data=correlation_table)
    plt.title('Korelácia medzi DAM a IDM pre každú hodinu')
    plt.xlabel('Hodina')
    plt.ylabel('Korelácia')
    #plt.savefig("Korelacia medzi hodinami DAM-IDM")
    plt.show()

    # Vypíšte tabuľku
    print(correlation_table)


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
    numeric_data = df_dam.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numeric_data.corr()  # df_dam.drop(columns=['deliveryStart', 'deliveryEnd', 'deliveryDay']).corr()

    # Nastavenie štýlu heatmapy
    sns.set(style="white")

    # Vytvorenie heatmapy
    plt.figure(figsize=(20, 20))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title('Korelácia medzi DAM údajmi')
    plt.show()

    # print(merged_data.head(50))


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
    #plt.savefig("Graphs/Korelacia_DAM_IDM_All_Hours")
    plt.show()

    print(
        f"Correlation between DAM price and IDM priceWeightedAverage (All Hours): {merged_data['price'].corr(merged_data['priceWeightedAverage'])}")
    print(f"Linear Regression Equation for All Hours: IDM_Price = {fit[0]:.4f} * DAM_Price + {fit[1]:.4f}")
