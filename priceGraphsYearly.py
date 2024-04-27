import pickle
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import datetime


def load_and_store_data_okte():
    api_url = "https://isot.okte.sk/api/v1/idm/results?deliveryDayFrom=2023-01-01&deliveryDayTo=2023-12-31&productType=15"

    response = requests.get(api_url)

    if response.status_code == 200:
        filename = "Data/IDM15_results_2023.pkl"
        with open(filename, "wb") as file:
            pickle.dump(response.json(), file)
    else:
        print(f"Error: {response.status_code}")




def prices_from_to(market_type, date_from, date_to):
    with open(f"Data/{market_type}_results_2024-JAN-APR.pkl", "rb") as file_dam:
        data_dam_2024 = pickle.load(file_dam)

    with open(f"Data/{market_type}_results_2023.pkl", "rb") as file_dam:
        data_dam_2023 = pickle.load(file_dam)

    with open(f"Data/{market_type}_results_2022.pkl", "rb") as file_dam:
        data_dam2022 = pickle.load(file_dam)

    with open(f"Data/{market_type}_results_2021.pkl", "rb") as file_dam:
        data_dam2021 = pickle.load(file_dam)

    with open(f"Data/{market_type}_results_2020.pkl", "rb") as file_dam:
        data_dam2020 = pickle.load(file_dam)

    date_from = datetime.strptime(date_from, "%Y-%m-%d")
    date_to = datetime.strptime(date_to, "%Y-%m-%d")

    df_dam2024 = pd.DataFrame(data_dam_2024)
    df_dam2023 = pd.DataFrame(data_dam_2023)
    df_dam2022 = pd.DataFrame(data_dam2022)
    df_dam2021 = pd.DataFrame(data_dam2021)
    df_dam2020 = pd.DataFrame(data_dam2020)

    for df in [df_dam2024, df_dam2023, df_dam2022, df_dam2021, df_dam2020]:
        df['deliveryEnd'] = pd.to_datetime(df['deliveryEnd']).dt.tz_localize(None)

    if market_type == "IDM":
        for df in [df_dam2024, df_dam2023, df_dam2022, df_dam2021, df_dam2020]:
            df.rename(columns={'priceWeightedAverage': 'price'}, inplace=True)

    df_dam2024 = df_dam2024[['deliveryEnd', 'price']]
    df_dam2023 = df_dam2023[['deliveryEnd', 'price']]
    df_dam2022 = df_dam2022[['deliveryEnd', 'price']]
    df_dam2021 = df_dam2021[['deliveryEnd', 'price']]
    df_dam2020 = df_dam2020[['deliveryEnd', 'price']]

    for df in [df_dam2024, df_dam2023, df_dam2022, df_dam2021, df_dam2020]:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')

    merged_df = pd.concat([df_dam2020, df_dam2021, df_dam2022, df_dam2023, df_dam2024])
    merged_df = merged_df[(merged_df['deliveryEnd'] >= date_from) & (merged_df['deliveryEnd'] <= date_to)]

    plt.figure(figsize=(10, 6))
    plt.plot(merged_df['deliveryEnd'], merged_df['price'], linestyle='-', color='b')

    if (market_type == "IDM"):
        plt.title(f'Vývoj cien vnútrodenného trhu s 60 minútovou periódou', fontsize=16)
    if (market_type == "DAM"):
        plt.title(f'Vývoj cien  denného trhu', fontsize=16)
    if (market_type == "IDM15"):
        plt.title(f'Vývoj cien vnútrodenného trhu s 15 minútovou periódou', fontsize=16)

    plt.xlabel('Dátum', fontsize=16)
    plt.ylabel('Cena €/MWh', fontsize=16)
    plt.tight_layout()
    plt.savefig("Graphs/prices_from_to")
    plt.show()


def prices_from_to_IDM15(market_type, date_from, date_to):
    with open(f"Data/{market_type}_results_2024-JAN-APR.pkl", "rb") as file_dam:
        data_dam_2024 = pickle.load(file_dam)

    with open(f"Data/{market_type}_results_2023.pkl", "rb") as file_dam:
        data_dam_2023 = pickle.load(file_dam)

    date_from = datetime.strptime(date_from, "%Y-%m-%d")
    date_to = datetime.strptime(date_to, "%Y-%m-%d")

    df_dam2024 = pd.DataFrame(data_dam_2024)
    df_dam2023 = pd.DataFrame(data_dam_2023)

    for df in [df_dam2024, df_dam2023]:
        df['deliveryEnd'] = pd.to_datetime(df['deliveryEnd']).dt.tz_localize(None)

    if market_type == "IDM15":
        for df in [df_dam2024, df_dam2023]:
            df.rename(columns={'priceWeightedAverage': 'price'}, inplace=True)

    df_dam2024 = df_dam2024[['deliveryEnd', 'price']]
    df_dam2023 = df_dam2023[['deliveryEnd', 'price']]

    for df in [df_dam2024, df_dam2023]:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')

    merged_df = pd.concat([df_dam2023, df_dam2024])
    merged_df = merged_df[(merged_df['deliveryEnd'] >= date_from) & (merged_df['deliveryEnd'] <= date_to)]

    plt.figure(figsize=(12, 6))
    plt.plot(merged_df['deliveryEnd'], merged_df['price'], linestyle='-', color='b', label="Cena")

    plt.title(f'Vývoj cien vnútrodenného trhu s 15 minútovou periódou', fontsize=16)
    plt.xlabel('Dátum', fontsize=16)
    plt.ylabel('Cena €/MWh', fontsize=16)
    plt.tight_layout()
    plt.legend(fontsize=14)
    plt.savefig("Graphs/prices_from_to")
    plt.show()


def barplot():
    with open(f"Data/DAM_results_2024-JAN-APR.pkl", "rb") as file_dam:
        data_dam_2023 = pickle.load(file_dam)

    with open(f"Data/IDM_results_2024-JAN-APR.pkl", "rb") as file_dam:
        data_idm_2023 = pickle.load(file_dam)

    df_idm2023 = pd.DataFrame(data_idm_2023)
    df_dam2023 = pd.DataFrame(data_dam_2023)

    for df in [df_idm2023, df_dam2023]:
        df['deliveryEnd'] = pd.to_datetime(df['deliveryEnd']).dt.tz_localize(None)

    for df in [df_idm2023]:
        df.rename(columns={'priceWeightedAverage': 'price'}, inplace=True)

    df_idm2023 = df_idm2023[['deliveryEnd', 'price', 'period']]
    df_dam2023 = df_dam2023[['deliveryEnd', 'price','period']]

    print("Priemerná cena pre IDM 2023:", df_idm2023['price'].mean())
    print("Priemerná cena pre DAM 2023:", df_dam2023['price'].mean())

    for df in [df_idm2023, df_dam2023]:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')

    df_idm2023['higher_than_dam'] = df_idm2023['price'] > df_dam2023['price']
    df_idm2023['lower_than_dam'] = df_idm2023['price'] < df_dam2023['price']

    # Spojenie dát na základe hodín dodávky
    merged_df = pd.merge(df_idm2023, df_dam2023, on='period', suffixes=('_idm', '_dam'))

    # Výpočet počtu, kedy bola cena vyšia a nižšia
    count_higher_lower_prices = merged_df.groupby('period')['higher_than_dam'].agg(higher_prices='sum',
                                                                                   lower_prices=lambda x: (
                                                                                               x == False).sum()).reset_index()
    print(count_higher_lower_prices.head(30))

    plt.figure(figsize=(12, 6))

    # Šírka jedného stĺpca
    bar_width = 0.35

    # Indexy pre vyššie a nižšie ceny
    index = count_higher_lower_prices.index

    # Vykreslenie stĺpcov pre vyššie ceny
    plt.bar(index, count_higher_lower_prices['higher_prices'], color='blue', label='Vyššia cena', width=bar_width)

    # Vykreslenie stĺpcov pre nižšie ceny
    plt.bar(index + bar_width, count_higher_lower_prices['lower_prices'], color='red', label='Nižšia cena',
            width=bar_width)

    plt.xlabel('Perióda', fontsize=14)
    plt.ylabel('Počet', fontsize=14)
    plt.title(
        'Porovnanie kedy je cena vyššia alebo nižšia na VDT60 trhu v porovnaní s DT za rok 2023 - granularita 1 hodina',
        fontsize=16)
    plt.legend(fontsize=12)
    plt.xticks(index + bar_width / 2, count_higher_lower_prices['period'], fontsize=10)
    plt.tight_layout()
    plt.show()

    """
    plt.figure(figsize=(12, 6))
    plt.bar(count_higher_lower_prices['period'], count_higher_lower_prices['higher_prices'], color='blue', label='Higher Prices')
    #plt.bar(count_higher_lower_prices['period'], count_higher_lower_prices['lower_prices'], color='red', label='Lower Prices')
    plt.xlabel('Period')
    plt.ylabel('Count')
    plt.title('Comparison of IDM and DAM Prices')
    plt.legend()
    plt.tight_layout()
    plt.show()
    """

def barplot_comparasion_idm_idm15():
    with open(f"Data/IDM_results_2023.pkl", "rb") as file_dam:
        data_idm_2023 = pickle.load(file_dam)

    with open(f"Data/IDM15_results_2023.pkl", "rb") as file_dam:
        data_idm15_2023 = pickle.load(file_dam)

    df_idm15_2023 = pd.DataFrame(data_idm15_2023)
    df_idm2023 = pd.DataFrame(data_idm_2023)

    for df in [df_idm2023, df_idm15_2023]:
        df['deliveryEnd'] = pd.to_datetime(df['deliveryEnd']).dt.tz_localize(None)



def comparasion_idm_idm15():
    with open(f"Data/IDM15_results_2024-JAN-APR.pkl", "rb") as file_idm15_2023:
        data_idm15_2023 = pickle.load(file_idm15_2023)

    with open(f"Data/IDM_results_2024-JAN-APR.pkl", "rb") as file_idm_2023:
        data_idm_2023 = pickle.load(file_idm_2023)

    df_idm15_2023 = pd.DataFrame(data_idm15_2023)
    df_idm_2023 = pd.DataFrame(data_idm_2023)

    df_idm15_2023 = df_idm15_2023[['priceWeightedAverage', 'deliveryEnd','period']]
    df_idm_2023 = df_idm_2023[['priceWeightedAverage', 'deliveryEnd','period']]

    for df in [df_idm_2023, df_idm15_2023]:
        df['deliveryEnd'] = pd.to_datetime(df['deliveryEnd'])

    for df in [df_idm_2023, df_idm15_2023]:
        df.rename(columns={'priceWeightedAverage': 'price'}, inplace=True)

    # Select data for the specified date range
    df_idm_selected = df_idm_2023[
        (df_idm_2023['deliveryEnd'] >= '2024-03-04') & (df_idm_2023['deliveryEnd'] <= '2024-03-07')]
    df_idm15_selected = df_idm15_2023[
        (df_idm15_2023['deliveryEnd'] >= '2024-03-04') & (df_idm15_2023['deliveryEnd'] <= '2024-03-07')]

    df_idm_selected['formatted_date'] = df_idm_selected['deliveryEnd'].dt.strftime('%d-%m')
    df_idm15_selected['formatted_date'] = df_idm15_selected['deliveryEnd'].dt.strftime('%d-%m')

    df_idm15_selected['hour'] = df_idm15_selected['deliveryEnd'].dt.hour

    # Vypočítame priemer ceny za každú hodinu
    hourly_avg_price = df_idm15_selected.groupby('hour')['price'].mean()

    # Vytvoríme nový stĺpec s priemernými cenami pre každú hodinu
    df_idm15_selected['hourly_avg_price'] = df_idm15_selected['hour'].map(hourly_avg_price)


    print(df_idm_selected.head(30))
    # Plot the 'price' for IDM
    plt.figure(figsize=(12, 6))
    plt.plot(df_idm_selected['deliveryEnd'], df_idm_selected['price'], label='Cena VDT60')

    # Plot the 'price' for IDM15
    plt.plot(df_idm15_selected['deliveryEnd'], df_idm15_selected['hourly_avg_price'], label='Cena VDT15')

    # Set labels and title
    plt.xlabel('Dátum a perióda', fontsize=14)
    plt.ylabel('Cena €/MWh', fontsize=14)
    plt.title('Priemerna cena za hodinu VDT15 s porovnaním cien VDT60 od 3.4 do 7.3.2024 - granularita 1 hodina', fontsize=14)

    # Set grid and legend
    plt.legend(fontsize=12)

    # Increase font size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Show plot
    plt.tight_layout()
    plt.show()



def total_volume_idt15():
    with open(f"Data/IDM15_results_2023.pkl", "rb") as file_idm15_2023:
        data_idm15_2023 = pickle.load(file_idm15_2023)

    # Načítanie dát pre rok 2024 (január až apríl)
    with open(f"Data/IDM15_results_2024-JAN-APR.pkl", "rb") as file_idm15_2024:
        data_idm15_2024 = pickle.load(file_idm15_2024)

    # Načítanie dát pre rok 2022 (december)
    with open(f"Data/IDM15_results_2022DEC.pkl", "rb") as file_idm15_2022:
        data_idm15_2022 = pickle.load(file_idm15_2022)

    # Načítanie dát pre rok 2023
    with open(f"Data/IDM_results_2023.pkl", "rb") as file_idm_2023:
        data_idm_2023 = pickle.load(file_idm_2023)

    # Načítanie dát pre rok 2024 (január až apríl)
    with open(f"Data/IDM_results_2024-JAN-APR.pkl", "rb") as file_idm_2024:
        data_idm_2024 = pickle.load(file_idm_2024)

    # Načítanie dát pre rok 2022 (december)
    with open(f"Data/IDM_results_2022-JUN-DEC.pkl", "rb") as file_idm_2022:
        data_idm_2022 = pickle.load(file_idm_2022)

    # Vytvorenie DataFrame z načítaných dát
    df_idm15_2023 = pd.DataFrame(data_idm15_2023)
    df_idm15_2024 = pd.DataFrame(data_idm15_2024)
    df_idm15_2022 = pd.DataFrame(data_idm15_2022)
    df_idm_2023 = pd.DataFrame(data_idm_2023)
    df_idm_2024 = pd.DataFrame(data_idm_2024)
    df_idm_2022 = pd.DataFrame(data_idm_2022)

    for df in [df_idm_2023, df_idm_2024, df_idm_2022]:
        df['deliveryEnd'] = pd.to_datetime(df['deliveryEnd'])

    # Konverzia stĺpca deliveryEnd na dátumový formát
    for df in [df_idm15_2023, df_idm15_2024, df_idm15_2022]:
        df['deliveryEnd'] = pd.to_datetime(df['deliveryEnd'])

    # Spojenie dát pre všetky roky
    df_idm15 = pd.concat([df_idm15_2022, df_idm15_2023, df_idm15_2024])
    df_idm = pd.concat([ df_idm_2022 ,df_idm_2023, df_idm_2024])

    df_idm15.dropna(inplace=True)
    df_idm.dropna(inplace=True)


    # Agregácia týždenného objemu obchodov
    weekly_volume_idm15 = df_idm15.resample('W-Mon', on='deliveryEnd').sum()
    weekly_volume_idm = df_idm.resample('W-Mon', on='deliveryEnd').sum()




    # Vykreslenie grafu
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_volume_idm15.index,
             (weekly_volume_idm15['purchaseTotalVolume'] + weekly_volume_idm15['saleTotalVolume']) / 1000,
             marker=".", linestyle='-', label="Objem VDT15")

    plt.plot(weekly_volume_idm.index,
             (weekly_volume_idm['purchaseTotalVolume'] + weekly_volume_idm['saleTotalVolume'])/1000,
             marker=".", linestyle='-', label="Objem VDT60")
    plt.xlabel('Dátum', fontsize=14)
    plt.ylabel('Objem zobchodovanej energie za týždeň (GWh)', fontsize=14)
    plt.title('Týždenný objem obchodov na VDT15 a VDT60 - granularita 1 hodina', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def dopyt_ponuka_idm_idm15():
    with open(f"Data/IDM15_results_2023.pkl", "rb") as file_idm15_2023:
        data_idm15_2023 = pickle.load(file_idm15_2023)

    # Načítanie dát pre rok 2024 (január až apríl)
    with open(f"Data/IDM15_results_2024-JAN-APR.pkl", "rb") as file_idm15_2024:
        data_idm15_2024 = pickle.load(file_idm15_2024)

    # Načítanie dát pre rok 2022 (december)
    with open(f"Data/IDM15_results_2022DEC.pkl", "rb") as file_idm15_2022:
        data_idm15_2022 = pickle.load(file_idm15_2022)

    # Načítanie dát pre rok 2023
    with open(f"Data/IDM_results_2023.pkl", "rb") as file_idm_2023:
        data_idm_2023 = pickle.load(file_idm_2023)

    # Načítanie dát pre rok 2024 (január až apríl)
    with open(f"Data/IDM_results_2024-JAN-APR.pkl", "rb") as file_idm_2024:
        data_idm_2024 = pickle.load(file_idm_2024)

    # Načítanie dát pre rok 2022 (december)
    with open(f"Data/IDM_results_2022-JUN-DEC.pkl", "rb") as file_idm_2022:
        data_idm_2022 = pickle.load(file_idm_2022)

    # Vytvorenie DataFrame z načítaných dát
    df_idm15_2023 = pd.DataFrame(data_idm15_2023)
    df_idm15_2024 = pd.DataFrame(data_idm15_2024)
    df_idm15_2022 = pd.DataFrame(data_idm15_2022)
    df_idm_2023 = pd.DataFrame(data_idm_2023)
    df_idm_2024 = pd.DataFrame(data_idm_2024)
    df_idm_2022 = pd.DataFrame(data_idm_2022)

    for df in [df_idm_2023, df_idm_2024, df_idm_2022]:
        df['deliveryEnd'] = pd.to_datetime(df['deliveryEnd'])

    # Konverzia stĺpca deliveryEnd na dátumový formát
    for df in [df_idm15_2023, df_idm15_2024, df_idm15_2022]:
        df['deliveryEnd'] = pd.to_datetime(df['deliveryEnd'])

    df_idm15 = pd.concat([df_idm15_2022, df_idm15_2023, df_idm15_2024])
    df_idm = pd.concat([ df_idm_2022 ,df_idm_2023, df_idm_2024])

    weekly_volume_idm15 = df_idm15.resample('W-Mon', on='deliveryEnd').sum()
    weekly_volume_idm = df_idm.resample('W-Mon', on='deliveryEnd').sum()

    max_volume = max(weekly_volume_idm15['purchaseTotalVolume'].max(),
                     weekly_volume_idm['purchaseTotalVolume'].max(),
                     weekly_volume_idm15['saleTotalVolume'].max(),
                     weekly_volume_idm['saleTotalVolume'].max())

    plt.figure(figsize=(12,6))
    plt.ylim(0, max_volume)

    plt.plot(weekly_volume_idm15.index,
             weekly_volume_idm15['purchaseTotalVolume'],
             marker=".", linestyle='-', label="Dopyt VDT15")

    plt.plot(weekly_volume_idm.index,
             weekly_volume_idm['purchaseTotalVolume'],
             marker=".", linestyle='-', label="Dopyt VDT60")

    """
    plt.plot(weekly_volume_idm15.index,
             weekly_volume_idm15['saleTotalVolume'],
             marker=".", linestyle='-', label="Ponuka VDT15")
    plt.plot(weekly_volume_idm.index,
             weekly_volume_idm['saleTotalVolume'],
             marker=".", linestyle='-', label="Ponuka VDT60")
    """

    plt.xlabel('Dátum', fontsize=14)  # Zvětšení písma osy x
    plt.ylabel('Týždenný objem obchodov (MWh)', fontsize=14)  # Zvětšení písma osy y
    plt.title('Týždenný objem obchodov na VDT15 a VDT60 - granularita 1 hodina', fontsize=16)  # Zvětšení písma titulku
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

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

        # dates_dam = [datetime.strptime(date, '%Y-%m-%d') for date in date_idm]
        dates_idm = [datetime.strptime(date, '%Y-%m-%d') for date in date_dam]

        # Create the plot with two lines
        plt.figure(figsize=(12, 6))
        plt.plot(price_dam, color='g', label='Ceny za prvý týždeň')
        plt.plot(price_dam2, color='r', label='Ceny za druhý týždeň')
        plt.plot(price_dam3, color='b', label='Ceny za tretí týždeň')
        plt.plot(price_dam4, color='olive', label='Ceny za štvrtý týždeň')

        plt.title('Cenny denného trhu za december 2023 - granularita 1 hodina', fontsize=16)
        plt.xlabel('Perióda', fontsize=16)
        plt.ylabel('Cena €/MWh', fontsize=16)
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
# visualize_av_prices_overlay("Data/IDM_results_2021.pkl", "Data/Oil_price_2021")
#load_and_store_data_okte()
# prices_from_to("DAM", "2023-12-01", "2024-02-01")
# data_preparing2("DAM")#, "2023-01-01", "2023-02-01")
# visualize_av_prices_overlay()
# visualize_av_prices_overlay()
# load_and_store_data_borrowed()
# load_and_store_data_oil()
# print_data()
#barplot_volume_of_trades()
#total_volume_idt15()
comparasion_idm_idm15()
#dopyt_ponuka_idm_idm15()
# visualize_av_prices_overlay()
# visualize_av_prices_not_overlay("Data/IDM_results_2022.pkl", "Data/IDM_results_2023.pkl")
# visualize_av_price_in_year("IDM_results_2020.pkl")
