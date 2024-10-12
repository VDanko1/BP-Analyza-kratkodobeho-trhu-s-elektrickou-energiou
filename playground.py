import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def prices_from_to(market_type, date_from, date_to):
    years = [2024, 2023, 2022, 2021, 2020]
    dataframes = []

    for year in years:
        with open(f"Data/{market_type}_results_{year}.pkl", "rb") as file_dam:
            data_dam = pickle.load(file_dam)
        df = pd.DataFrame(data_dam)

        df['deliveryEnd'] = pd.to_datetime(df['deliveryEnd']).dt.tz_localize(None)

        if market_type == "IDM":
            df.rename(columns={'priceWeightedAverage': 'price'}, inplace=True)

        df = df[['deliveryEnd', 'price']]
        df['price'] = pd.to_numeric(df['price'], errors='coerce')

        dataframes.append(df)

    merged_df = pd.concat(dataframes)

    date_from = datetime.strptime(date_from, "%Y-%m-%d")
    date_to = datetime.strptime(date_to, "%Y-%m-%d")
    merged_df = merged_df[(merged_df['deliveryEnd'] >= date_from) & (merged_df['deliveryEnd'] <= date_to)]

    plt.figure(figsize=(10, 6))
    plt.plot(merged_df['deliveryEnd'], merged_df['price'], linestyle='-', color='b')

    titles = {
        "IDM": 'Vývoj cien vnútrodenného trhu s 60 minútovou periódou',
        "DAM": 'Vývoj cien denného trhu',
    }
    plt.title(titles.get(market_type, "Vývoj cien trhu"), fontsize=16)

    plt.xlabel('Dátum', fontsize=16)
    plt.ylabel('Cena €/MWh', fontsize=16)
    plt.tight_layout()
    plt.savefig("Graphs/prices_from_to")
    plt.show()

def plot_idm15_prices(market_type, date_from, date_to):
    years = ["2024-JAN-APR", 2023]
    dataframes = []

    for year in years:
        with open(f"Data/{market_type}_results_{year}.pkl", "rb") as file_idm15:
            data_idm15 = pickle.load(file_idm15)
        df = pd.DataFrame(data_idm15)
        df['deliveryEnd'] = pd.to_datetime(df['deliveryEnd']).dt.tz_localize(None)
        df.rename(columns={'priceWeightedAverage': 'price'}, inplace=True)
        df = df[['deliveryEnd', 'price']]
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        dataframes.append(df)

    merged_df = pd.concat(dataframes)
    date_from = datetime.strptime(date_from, "%Y-%m-%d")
    date_to = datetime.strptime(date_to, "%Y-%m-%d")
    merged_df = merged_df[(merged_df['deliveryEnd'] >= date_from) & (merged_df['deliveryEnd'] <= date_to)]

    plt.figure(figsize=(12, 6))
    plt.plot(merged_df['deliveryEnd'], merged_df['price'], linestyle='-', color='b', label="Cena")

    plt.title('Vývoj cien vnútrodenného trhu s 15 minútovou periódou', fontsize=16)
    plt.xlabel('Dátum', fontsize=16)
    plt.ylabel('Cena €/MWh', fontsize=16)
    plt.tight_layout()
    plt.legend(fontsize=14)
    plt.savefig("Graphs/idm15_prices_from_to")
    plt.show()


def barplot(): # nefunguje chybaju data
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

    merged_df = pd.merge(df_idm2023, df_dam2023, on='period', suffixes=('_idm', '_dam'))

    count_higher_lower_prices = merged_df.groupby('period')['higher_than_dam'].agg(higher_prices='sum',
                                                                                   lower_prices=lambda x: (
                                                                                               x == False).sum()).reset_index()
    print(count_higher_lower_prices.head(30))

    plt.figure(figsize=(12, 6))

    bar_width = 0.35

    index = count_higher_lower_prices.index

    plt.bar(index, count_higher_lower_prices['higher_prices'], color='blue', label='Vyššia cena', width=bar_width)

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


def comparasion_idm_idm15(): #nefunguje chybaju data
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

    df_idm_selected = df_idm_2023[
        (df_idm_2023['deliveryEnd'] >= '2024-03-04') & (df_idm_2023['deliveryEnd'] <= '2024-03-09')]
    df_idm15_selected = df_idm15_2023[
        (df_idm15_2023['deliveryEnd'] >= '2024-03-04') & (df_idm15_2023['deliveryEnd'] <= '2024-03-09')]

    df_idm_selected['formatted_date'] = df_idm_selected['deliveryEnd'].dt.strftime('%d-%m')
    df_idm15_selected['formatted_date'] = df_idm15_selected['deliveryEnd'].dt.strftime('%d-%m')

    df_idm15_selected['hour'] = df_idm15_selected['deliveryEnd'].dt.hour

    hourly_avg_price = df_idm15_selected.groupby('hour')['price'].mean()

    df_idm15_selected['hourly_avg_price'] = df_idm15_selected['hour'].map(hourly_avg_price)

    plt.figure(figsize=(12, 6))
    plt.plot(df_idm_selected['deliveryEnd'], df_idm_selected['price'], label='Cena VDT60')

    plt.plot(df_idm15_selected['deliveryEnd'], df_idm15_selected['hourly_avg_price'], label='Cena VDT15')

    plt.xlabel('Dátum', fontsize=14)
    plt.ylabel('Cena €/MWh', fontsize=14)
    plt.title('Priemerna cena za hodinu VDT15 s porovnaním cien VDT60 od 3.4 do 9.4.2024 - granularita 1 hodina', fontsize=14)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()


def barplot_idm15_idm60_comparasion_per_period():
    with open(f"Data/IDM15_results_2023.pkl", "rb") as file_idm15_2023:
        data_idm15_2023 = pickle.load(file_idm15_2023)

    with open(f"Data/IDM15_results_2024-JAN-APR.pkl", "rb") as file_idm15_2024:
        data_idm15_2024 = pickle.load(file_idm15_2024)

    with open(f"Data/IDM_results_2023.pkl", "rb") as file_idm_2023:
        data_idm_2023 = pickle.load(file_idm_2023)

    with open(f"Data/IDM_results_2024-JAN-APR.pkl", "rb") as file_idm_2024:
        data_idm_2024 = pickle.load(file_idm_2024)

    df_idm15_2023 = pd.DataFrame(data_idm15_2023)
    df_idm15_2024 = pd.DataFrame(data_idm15_2024)
    df_idm_2023 = pd.DataFrame(data_idm_2023)
    df_idm_2024 = pd.DataFrame(data_idm_2024)

    for df in [df_idm_2023, df_idm_2024]:
        df['deliveryEnd'] = pd.to_datetime(df['deliveryEnd'])

    for df in [df_idm15_2023, df_idm15_2024]:
        df['deliveryEnd'] = pd.to_datetime(df['deliveryEnd'])

    df_idm15 = pd.concat([ df_idm15_2023, df_idm15_2024])
    df_idm = pd.concat([ df_idm_2023, df_idm_2024])

    df_idm15.dropna(inplace=True)
    df_idm.dropna(inplace=True)

    for df in [df_idm15,df_idm]:
        df.rename(columns={'priceWeightedAverage': 'price'}, inplace=True)


    prices_idm15 = [df_idm15[df_idm15['period'] == i]['price'].iloc[0] for i in range(1, 5)]


    price_idm60 = df_idm[df_idm['period'] == 1]['price'].iloc[0]


    comparisons = ['Áno' if price > price_idm60 else 'Nie' for price in prices_idm15]


    comparison_table = pd.DataFrame({
        'Perióda IDM15': ['Perióda 1', 'Perióda 2', 'Perióda 3', 'Perióda 4'],
        'Vyššia cena ako IDM60?': comparisons
    })

    print(comparison_table)


    plt.figure(figsize=(10, 6))


    bar_width = 0.35


    index = np.arange(4)


    plt.bar(index, [1 if comp == 'Áno' else 0 for comp in comparisons], bar_width, label='Vyššia cena', color='blue')
    plt.bar(index + bar_width, [1 if comp == 'Nie' else 0 for comp in comparisons], bar_width, label='Nižšia cena',
            color='red')

    plt.xlabel('Perióda IDM15')
    plt.ylabel('Počet')
    plt.title('Porovnanie cien IDM15 s cenou prvej periódy IDM60')
    plt.xticks(index + bar_width / 2, ['Perióda 1', 'Perióda 2', 'Perióda 3', 'Perióda 4'])
    plt.legend()

    plt.show()




def total_volume_idt15():
    with open(f"Data/IDM15_results_2023.pkl", "rb") as file_idm15_2023:
        data_idm15_2023 = pickle.load(file_idm15_2023)

    with open(f"Data/IDM15_results_2024-JAN-APR.pkl", "rb") as file_idm15_2024:
        data_idm15_2024 = pickle.load(file_idm15_2024)

    with open(f"Data/IDM15_results_2022DEC.pkl", "rb") as file_idm15_2022:
        data_idm15_2022 = pickle.load(file_idm15_2022)

    with open(f"Data/IDM_results_2023.pkl", "rb") as file_idm_2023:
        data_idm_2023 = pickle.load(file_idm_2023)

    with open(f"Data/IDM_results_2024-JAN-APR.pkl", "rb") as file_idm_2024:
        data_idm_2024 = pickle.load(file_idm_2024)

    with open(f"Data/IDM_results_2022-JUN-DEC.pkl", "rb") as file_idm_2022:
        data_idm_2022 = pickle.load(file_idm_2022)

    df_idm15_2023 = pd.DataFrame(data_idm15_2023)
    df_idm15_2024 = pd.DataFrame(data_idm15_2024)
    df_idm15_2022 = pd.DataFrame(data_idm15_2022)
    df_idm_2023 = pd.DataFrame(data_idm_2023)
    df_idm_2024 = pd.DataFrame(data_idm_2024)
    df_idm_2022 = pd.DataFrame(data_idm_2022)

    for df in [df_idm_2023, df_idm_2024, df_idm_2022]:
        df['deliveryEnd'] = pd.to_datetime(df['deliveryEnd'])

    for df in [df_idm15_2023, df_idm15_2024, df_idm15_2022]:
        df['deliveryEnd'] = pd.to_datetime(df['deliveryEnd'])

    df_idm15 = pd.concat([df_idm15_2022, df_idm15_2023, df_idm15_2024])
    df_idm = pd.concat([ df_idm_2022 ,df_idm_2023, df_idm_2024])

    df_idm15.dropna(inplace=True)
    df_idm.dropna(inplace=True)

    weekly_volume_idm15 = df_idm15.resample('W-Mon', on='deliveryEnd').sum()
    weekly_volume_idm = df_idm.resample('W-Mon', on='deliveryEnd').sum()

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

    with open(f"Data/IDM15_results_2024-JAN-APR.pkl", "rb") as file_idm15_2024:
        data_idm15_2024 = pickle.load(file_idm15_2024)

    with open(f"Data/IDM15_results_2022DEC.pkl", "rb") as file_idm15_2022:
        data_idm15_2022 = pickle.load(file_idm15_2022)

    with open(f"Data/IDM_results_2023.pkl", "rb") as file_idm_2023:
        data_idm_2023 = pickle.load(file_idm_2023)

    with open(f"Data/IDM_results_2024-JAN-APR.pkl", "rb") as file_idm_2024:
        data_idm_2024 = pickle.load(file_idm_2024)

    with open(f"Data/IDM_results_2022-JUN-DEC.pkl", "rb") as file_idm_2022:
        data_idm_2022 = pickle.load(file_idm_2022)

    df_idm15_2023 = pd.DataFrame(data_idm15_2023)
    df_idm15_2024 = pd.DataFrame(data_idm15_2024)
    df_idm15_2022 = pd.DataFrame(data_idm15_2022)
    df_idm_2023 = pd.DataFrame(data_idm_2023)
    df_idm_2024 = pd.DataFrame(data_idm_2024)
    df_idm_2022 = pd.DataFrame(data_idm_2022)

    for df in [df_idm_2023, df_idm_2024, df_idm_2022]:
        df['deliveryEnd'] = pd.to_datetime(df['deliveryEnd'])

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

    plt.xlabel('Dátum', fontsize=14)
    plt.ylabel('Týždenný objem obchodov (MWh)', fontsize=14)
    plt.title('Týždenný objem obchodov na VDT15 a VDT60- granularita 1 hodina', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
