import pickle
import requests

def load_and_store_data_okte():
    api_url = "https://isot.okte.sk/api/v1/idm/results?deliveryDayFrom=2024-01-01&deliveryDayTo=2024-10-10&productType=60"
    response = requests.get(api_url)

    if response.status_code == 200:
        filename = "Data/IDM_results_2024.pkl"
        with open(filename, "wb") as file:
            pickle.dump(response.json(), file)
    else:
        print(f"Error: {response.status_code}")

