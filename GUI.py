import datetime
import PySimpleGUI as sg
import priceGraphsYearly as pgy
import PredictionAnalysis as pal
import Predictions as pred

def home_page():
    layout_color = '#1045B0'
    layout = [
        [
            sg.Column([
                [sg.Button("Analýzy", size=(14, 2))],
                [sg.Button("Predikcie", size=(14, 2))],
                [sg.Button("Ukoncit", size=(14, 2))]
            ], justification='center', pad=(0, (110, 0)), expand_y=True, background_color=layout_color)
        ]
    ]

    # Vytvorenie okna
    window = sg.Window("Domov", layout, size=(600, 400), background_color=layout_color)

    while True:
        event, values = window.read()

        if event == "Analýzy":
            window.close()
            vykreslovanie_analyz()
            break

        if event == sg.WINDOW_CLOSED or event == "Ukoncit":
            break

        if event == "Predikcie":
            window.close()
            predictions_visualization()
            break

    # Zatvorenie okna
    window.close()


def predictions_visualization():
    layout = [
        [sg.Button("Naspäť",size=(30, 2)), sg.Button("Predikuj", size=(30, 2))],
        [sg.Combo(["1 deň", "2 dni", "3 dni", "4 dni", "5 dní", "6 dní", "7 dní"], enable_events=True, size=(33, 8),
                  key='-COMBO-',readonly=True,default_value="Počet dní na predikovanie"),
         sg.Combo(["Predikcie denného trhu", "Predikcie vnútrodenného trhu"], enable_events=True, size=(33, 8),
                  key='-COMBO-2-',default_value= "Výber trhu na predikovanie",readonly=True),
         sg.Combo(["Model SARIMA", "Model AR"], enable_events=True, size=(33, 8),
                  key='-COMBO-3-', default_value="Výber modelu", readonly=True)
         ],
        [sg.Image(key="-IMAGE-", size=(1000, 600)),
         sg.Multiline(size=(37, 37), key='-MULTILINE-', background_color="white")]
    ]

    predictions_market = "DAM"
    number_of_days_to_predict = 24
    layout_color = '#1045B0'
    Model = "SARIMA"

    window = sg.Window("Predikcie cien", titlebar_background_color="green" ,size=(1350, 700), layout=layout, background_color=layout_color)

    while True:
        event, values = window.read()

        if event == "Naspäť":
            window.close()
            home_page()
            break

        if event == '-COMBO-':
            if values[event] == "1 deň":
                number_of_days_to_predict = 1 * 24
                print(number_of_days_to_predict)
                continue
            elif values[event] == "2 dni":
                number_of_days_to_predict = 2 * 24
                print(number_of_days_to_predict)
                continue
            elif values[event] == "3 dni":
                number_of_days_to_predict = 3 * 24
                print(number_of_days_to_predict)
                continue
            elif values[event] == "4 dni":
                number_of_days_to_predict = 4 * 24
                print(number_of_days_to_predict)
                continue
            elif values[event] == "5 dní":
                number_of_days_to_predict = 5 * 24
                print(number_of_days_to_predict)
                continue
            elif values[event] == "6 dní":
                number_of_days_to_predict = 6 * 24
                print(number_of_days_to_predict)
                continue
            elif values[event] == "7 dní":
                number_of_days_to_predict = 7 * 24
                print(number_of_days_to_predict)
                continue

        if event == "-COMBO-2-":
            if values[event] == "Predikcie denného trhu":
                predictions_market = "DAM"
                print(predictions_market)
                continue
            if values[event] == "Predikcie vnútrodenného trhu":
                predictions_market = "IDM"
                print(predictions_market)
                continue

        if event == "-COMBO-3-":
            if values[event] == "Model SARIMA":
                Model = "SARIMA"
                print(Model)
                continue
            if values[event] == "Model AR":
                Model = "AR"
                print(Model)
                continue

        if event == "Predikuj":
            if Model == "SARIMA":
                predikcie = pred.SarimaPredikcie(number_of_days_to_predict, predictions_market)
                image_path = "Graphs/SARIMA_from_to.png"
                window["-IMAGE-"].update(filename=image_path)
                window['-MULTILINE-'].update(predikcie.to_string(index=False))
                continue
            if Model == "AR":
                predikcie = pred.AutoRegressiveModel(number_of_days_to_predict, predictions_market)
                image_path = "Graphs/AR_from_to.png"
                window["-IMAGE-"].update(filename=image_path)
                window['-MULTILINE-'].update(predikcie.to_string(index=False))
                continue

        if event == sg.WINDOW_CLOSED:
            break

        window.close()

def vykreslovanie_analyz():
    layout = [
        [sg.Button("Naspäť", size=(30, 2)), sg.Button("Vykresli", size=(30, 2))],
        [sg.Input(readonly=True, enable_events=True, key='INPUT 1',default_text="Dátum od"),
         sg.CalendarButton('Dátum od', close_when_date_chosen=True, format='%Y-%m-%d', key='Calendar 1')],
        [sg.Input(readonly=True, disabled=True, enable_events=True, key='INPUT 2',default_text="Dátum do"),
         sg.CalendarButton('Dátum do', close_when_date_chosen=True, format='%Y-%m-%d', key='Calendar 2')],
        [sg.Combo(['Vývoj cien denného trhu', 'Vývoj cien vnutrodenného trhu (60 minútový)',
                   'Vývoj cien vnútrodenného trhu (15 minutový)', 'Histogram denného trhu',
                   'Histogram vnútrodenného trhu (60 minútový)', 'Histogram vnútrodenného trhu (15 minútový)',
                   'ACF graf denného trhu', 'ACF graf vnútrodenného trhu (60 minútový)',
                   'ACF graf vnútrodenného trhu (15 minútový)', 'PACF graf denného trhu',
                   'PACF graf vnútrodenného trhu (60 minútový)', 'PACF graf vnútrodenného trhu (15 minútový)',
                   'Q-Q graf vnútrodenného trhu (60 minútový)', 'Q-Q graf vnútrodenného trhu (15 minútový)',
                   'Q-Q graf denného trhu'],
                  enable_events=True, size=(45, 6), key='-COMBO-')],
        [sg.Image(key="-IMAGE-", size=(1000, 600))]
    ]

    layout_color = '#1045B0'

    window = sg.Window("Vykreslovanie cien", size=(1200, 750), layout=layout, background_color=layout_color)

    typ_grafu = "DAM"
    datum_od = "2022-01-01"
    datum_do = "2023-01-01"

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break

        if event == "Naspäť":
            window.close()
            home_page()
            break

        if event == '-COMBO-':
            selected_value = values['-COMBO-']
            typ_grafu = selected_value
            print(f'Vybraná hodnota: {selected_value}')

        if event == 'INPUT 1':
            vybrany_datum_str = values[event]  # Predpokladám, že hodnota je reťazec (string)
            print(vybrany_datum_str)
            try:
                vybrany_datum_od = datetime.datetime.strptime(vybrany_datum_str, '%Y-%m-%d')
                datum_od = vybrany_datum_str
                if datetime.datetime(2020, 1, 1) <= vybrany_datum_od <= datetime.datetime(2024, 4, 1):
                    continue
                else:
                    sg.popup_error("Zadaný dátum musí byť medzi 1.1.2020 a 1.4.2024.")
            except ValueError:
                sg.popup_error("Nesprávny formát dátumu. Použite formát YYYY-MM-DD.")

        if event == 'INPUT 2':
            vybrany_datum_str = values[event]  # Predpokladám, že hodnota je reťazec (string)
            print(vybrany_datum_str)
            try:
                vybrany_datum_do = datetime.datetime.strptime(vybrany_datum_str, '%Y-%m-%d')
                datum_do = vybrany_datum_str
                if datetime.datetime(2020, 1, 1) <= vybrany_datum_do <= datetime.datetime(2024, 4, 1):
                    if vybrany_datum_od <= vybrany_datum_do:
                        continue
                    else:
                        sg.popup_error("Dátum od musí byť menší alebo rovný dátumu do.")
                else:
                    sg.popup_error("Zadaný dátum musí byť medzi 1.1.2020 a 1.4.2024.")
            except ValueError:
                sg.popup_error("Nesprávny formát dátumu. Použite formát YYYY-MM-DD.")

        if event == "Vykresli":
            if typ_grafu == "Vývoj cien denného trhu":
                pgy.prices_from_to("DAM", datum_od, datum_do)
                image_path = "Graphs/prices_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

            if typ_grafu == "Vývoj cien vnutrodenného trhu (60 minútový)":
                pgy.prices_from_to("IDM", datum_od, datum_do)
                image_path = "Graphs/prices_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

            elif typ_grafu == "Vývoj cien vnútrodenného trhu (15 minutový)":
                pgy.prices_from_to_IDM15("IDM15", datum_od, datum_do)
                image_path = "Graphs/prices_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

            elif typ_grafu == "Histogram denného trhu":
                pal.Histogram("DAM", datum_od, datum_do)
                image_path = "Graphs/Histogram_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

            elif typ_grafu == 'Histogram vnútrodenného trhu (60 minútový)':
                pal.Histogram("IDM", datum_od, datum_do)
                image_path = "Graphs/Histogram_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

            elif typ_grafu == 'Histogram vnútrodenného trhu (15 minútový)':
                pal.Histogram("IDM15", datum_od, datum_do)
                image_path = "Graphs/Histogram_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

            elif typ_grafu == 'ACF graf denného trhu':
                pal.ACF("DAM", datum_od, datum_do)
                image_path = "Graphs/ACF_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

            elif typ_grafu == 'ACF graf vnútrodenného trhu (60 minútový)':
                pal.ACF("IDM", datum_od, datum_do)
                image_path = "Graphs/ACF_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

            elif typ_grafu == 'ACF graf vnútrodenného trhu (15 minútový)':
                pal.ACF("IDM15", datum_od, datum_do)
                image_path = "Graphs/ACF_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

            elif typ_grafu == 'PACF graf denného trhu':
                pal.PACF("DAM", datum_od, datum_do)
                image_path = "Graphs/PACF_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

            elif typ_grafu == 'PACF graf vnútrodenného trhu (60 minútový)':
                pal.PACF("IDM", datum_od, datum_do)
                image_path = "Graphs/PACF_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

            elif typ_grafu == 'PACF graf vnútrodenného trhu (15 minútový)':
                pal.PACF("IDM15", datum_od, datum_do)
                image_path = "Graphs/PACF_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

            elif typ_grafu == 'Q-Q graf vnútrodenného trhu (15 minútový)':
                pal.qq_plot("IDM15", datum_od, datum_do)
                image_path = "Graphs/QQ_plot_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

            elif typ_grafu == 'Q-Q graf vnútrodenného trhu (60 minútový)':
                pal.qq_plot("IDM", datum_od, datum_do)
                image_path = "Graphs/QQ_plot_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

            elif typ_grafu == 'Q-Q graf denného trhu':
                pal.qq_plot("DAM", datum_od, datum_do)
                image_path = "Graphs/QQ_plot_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

    window.close()


home_page()
