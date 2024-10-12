import datetime
import PySimpleGUI as sg
import playground as plg
import prediction_analysis as pal
import predictions as pred

def home_page():
    sg.theme('SystemDefaultForReal')
    ttk_style = 'vista'
    layout = [
        [
            sg.Column([
                [sg.Button("Analýzy", use_ttk_buttons=True, size=(10, 2))],
                [sg.Button("Predikcie", use_ttk_buttons=True, size=(10, 2))],
                [sg.Button("Ukoncit", use_ttk_buttons=True, size=(10, 2))]
            ], justification='center', pad=(0, (50, 0)), expand_y=True)
        ]
    ]

    window = sg.Window("Domov", layout, size=(600, 400), ttk_theme=ttk_style)

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

    window.close()


def predictions_visualization():
    sg.theme('SystemDefaultForReal')
    ttk_style = 'vista'

    layout = [
        [sg.Button("Naspäť", size=(30, 1), use_ttk_buttons=True)],
        [sg.Combo(["1 deň", "2 dni", "3 dni", "4 dni", "5 dní", "6 dní", "7 dní"], enable_events=True, size=(33, 8),
                  key='-COMBO-', readonly=True, default_value="Počet dní na predikovanie"),

         sg.Combo(["Predikcie denného trhu", "Predikcie vnútrodenného trhu"], enable_events=True, size=(33, 8),
                  key='-COMBO-2-', default_value="Výber trhu na predikovanie", readonly=True),

         sg.Combo(["Model SARIMAX", "Model SARIMA", "Model AR"], enable_events=True, size=(33, 8),
                  key='-COMBO-3-', default_value="Výber modelu", readonly=True)
            , sg.Button("Predikuj", use_ttk_buttons=True, size=(30, 1))],
        [sg.Image(key="-IMAGE-", size=(1000, 600), background_color="#D2D0D0"),
         sg.Multiline(size=(37, 37), key='-MULTILINE-', background_color="#D2D0D0")]
    ]

    model = "SARIMA"
    predictions_market = "DAM"
    periods_to_predict = 24

    window = sg.Window("Predikcie cien", titlebar_background_color="green", ttk_theme=ttk_style, size=(1350, 700),
                       layout=layout)

    while True:
        event, values = window.read()

        if event == "Naspäť":
            window.close()
            home_page()
            break

        if event == '-COMBO-':
            if values[event] == "1 deň":
                periods_to_predict = 1 * 24
                print(periods_to_predict)
                continue
            elif values[event] == "2 dni":
                periods_to_predict = 2 * 24
                print(periods_to_predict)
                continue
            elif values[event] == "3 dni":
                periods_to_predict = 3 * 24
                print(periods_to_predict)
                continue
            elif values[event] == "4 dni":
                periods_to_predict = 4 * 24
                print(periods_to_predict)
                continue
            elif values[event] == "5 dní":
                periods_to_predict = 5 * 24
                print(periods_to_predict)
                continue
            elif values[event] == "6 dní":
                periods_to_predict = 6 * 24
                print(periods_to_predict)
                continue
            elif values[event] == "7 dní":
                periods_to_predict = 7 * 24
                print(periods_to_predict)
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
                model = "SARIMA"
                print(model)
                continue
            if values[event] == "Model AR":
                model = "AR"
                print(model)
                continue
            if values[event] == "Model SARIMAX":
                model = "SARIMAX"
                print(model)
                continue

        if event == "Predikuj":
            if model == "SARIMA":
                predikcie = pred.sarima_model(periods_to_predict, predictions_market)
                predikcie = predikcie.rename(columns={'deliveryEnd': 'Dátum dodania', 'price': 'Cena €/MWh'})
                image_path = "Graphs/SARIMA_from_to.png"
                window["-IMAGE-"].update(filename=image_path)
                window['-MULTILINE-'].update(predikcie.to_string(index=False))
                continue

            if model == "AR":
                predikcie = pred.auto_regressive_model(periods_to_predict, predictions_market)
                predikcie = predikcie.rename(columns={'deliveryEnd': 'Dátum dodania', 'price': 'Cena €/MWh'})
                image_path = "Graphs/AR_from_to.png"
                window["-IMAGE-"].update(filename=image_path)
                window['-MULTILINE-'].update(predikcie.to_string(index=False))
                continue

            if model == "SARIMAX":
                predikcie = pred.sarimax_model(periods_to_predict, predictions_market)
                predikcie = predikcie.rename(columns={'deliveryEnd': 'Dátum dodania', 'price': 'Cena €/MWh'})
                image_path = "Graphs/SARIMAX_from_to.png"
                window["-IMAGE-"].update(filename=image_path)
                window['-MULTILINE-'].update(predikcie.to_string(index=False))
                continue

        if event == sg.WINDOW_CLOSED:
            break

        window.close()


def vykreslovanie_analyz():
    layout = [
        [sg.Button("Naspäť", size=(30, 1), use_ttk_buttons=True),
         ],
        [sg.Input(readonly=True, enable_events=True, key='INPUT 1', default_text="Dátum od"),

         sg.CalendarButton('Dátum od', close_when_date_chosen=True, format='%Y-%m-%d', key='Calendar 1')],
        [sg.Input(readonly=True, disabled=True, enable_events=True, key='INPUT 2', default_text="Dátum do"),

         sg.CalendarButton('Dátum do', close_when_date_chosen=True, format='%Y-%m-%d', key='Calendar 2')],
        [sg.Combo(['Vývoj cien denného trhu', 'Vývoj cien vnutrodenného trhu (60 minútový)',
                   'Vývoj cien vnútrodenného trhu (15 minutový)', 'Histogram denného trhu',
                   'Histogram vnútrodenného trhu (60 minútový)', 'Histogram vnútrodenného trhu (15 minútový)',
                   'ACF graf denného trhu', 'ACF graf vnútrodenného trhu (60 minútový)',
                   'ACF graf vnútrodenného trhu (15 minútový)', 'PACF graf denného trhu',
                   'PACF graf vnútrodenného trhu (60 minútový)', 'PACF graf vnútrodenného trhu (15 minútový)',
                   'Q-Q graf vnútrodenného trhu (60 minútový)', 'Q-Q graf vnútrodenného trhu (15 minútový)',
                   'Q-Q graf denného trhu'],
                  enable_events=True, size=(45, 6), key='-COMBO-'),
         sg.Button("Vykresli", size=(30, 1), use_ttk_buttons=True)],

        [sg.Text("", size=(8, 1)),  # Prázdny text vytvára priestor pred obrázkom
         sg.Image(key="-IMAGE-", size=(1000, 600), background_color="#D2D0D0")]
    ]

    ttk_style = "vista"
    sg.theme('SystemDefaultForReal')

    window = sg.Window("Analýzy", size=(1200, 750), titlebar_background_color="green", layout=layout,
                       ttk_theme=ttk_style)

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
            vybrany_datum_str = values[event]
            print(vybrany_datum_str)
            try:
                vybrany_datum_od = datetime.datetime.strptime(vybrany_datum_str, '%Y-%m-%d')
                datum_od = vybrany_datum_str
                if datetime.datetime(2020, 1, 1) <= vybrany_datum_od <= datetime.datetime.now():
                    continue
                else:
                    sg.popup_error("Zadaný dátum musí byť medzi 1.1.2020 a 1.4.2024.")
            except ValueError:
                sg.popup_error("Nesprávny formát dátumu. Použite formát YYYY-MM-DD.")

        if event == 'INPUT 2':
            vybrany_datum_str = values[event]
            print(vybrany_datum_str)
            try:
                vybrany_datum_do = datetime.datetime.strptime(vybrany_datum_str, '%Y-%m-%d')
                datum_do = vybrany_datum_str
                if datetime.datetime(2020, 1, 1) <= vybrany_datum_do <= datetime.datetime.now():
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
                plg.prices_from_to("DAM", datum_od, datum_do)
                image_path = "Graphs/prices_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

            if typ_grafu == "Vývoj cien vnutrodenného trhu (60 minútový)":
                plg.prices_from_to("IDM", datum_od, datum_do)
                image_path = "Graphs/prices_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

            elif typ_grafu == "Vývoj cien vnútrodenného trhu (15 minutový)":
                plg.plot_idm15_prices("IDM15", datum_od, datum_do)
                image_path = "Graphs/prices_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

            elif typ_grafu == "Histogram denného trhu":
                pal.histogram("DAM", datum_od, datum_do)
                image_path = "Graphs/Histogram_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

            elif typ_grafu == 'Histogram vnútrodenného trhu (60 minútový)':
                pal.histogram("IDM", datum_od, datum_do)
                image_path = "Graphs/Histogram_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

            elif typ_grafu == 'Histogram vnútrodenného trhu (15 minútový)':
                pal.histogram("IDM15", datum_od, datum_do)
                image_path = "Graphs/Histogram_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

            elif typ_grafu == 'ACF graf denného trhu':
                pal.acf_plot("DAM", datum_od, datum_do)
                image_path = "Graphs/ACF_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

            elif typ_grafu == 'ACF graf vnútrodenného trhu (60 minútový)':
                pal.acf_plot("IDM", datum_od, datum_do)
                image_path = "Graphs/ACF_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

            elif typ_grafu == 'ACF graf vnútrodenného trhu (15 minútový)':
                pal.acf_plot("IDM15", datum_od, datum_do)
                image_path = "Graphs/ACF_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

            elif typ_grafu == 'PACF graf denného trhu':
                pal.pacf_plot("DAM", datum_od, datum_do)
                image_path = "Graphs/PACF_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

            elif typ_grafu == 'PACF graf vnútrodenného trhu (60 minútový)':
                pal.pacf_plot("IDM", datum_od, datum_do)
                image_path = "Graphs/PACF_from_to.png"
                window["-IMAGE-"].update(filename=image_path)

            elif typ_grafu == 'PACF graf vnútrodenného trhu (15 minútový)':
                pal.pacf_plot("IDM15", datum_od, datum_do)
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
