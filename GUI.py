import datetime

import PySimpleGUI as sg
import os
from fontTools.merge import layout
from PIL import Image
import time
import priceGraphsYearly as pgy
import PredictionAnalysis as pal

date_from = "som kokot"
date_to = "som kokot"


def home_page():
    layout_color = '#1045B0'
    layout = [
        [
            sg.Column([
                [sg.Button("Analýzy", size=(14, 2))],
                [sg.Button("Predikcie", size=(14, 2))],
                [sg.Button("Ukoncit", size=(14, 2))]
            ], justification='center', pad=(0, (110, 0)), expand_y=True,background_color=layout_color)
        ]
    ]

    # Vytvorenie okna
    window = sg.Window("Názov okna", layout, size=(600, 400),background_color=layout_color)

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
            vyber_datumov("DAM")
            break

    # Zatvorenie okna
    window.close()


def predictions_visualization(selected_image):

    layout = [
        [sg.Button("Naspäť", size=(14, 2))],
        [sg.Image(key="-IMAGE-", size=(500, 500))]
    ]

    # Vytvorenie okna
    window = sg.Window("Prehliadač Obrázkov", layout, finalize=True)

    while True:
        event, values = window.read()

        window.close()

"""
def analysis_page():
    layout = [
        [
            sg.Column([
                [sg.Button("Vykreslenie cien denného trhu", size=(30, 3))],
                [sg.Button("Vykreslenie cien vnútrodenného trhu", size=(30, 3))],
                [sg.Button("Vykreslenie cien vnútrodenného trhu s 15 minútovou periódou", size=(30, 3))],
                [sg.Button("Naspäť", size=(30, 3))]
            ], justification='center', pad=(0, (50, 0)), expand_y=True),
            sg.Column([
                [sg.Button("ACF", size=(30, 3))],
                [sg.Button("PACF", size=(30, 3))],
                [sg.Button("Histogram", size=(30, 3))]
            ], justification='center', pad=(0, (50, 0)), expand_y=True)
        ]
    ]

    # Vytvorenie okna
    window = sg.Window("Vykreslovanie analýz", size=(600, 400), layout=layout, finalize=True)

    while True:
        event, values = window.read()

        if event == "Vykreslenie cien denného trhu":
            window.close()
            vyber_datumov("DAM")
            break

        if event == "Vykreslenie cien vnútrodenného trhu":
            window.close()
            vyber_datumov("IDM")
            break

        if event == "Vykreslenie cien vnútrodenného trhu s 15 minútovou periódou":
            window.close()
            vyber_datumov("IDM15")
            break

        if event == sg.WINDOW_CLOSED:
            break

        if event == "Naspäť":
            window.close()
            home_page()
            break

        window.close()

"""
def vyber_datumov(typ_marketu):
    layout = [
        [sg.Input(readonly=True, enable_events=True, key='INPUT 1'),
         sg.CalendarButton('Dátum od', close_when_date_chosen=True, format='%Y-%m-%d', key='Calendar 1')],
        [sg.Input(readonly=True, disabled=True, enable_events=True, key='INPUT 2'),
         sg.CalendarButton('Dátum do', close_when_date_chosen=True, format='%Y-%m-%d', key='Calendar 2')],
        [sg.Button("Vykresliť", size=(39, 2))],
        [sg.Button("Naspäť", size=(39, 2))]

    ]
    layout_color = '#1045B0'
    window = sg.Window("Vykreslovanie cien", size=(600, 400), layout=layout)
    market_typ = typ_marketu
    datum_odd = "2022-01-01"
    datum_doo = "2023-01-01"

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break

        if event == "Vykresliť":
            window.close()
            #vykreslovanie(market_typ, datum_odd, datum_doo)
            break

        if event == "Naspäť":
            window.close()

            break

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

    window.close()

def vykreslovanie_analyz():
    layout = [
        [sg.Button("Naspäť", size=(30, 2)), sg.Button("Vykresli", size=(30, 2))],
        [sg.Input(readonly=True, enable_events=True, key='INPUT 1'),
         sg.CalendarButton('Dátum od', close_when_date_chosen=True, format='%Y-%m-%d', key='Calendar 1')],
        [sg.Input(readonly=True, disabled=True, enable_events=True, key='INPUT 2'),
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
        [sg.Image(key="-IMAGE-", size=(1000, 600))],
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
#vykreslovanie()