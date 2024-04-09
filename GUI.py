import PySimpleGUI as sg
import os
from fontTools.merge import layout
from PIL import Image
import time


def analysis_list():
    IMAGE_DIRECTORY = "Graphs"

    obrazky = os.listdir(IMAGE_DIRECTORY)

    layout = [
        [
            sg.Listbox(obrazky, size=(40, 30), key="-IMAGE LIST-", enable_events=True),
            sg.Column([
                [sg.Button("Predikcie", size=(14, 2))],
                [sg.Button("Ukoncit", size=(14, 2))]
            ], element_justification='c')
        ]
    ]

    # Vytvorenie okna
    window = sg.Window("Prehliadač Obrázkov", layout, finalize=True)
    last_click_time = None
    last_click_row = None

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED or event == "Ukoncit":
            break

        if event == "Predikcie":
            window.close()
            show_analysis()
            break


        elif event == "-IMAGE LIST-":
            current_time = time.time()
            current_row = values["-IMAGE LIST-"][0]
            if last_click_time is not None and current_row == last_click_row and current_time - last_click_time < 0.5:
                print(current_row)
                window.close()
                show_analysis(current_row)
                break
            last_click_time = current_time
            last_click_row = current_row

    # Zatvorenie okna
    window.close()

def show_analysis(selected_image):
    max_width = 800
    max_height = 600

    layout = [
        [sg.Button("Naspäť", size=(14, 2))],
        [sg.Image(key="-IMAGE-", size=(500, 500))]
    ]

    # Vytvorenie okna
    window = sg.Window("Prehliadač Obrázkov", layout, finalize=True)

    # Adresár, kde sa nachádzajú obrázky
    image_directory = "Graphs"

    # Získanie zoznamu súborov v adresári
    image_path = os.path.join(image_directory, selected_image)

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break

        if event == "Naspäť":
            window.close()
            analysis_list()
            break

        with Image.open(image_path) as img:
            img.thumbnail((max_width, max_height))

        window["-IMAGE-"].update(filename=image_path)

        while True:
            event, values = window.read()

            if event == sg.WINDOW_CLOSED or event == "Naspäť":
                window.close()
                break

        window.close()

        """
        elif event.startswith("Načítať Obrázok"):
            try:
                # Získanie indexu vybraného obrázka
                index_obrazka = int(event.split()[-1]) - 1

                # Načítanie obrázka
                cesta_obrazka = cesty_obrazkov[index_obrazka]

                with Image.open(cesta_obrazka) as img:
                    # Zmenšenie obrázka na maximálne rozmery 800x600
                    img.thumbnail((max_width, max_height))

                # Aktualizácia oblasti pre zobrazenie obrázka
                window["-IMAGE-"].update(filename=cesta_obrazka)

            except IndexError:
                sg.popup_error("Neplatný index obrázka. Prosím, vyberte platný obrázok.")
        """

# Volanie metódy analyza_zobrazenie()
#analyza_zobrazenie()
analysis_list()