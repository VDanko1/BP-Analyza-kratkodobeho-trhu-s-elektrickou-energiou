import PySimpleGUI as sg
import os
from fontTools.merge import layout
from PIL import Image
import time

date_from = "som kokot"
date_to = "som kokot"

def home_page():
    IMAGE_DIRECTORY = "Graphs"

    obrazky = os.listdir(IMAGE_DIRECTORY)

    layout = [
        [
            sg.Column([
                [sg.Button("Analýzy", size=(14, 2))],
                [sg.Button("Predikcie", size=(14, 2))],
                [sg.Button("Ukoncit", size=(14, 2))]
            ], justification='center', pad=(0, (110, 0)), expand_y=True)
        ]
    ]

    # Vytvorenie okna
    window = sg.Window("Názov okna", layout, size=(600, 400))

    while True:
        event, values = window.read()

        if event == "Analýzy":
            window.close()
            analysis_page()
            break

        if event == sg.WINDOW_CLOSED or event == "Ukoncit":
            break

        if event == "Predikcie":
            window.close()
            show_analysis()
            break


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
            home_page()
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

def analysis_page():
    layout = [
        [
            sg.Column([
                [sg.Button("Vykreslenie cien vnútrodenného trhu", size=(30, 3))],
                [sg.Button("Vykreslenie cien denného trhu", size=(30, 3))],
                [sg.Button("Vykreslenie cien vnútrodenného trhu s 15 minútovou periódou", size=(30, 3))]
            ], justification='center', pad=(0, (100, 0)), expand_y=True)
        ]
    ]

    # Vytvorenie okna
    window = sg.Window("Analýzy", size=(600, 400), layout=layout, finalize=True)

    while True:
        event, values = window.read()

        if event == "Vykreslenie cien vnútrodenného trhu":
            window.close()
            vykreslenie_cien()
            break

        if event == sg.WINDOW_CLOSED:
            break

        if event == "Naspäť":
            window.close()
            home_page()
            break

        window.close()

def vykreslenie_cien():
    layout = [
        #[sg.Text("Dátum od:"), sg.CalendarButton("Vybrať dátum od", key="od", format="%Y-%m-%d")],
        [sg.Input(readonly=True, enable_events=True, key='INPUT 1'),
         sg.CalendarButton('Dátum od', close_when_date_chosen=True, format='%Y-%m-%d', key='Calendar 1')],
        [sg.Input(readonly=True, enable_events=True, key='INPUT 2'),
         sg.CalendarButton('Dátum do', close_when_date_chosen=True, format='%Y-%m-%d', key='Calendar 2')],
        [sg.Button("Vykresliť", size=(40, 2))],
    ]

    window = sg.Window("Vykreslovanie cien", size=(600, 400), layout=layout)

    while True:
        event, values = window.read()


        if event == sg.WINDOW_CLOSED:
            break

        if event == "Vykresliť":
            break

        if event == 'INPUT 1':
            print(f'{values[event]}')

        if event == 'INPUT 2':
            print(f'{values[event]}')

    window.close()

home_page()