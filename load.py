import os
import pandas as pd

TRAINING_DATA = "data/Data_Train.xlsx"

def load_clean_flights():
    abs_path = os.path.abspath(TRAINING_DATA)
    flights = pd.read_excel(abs_path, engine='openpyxl')

    flights['Total_Stops'].replace(['non-stop','1 stop','2 stops','3 stops','4 stops'], [0, 1, 2, 3, 4], inplace=True)

    return flights