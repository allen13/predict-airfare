import os
import pandas as pd

TRAINING_DATA = "data/Data_Train.xlsx"

def load_training_airfares():
    abs_path = os.path.abspath(TRAINING_DATA)
    training_airfares = pd.read_excel(abs_path, nrows=5,engine='openpyxl')
    return training_airfares