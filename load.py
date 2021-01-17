import os
import pandas as pd

FLIGHT_DATA = "data/Data_Train.xlsx"
TEST_FLIGHT_DATA = "data/Test_set.xlsx"
SAMPLE_FLIGHT_DATA = "data/Sample_submission.xlsx"

def load_excel_data(path):
    abs_path = os.path.abspath(path)
    data = pd.read_excel(abs_path, engine='openpyxl')

    return data  

def clean_flight_data(flights):

    # drop rows with null values
    # check for null values: flights.isnull().values.any() -> True
    flights.dropna(inplace = True)

    # check for duplicate flights: flights.duplicated()
    # drop duplicate rows but keep the first
    flights.drop_duplicates(keep='first',inplace=True)

    # convert flight stops to integers
    flights['Total_Stops'].replace(['non-stop','1 stop','2 stops','3 stops','4 stops'], [0, 1, 2, 3, 4], inplace=True)

    # drop route data covered by stops. 
    flights.drop(["Route"], axis = 1, inplace = True)

    # drop additional info mostly useless data
    flights.drop(["Additional_Info"], axis = 1, inplace = True)

    flights['Day'] = pd.to_datetime(flights["Date_of_Journey"], format = "%d/%m/%Y").dt.day
    flights['Month'] = pd.to_datetime(flights["Date_of_Journey"], format = "%d/%m/%Y").dt.month
    # ignore flight year as all data occured in 2019

    # add day of week as a point of interest
    flights['Weekday']=pd.to_datetime(flights["Date_of_Journey"], format = "%d/%m/%Y").dt.day_name()

    # No longer need date field
    flights.drop(["Date_of_Journey"], axis = 1, inplace = True)

    # break departure time up into parts of the day
    flights['Departure']=flights['Dep_Time'].apply(part_of_day)
    flights['Arrival']=flights['Arrival_Time'].apply(part_of_day)
    flights.drop(['Dep_Time','Arrival_Time'], axis = 1, inplace = True)
    
    # standardize duration on minutes
    flights['Duration']=flights['Duration'].str.replace("h", '*60').str.replace(' ','+').str.replace('m','*1').apply(eval)

    return flights

def part_of_day(time):
    time=time.strip()
    hour=(int)(time.split(':')[0])
    part = hour//4

    parts_of_day = {
        0: "mid-night",
        1: "early-morning",
        2: "morning",
        3: "afternoon",
        4: "evening",
        5: "night",
    }
    
    return parts_of_day[part]

# flights = clean_flight_data(load_excel_data(FLIGHT_DATA))