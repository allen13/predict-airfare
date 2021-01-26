import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

from math import sqrt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from joblib import dump, load

from data import *

from collections import defaultdict

KNN_MODEL_PATH = 'models/knn.joblib'
ENCODER_KEY_PATH = 'models/encoder-key.joblib'
FEATURE_IMPORTANCE_PATH = 'models/feature-importance.joblib'

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class FlightsModel:
  def __init__(self):
    self.load_flight_data()

  def load_flight_data(self):

    self.flights = clean_flight_data(load_excel_data(FLIGHT_DATA))
    self.test_flights = clean_flight_data(load_excel_data(TEST_FLIGHT_DATA))
    self.test_flight_prices = load_excel_data(SAMPLE_FLIGHT_DATA)

  def prepare_training_data(self):
    self.encode_flight_training_data(self.flights)
    self.flights_training_data = self.transform_flight_training_data(self.flights)
    self.test_flights_training_data = self.transform_flight_training_data(self.test_flights)
    self.split_training_data()

  def encode_flight_training_data(self, flights):
    # remove price column if exists as it is the target of the training
    if 'Price' in flights:
        flights = flights.drop(['Price'], axis=1)
    
    # encode text columns as categorical numerals
    categorical_data = flights.select_dtypes(include=['object'])

    self.encoder_key = defaultdict(LabelEncoder)
    # Encoding the variable
    categorical_data.apply(lambda x: self.encoder_key[x.name].fit_transform(x))

  def transform_flight_training_data(self, flights):
    # remove price column if exists as it is the target of the training
    if 'Price' in flights:
        flights = flights.drop(['Price'], axis=1)

    # encode text columns as categorical numerals
    categorical_data = flights.select_dtypes(include=['object'])
    numerical_data = flights.select_dtypes(include=['int64'])
    # Using the dictionary to label future data
    transformed_data = categorical_data.apply(lambda x: self.encoder_key[x.name].transform(x))

    return pd.concat([numerical_data, transformed_data], axis=1)

  def split_training_data(self):

    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.flights_training_data, 
        self.flights['Price'], 
        test_size = 0.3, 
        random_state = 42)

  def train_models(self):
    self.prepare_training_data()
    self.train_knn_model()
    self.extract_feature_importance()
    self.save_models()

  def save_models(self):
    dump(self.encoder_key, ENCODER_KEY_PATH)
    dump(self.knn_model, KNN_MODEL_PATH)
    dump(self.feature_importance, FEATURE_IMPORTANCE_PATH)

  def load_models(self):
    self.encoder_key = load(ENCODER_KEY_PATH)
    self.knn_model = load(KNN_MODEL_PATH)
    self.feature_importance = load(FEATURE_IMPORTANCE_PATH)

    self.prepare_training_data()

  def extract_feature_importance(self):
    rfr=RandomForestRegressor(n_estimators=100)
    rfr.fit(self.X_train, self.y_train)
    importances = rfr.feature_importances_
    sorted_indices = np.argsort(importances)
    self.feature_importance = {
      "features": self.X_train.columns[sorted_indices].to_list(),
      "importances": importances[sorted_indices].tolist(),
    }

  def train_knn_model(self):
    k_range = list(range(1, 30))
    params = dict(n_neighbors = k_range)
    self.knn_model = GridSearchCV(
      KNeighborsRegressor(), 
      params, cv =10, 
      scoring = 'neg_mean_squared_error'
    )
    self.knn_model.fit(self.X_train, self.y_train)

  def predict_price(self, flight):
    flight_data = pd.DataFrame(data=flight)
    prepped_flight_data = self.transform_flight_training_data(flight_data)
    result = self.knn_model.predict(prepped_flight_data)
    return result[0]

  def calculate_model_accuracy(self):
      y_train_pred = self.knn_model.predict(self.X_train)

      model_accuracy_data = pd.DataFrame(
        {
            "Model Accuracy Measure": ["Root mean squared error","Mean absolute % error","R-squared"],
            "Value": [
              sqrt(mse(self.y_train.values, y_train_pred)),
              round(mean_absolute_percentage_error(self.y_train.values, y_train_pred)),
              r2_score(self.y_train.values, y_train_pred)],
        }
      )

      return model_accuracy_data

if __name__ == "__main__":
  # use to update data models
  flights_model = FlightsModel()
  flights_model.train_models()
  # flights_model.load_models()
  print(flights_model.calculate_model_accuracy())
  flight = {
    'Airline': ['Air India'], 
    'Source': ['Banglore'],
    'Destination': ['New Delhi'],
    'Total_Stops': [0],
    'Day': [24],
    'Month': [3],
    'Weekday': ['Sunday'],
    'Departure': ['night'],
    'Arrival': ['mid-night'],
  }
  print(flights_model.predict_price(flight))
  print(flights_model.feature_importance)