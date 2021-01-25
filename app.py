import os
import time
from textwrap import dedent

import dash
import dash_table
import dash_auth
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import metrics

from utils import *
from train import *

VALID_USERNAME_PASSWORD_PAIRS = {
    'user': 'users'
}

# Define app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# auth = dash_auth.BasicAuth(
#     app,
#     VALID_USERNAME_PASSWORD_PAIRS
# )
server = app.server
flights_model = FlightsModel()
flights_model.load_models()

# Build component parts
div_alert = dbc.Spinner(html.Div(id="alert-msg"))
query_card = dbc.Card(
    [
        html.H4("Predicted Airfare", className="card-title"),
        dcc.Markdown(id="prediction"),
    ],
    body=True,
)

model_accuracy_data = flights_model.calculate_model_accuracy()
model_accuracy_table = dbc.Table.from_dataframe(model_accuracy_data, striped=True, bordered=True, hover=True)

most_expensive_flights = dbc.Table.from_dataframe(flights_model.flights.nlargest(10,'Price'), striped=True, bordered=True, hover=True)
cheapest_flights = dbc.Table.from_dataframe(flights_model.flights.nsmallest(10,'Price'), striped=True, bordered=True, hover=True)

controls = [
    OptionMenu(id="airline", label="Airline", values=flights_model.flights["Airline"].unique()),
    OptionMenu(id="source", label="Source", values=flights_model.flights["Source"].unique()),
    OptionMenu(id="destination", label="Destination", values=flights_model.flights["Destination"].unique()),
    OptionMenu(id="departure", label="Departure", values=flights_model.flights["Departure"].unique()),
    OptionMenu(id="arrival", label="Arrival", values=flights_model.flights["Arrival"].unique()),
    OptionMenu(id="total_stops", label="Stops", values=flights_model.flights["Total_Stops"].apply(str).unique()),
    OptionMenu(id="day", label="Flight Day", values=flights_model.flights["Day"].apply(str).unique()),
    OptionMenu(id="month", label="Flight Month", values=flights_model.flights["Month"].apply(str).unique()),
    OptionMenu(id="weekday", label="Day Of Week", values=flights_model.flights["Weekday"].unique()),
    dbc.Button("Predict Airfare", color="primary", id="button-train"),
]
flight_count = dcc.Graph(
    figure=px.histogram(flights_model.flights, x='Airline', title="Flight count by Airline"))

feature_price_impact = dcc.Graph(figure=px.bar(
            x=flights_model.feature_importance['importances'],
            y=flights_model.feature_importance['features'],
            labels={
                'x':'Feature Importance %',
                'y':'Features'},
            orientation='h',
            title='Feature Impact on Price'))

def average_price_graph(column):
    mp = flights_model.flights[[column,"Price"]].groupby([column]).mean().reset_index()
    return dcc.Graph(figure=px.bar(mp, x=column, y='Price', title="Average flight cost by " + column))


flight_cost_by_month = average_price_graph('Airline')

# Define Layout
app.layout = dbc.Container(
    fluid=True,
    children=[
        html.H1("Predict Airfare"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col([dbc.Card(controls, body=True),query_card,div_alert], md=3),
                dbc.Col([flight_count,flight_cost_by_month], md=3),
                dbc.Col([feature_price_impact,model_accuracy_table], md=3),
            ]
        ),
        html.H2("Additional flight information"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col([],md=3),
                dbc.Col([html.Label("10 most expensive flights"),most_expensive_flights], md=4),
            ]

        ),
        dbc.Row(
            [
                dbc.Col([],md=3),
                dbc.Col([html.Label("10 cheapest flights"),cheapest_flights], md=4),
            ]

        ),
    ],
    style={"margin": "auto"},
)


@app.callback(
    [
        Output("alert-msg", "children"),
        Output("prediction", "children"),
    ],
    [Input("button-train", "n_clicks")],
    [
        State("airline", "value"),
        State("source", "value"),
        State("destination", "value"),
        State("total_stops", "value"),
        State("day", "value"),
        State("month", "value"),
        State("weekday", "value"),
        State("departure", "value"),
        State("arrival", "value"),
    ],
)
def query_and_train(n_clicks, airline, source, destination, total_stops, day, month, weekday, departure, arrival):
    t0 = time.time()

    flight = {
      'Airline': [airline], 
      'Source': [source],
      'Destination': [destination],
      'Total_Stops': [int(total_stops)],
      'Day': [int(day)],
      'Month': [int(month)],
      'Weekday': [weekday],
      'Departure': [departure],
      'Arrival': [arrival],
    }

    price = flights_model.predict_price(flight)

    t1 = time.time()
    exec_time = t1 - t0
    alert_msg = f"Query time: {exec_time:.2f}s."
    alert = dbc.Alert(alert_msg, color="success", dismissable=True)

    return alert, f"#### ₹{price:.2f} on {airline}"


if __name__ == "__main__":
    app.run_server(debug=True,host='0.0.0.0',port=8080)
