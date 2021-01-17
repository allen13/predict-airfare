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

from flask_caching import Cache
import plotly.express as px
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import metrics

from utils import *
from load import *

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

flights = clean_flight_data(load_excel_data(FLIGHT_DATA))
test_flights = clean_flight_data(load_excel_data(TEST_FLIGHT_DATA))
test_flight_prices = load_excel_data(SAMPLE_FLIGHT_DATA)

# Build component parts
div_alert = dbc.Spinner(html.Div(id="alert-msg"))
query_card = dbc.Card(
    [
        html.H4("Predicted Airfare", className="card-title"),
        dcc.Markdown(id="prediction"),
    ],
    body=True,
)

flight_table = dbc.Table.from_dataframe(flights.head(), striped=True, bordered=True, hover=True)
sample_table = dbc.Table.from_dataframe(test_flight_prices.head(), striped=True, bordered=True, hover=True)
test_flight_table = dbc.Table.from_dataframe(test_flights.head(), striped=True, bordered=True, hover=True)

controls = [
    OptionMenu(id="airline", label="Airline", values=flights["Airline"].unique()),
    OptionMenu(id="source", label="Source", values=flights["Source"].unique()),
    OptionMenu(id="destination", label="Destination", values=flights["Destination"].unique()),
    OptionMenu(id="depart", label="Departure", values=flights["Departure"].unique()),
    OptionMenu(id="arrive", label="Arrival", values=flights["Arrival"].unique()),
    OptionMenu(id="stops", label="Stops", values=flights["Total_Stops"].apply(str).unique()),
    OptionMenu(id="day", label="Flight Day", values=flights["Day"].apply(str).unique()),
    OptionMenu(id="month", label="Flight Month", values=flights["Month"].apply(str).unique()),
    OptionMenu(id="dayofweek", label="Day Of Week", values=flights["Weekday"].unique()),
    dbc.Button("Predict Airfare", color="primary", id="button-train"),
]

flight_count = dcc.Graph(figure=px.histogram(flights, x='Airline', title="Flight count by Airline"))

def average_price_graph(column):
    mp = flights[[column,"Price"]].groupby([column]).mean().reset_index()
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
                dbc.Col([dbc.Card(controls, body=True), div_alert], md=3),
                dbc.Col([query_card], md=3),
                dbc.Col([flight_count], md=4),
            ]
        ),
        html.H2("Additional flight information"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col([],md=3),
                dbc.Col([flight_count], md=4),
                dbc.Col([flight_cost_by_month], md=4),
            ]

        ),
        dbc.Row(
            [
                dbc.Col([],md=3),
                dbc.Col([flight_table,test_flight_table,sample_table], md=4),
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
        State("stops", "value"),
    ],
)
def query_and_train(n_clicks, airline, stops):
    t0 = time.time()
    t1 = time.time()
    exec_time = t1 - t0
    alert_msg = f"Queried 0 records. Total time: {exec_time:.2f}s."
    alert = dbc.Alert(alert_msg, color="success", dismissable=True)

    return alert, "#### $2123.30 " + airline


if __name__ == "__main__":
    app.run_server(debug=True,host='0.0.0.0',port=8080)
