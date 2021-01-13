import os
import time
from textwrap import dedent

import dash
import dash_table
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

# Define app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

flights = load_clean_flights()
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
controls = [
    OptionMenu(id="airline", label="Airline", values=flights["Airline"].unique()),
    OptionMenu(id="stops", label="Stops", values=['0','1','2','3','4']),
    OptionMenu(id="algorithm", label="Prediction Algorithm", values=['Linear Regression','Random Forest Regressor','Decision Tree Regressor']),
    dbc.Button("Predict Airfare", color="primary", id="button-train"),
]

flight_count = dcc.Graph(figure=px.histogram(flights, x='Airline', title="Flight count by Airline"))

# Define Layout
app.layout = dbc.Container(
    fluid=True,
    children=[
        html.H1("Predict Airfare"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col([dbc.Card(controls, body=True), div_alert], md=3),
                dbc.Col([query_card], md=4),
            ]
        ),
        html.H2("Additional flight information"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col([],md=3),
                dbc.Col([flight_count], md=4),
            ]

        ),
        dbc.Row(
            [
                dbc.Col([],md=3),
                dbc.Col([flight_table], md=4),
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
