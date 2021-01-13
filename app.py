import os
import time
from textwrap import dedent

import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from flask_caching import Cache
import plotly.express as px
import pandas as pd
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import metrics

from utils import *
from load import *

# Define app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

airfares = load_training_airfares()
# Build component parts
div_alert = dbc.Spinner(html.Div(id="alert-msg"))
query_card = dbc.Card(
    [
        html.H4("Predicted Airfare", className="card-title"),
        dcc.Markdown(id="sql-query"),
    ],
    body=True,
)

controls = [
    OptionMenu(id="airline", label="Airline", values=airfares["Airline"].unique()),
    OptionMenu(id="stops", label="Stops", values=['1','2','3','4']),
    dbc.Button("Predict Airfare", color="primary", id="button-train"),
]


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
    ],
    style={"margin": "auto"},
)


@app.callback(
    [
        Output("alert-msg", "children"),
        Output("sql-query", "children"),
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
