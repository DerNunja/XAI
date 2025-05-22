import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from IPython.display import Image
from src.Model.models import dt_model,rf_model,df

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Kreditwürdigkeits-Explorer", className="text-center my-4"),
            html.H3("Verständliche Einblicke in die Kreditentscheidung", className="text-center mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Einstellungen"),
                dbc.CardBody([
                    html.P("Wählen Sie einen Kredit-Score-Bereich (höher = besser):"),
                    dcc.RangeSlider(
                        id='risk-slider',
                        min=df['ExternalRiskEstimate'].min(),
                        max=df['ExternalRiskEstimate'].max(),
                        value=[df['ExternalRiskEstimate'].min(), df['ExternalRiskEstimate'].max()],
                        marks={int(i): str(i) for i in np.linspace(df['ExternalRiskEstimate'].min(), 
                                                             df['ExternalRiskEstimate'].max(), 5)},
                        step=1
                    ),
                    html.P("Mindestanzahl pünktlich bezahlter Rechnungen/Kredite:"),
                    dcc.Slider(
                        id='trades-slider',
                        min=df['NumSatisfactoryTrades'].min(),
                        max=df['NumSatisfactoryTrades'].max(),
                        value=df['NumSatisfactoryTrades'].min(),
                        marks={int(i): str(i) for i in np.linspace(df['NumSatisfactoryTrades'].min(), 
                                                             df['NumSatisfactoryTrades'].max(), 5)},
                        step=1
                    ),
                    html.P("Wählen Sie eine Darstellungsart:"),
                    dcc.Dropdown(
                        id='visualization-dropdown',
                        options=[
                            {'label': 'Entscheidungsbaum-Ansicht', 'value': 'tree'},
                            {'label': 'Entscheidungspfad-Ansicht', 'value': 'sankey'},
                            {'label': 'LIME', 'value': 'lime'},
                            {'label': 'Genauigkeits-Ansicht', 'value': 'roc'}
                        ],
                        value='tree'
                    ),
                    html.Br(),
                    dbc.Button("Analyse aktualisieren", id="update-button", color="primary", className="mt-2")
                ])
            ]),
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Visualisierung", id="viz-title")),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-1",
                        type="circle",
                        children=html.Div(id='visualization-content')
                    ),
                ])
            ]),
        ], width=9)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.Div(id='model-metrics', className="mt-4")
        ])
    ])
], fluid=True)
