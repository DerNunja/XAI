import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, export_text
import plotly.graph_objects as go
from dash import html, dcc

def build_surrogate_and_figure(X, blackbox_preds, feature_names=None, model="linear"):
    if model == "linear":
        surrogate_model = LinearRegression()
    elif model == "tree":
        surrogate_model = DecisionTreeRegressor(max_depth=4)
    else:
        raise ValueError("Modelltyp muss 'linear' oder 'tree' sein.")

    surrogate_model.fit(X, blackbox_preds)
    surrogate_preds = surrogate_model.predict(X)

    # Modellgüte (Bestimmtheitsmaß)
    sse = np.sum((blackbox_preds - surrogate_preds) ** 2)
    sst = np.sum((blackbox_preds - np.mean(blackbox_preds)) ** 2)
    goodness = 1 - sse / sst

    # Visualisierung (nur für 1 Feature sinnvoll)
    fig = go.Figure()
    if X.shape[1] == 1:
        fig.add_trace(go.Scatter(x=X.flatten(), y=blackbox_preds, mode='markers', name='Blackbox Vorhersage'))
        fig.add_trace(go.Scatter(x=X.flatten(), y=surrogate_preds, mode='lines', name='Surrogatmodell'))
        fig.update_layout(title="Surrogatmodell approximiert MLP", xaxis_title=feature_names[0] if feature_names else "Feature", yaxis_title="Vorhersage")
    else:
        fig.add_trace(go.Histogram(x=blackbox_preds - surrogate_preds, name="Fehlerverteilung"))
        fig.update_layout(title="Surrogatmodell: Fehlerverteilung", xaxis_title="Fehler", yaxis_title="Häufigkeit")

    return goodness, fig