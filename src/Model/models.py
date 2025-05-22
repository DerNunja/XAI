import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


try:
    df = pd.read_csv('heloc_openml.csv')
    df = df.replace('Special', np.nan)
    for col in df.columns:
        if col != 'RiskPerformance':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['target'] = df['RiskPerformance'].map({'Good': 1, 'Bad': 0})
except FileNotFoundError:
    print("Datei konnte nicht geladen werden")    

features = [col for col in df.columns if col != 'target' and col != 'RiskPerformance']
X = df[features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier(max_depth=7, random_state=42)
dt_model.fit(X_train, y_train)