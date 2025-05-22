import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Filepaths
FILE_PATH='Datensaetze/heloc_openml.csv'

def Datenerstellung():

    try:
            df = pd.read_csv(FILE_PATH)
            df = df.replace('Special', np.nan)
            for col in df.columns:
                if col != 'RiskPerformance':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df['target'] = df['RiskPerformance'].map({'Good': 1, 'Bad': 0})
            return df
            
    except FileNotFoundError:
            print("Datei konnte nicht geladen werden") 
            return None




#Pandas DATA HANDLING
df=Datenerstellung()
features = [col for col in df.columns if col != 'target' and col != 'RiskPerformance']

# Modelle
dt_model = DecisionTreeClassifier(max_depth=7, random_state=42)
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)

X = df[features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    
rf_model.fit(X_train, y_train)    
dt_model.fit(X_train, y_train)
