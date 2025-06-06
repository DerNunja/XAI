import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
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

df=Datenerstellung()
features = [col for col in df.columns if col != 'target' and col != 'RiskPerformance']

# -----------------------------------------
# -------------- Modelle ------------------
# -----------------------------------------

# ---- Trainingssplit ----
X = df[features]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---- Random Forest ----
rf_model = RandomForestClassifier(n_estimators=530, class_weight="balanced", max_depth=30, max_features=0.38, min_samples_leaf=4, min_samples_split=7, random_state=42)
rf_model.fit(X_train, y_train)

# ---- Decision Tree ----
dt_model = DecisionTreeClassifier(max_depth=7, random_state=42)
dt_model.fit(X_train, y_train)

# ---- Gradient Boosting ----
gb_model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.04,max_depth=None, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.01, interaction_cst='pairwise', categorical_features=None, random_state=42)
gb_model.fit(X_train, y_train)

# ─── Modell-Mapping ────
model_dict = {
    "rf": rf_model,
    "gb": gb_model
}

# -----------------------------------------
# -------- Erklärbarkeits Modelle ---------
# -----------------------------------------

# ---- Surrogate ----
def train_surrogate(base_model, X, max_depth=3):
    y_hat    = base_model.predict(X)           # harte Labels
    tree     = DecisionTreeClassifier(max_depth=max_depth,
                                      random_state=42).fit(X, y_hat)
    fidelity = (tree.predict(X) == y_hat).mean()
    return tree, fidelity

surrogate_dict      = {}
surrogate_fidelity  = {}

for key, mdl in model_dict.items():
    sur_tree, fid              = train_surrogate(mdl, X_train, max_depth=3)
    surrogate_dict[key]        = sur_tree
    surrogate_fidelity[key]    = fid

# ---- LIME ----
from sklearn.impute import SimpleImputer
from lime.lime_tabular import LimeTabularExplainer

imputer = SimpleImputer(strategy="median").fit(X_train)
X_train_imp = imputer.transform(X_train)

explainer = LimeTabularExplainer(training_data=X_train_imp, feature_names=features, class_names=["Zahlungsprobleme","Zuverlässiger Zahler"], mode="classification", discretize_continuous=True, random_state=42)

median_dict = X_train.median().to_dict()