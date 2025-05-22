import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler


def find_counterfactual(x_orig, model, X_data,
                        features_to_vary,
                        max_features_changed=2,
                        steps=10,
                        scaler=None,
                        verbose=True):
    """
    Suche einen einfachen Counterfactual für ein gegebenes Modell und eine Eingabeinstanz.

    Parameter:
    - x_orig: pd.Series – Eingabeinstanz
    - model: sklearn-Modell mit .predict()
    - X_data: pd.DataFrame – Trainingsdaten (für Feature-Namen und ggf. Skalierung)
    - features_to_vary: List[str] – Features, die geändert werden dürfen
    - max_features_changed: int – Max. Anzahl gleichzeitiger Feature-Änderungen
    - steps: int – Granularität pro Feature (je höher, desto feiner)
    - scaler: MinMaxScaler (optional) – falls verwendet, um Daten zu normalisieren
    - verbose: bool – ob Infos ausgegeben werden sollen

    Rückgabe:
    - Dict mit neuem Beispiel oder None
    """

    X_cols = X_data.columns.tolist()

    # Falls Skalierung aktiv, anwenden
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(X_data)

    x_scaled = scaler.transform([x_orig])[0]
    x_orig_dict = dict(zip(X_cols, x_scaled))

    for k in range(1, max_features_changed + 1):
        for feature_combo in combinations(features_to_vary, k):
            delta_range = np.linspace(-0.5, 0.5, steps)

            # Grid über alle Feature-Kombinationen bauen
            for deltas in np.array(np.meshgrid(*[delta_range]*k)).T.reshape(-1, k):
                x_cf = x_orig_dict.copy()
                for i, f in enumerate(feature_combo):
                    x_cf[f] = np.clip(x_cf[f] + deltas[i], 0, 1)

                # Rückskalieren
                x_cf_real = scaler.inverse_transform([list(x_cf.values())])[0]
                x_cf_series = pd.Series(x_cf_real, index=X_cols)

                # Vorhersage prüfen
                if model.predict([x_cf_series])[0] == 1:
                    if verbose:
                        print("\n✅ Counterfactual gefunden!")
                        print("Geänderte Features:")
                        for f in feature_combo:
                            old_val = x_orig[f]
                            new_val = x_cf_series[f]
                            print(f"- {f}: {old_val:.2f} → {new_val:.2f}")

                    return x_cf_series

    if verbose:
        print("❌ Kein gültiger Counterfactual gefunden.")
    return None


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Funktion von vorher hier einfügen oder importieren (siehe vorheriger Post)
# ➤ find_counterfactual(...)

# Schritt 1: Künstlichen Kredit-Datensatz erstellen
np.random.seed(42)
n = 300

df = pd.DataFrame({
    "income": np.random.normal(50000, 15000, n),       # Jahreseinkommen
    "age": np.random.normal(35, 10, n),                # Alter
    "loan_amount": np.random.normal(15000, 5000, n),   # gewünschte Kredithöhe
    "credit_score": np.random.normal(650, 70, n),      # Kredit-Score
})

# Zielvariable generieren: "approved" = 1, wenn Bedingungen erfüllt
df["approved"] = (
    (df["income"] > 40000) &
    (df["credit_score"] > 600) &
    (df["loan_amount"] < 20000)
).astype(int)

# Schritt 2: Modell trainieren
X = df.drop("approved", axis=1)
y = df["approved"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)

# Schritt 3: Beispiel einer abgelehnten Instanz
x_example = X_test[y_test == 0].iloc[0]
print("\n📌 Ursprüngliche Vorhersage:", model.predict([x_example])[0])
print(x_example)

# Schritt 4: Counterfactual finden
cf = find_counterfactual(
    x_orig=x_example,
    model=model,
    X_data=X_train,
    features_to_vary=['income', 'loan_amount', 'credit_score'],
    max_features_changed=2,
    steps=20
)

# Schritt 5: Vergleich ausgeben
if cf is not None:
    print("\n🔍 Vergleich:")
    print(pd.DataFrame([x_example, cf], index=["Original", "Counterfactual"]).T)
