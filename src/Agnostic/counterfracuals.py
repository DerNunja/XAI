import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
from src.Model.models import dt_model, X_train,X_test,y_test

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




def counterfracuals_tree():
    # Skalierung vorbereiten (wird für Counterfactual benötigt)
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    # Wähle ein Testbeispiel mit negativem Outcome (Bad = 0)
    x_example = X_test[y_test == 0].iloc[0]

    # Liste der veränderbaren Features – du kannst sie anpassen
    veränderbare_features = [
        'ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'NumBank2TradeLines', 
        'AverageMInFile', 'NumSatisfactoryTrades', 'PercentTradesNeverDelq'
    ]

    # Counterfactual berechnen
    cf = find_counterfactual(
        x_orig=x_example,
        model=dt_model,            # oder rf_model
        X_data=X_train,
        features_to_vary=veränderbare_features,
        max_features_changed=2,
        steps=15,
        scaler=scaler
    )

    # Vergleich anzeigen
    if cf is not None:
        print("\n🔍 Vergleich Original vs. Counterfactual:")
        print(pd.DataFrame([x_example, cf], index=["Original", "Counterfactual"]).T)

