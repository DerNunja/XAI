import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import base64
import pydotplus
from IPython.display import Image

@app.callback(
    [Output('visualization-content', 'children'),
     Output('viz-title', 'children'),
     Output('model-metrics', 'children')],
    [Input('update-button', 'n_clicks')],
    [State('risk-slider', 'value'),
     State('trades-slider', 'value'),
     State('visualization-dropdown', 'value')],
)
# Callback zum Aktualisieren der Visualisierung

def update_visualization(n_clicks, risk_range, min_trades, selected_viz):
    # Daten filtern
    filtered_df = df[(df['ExternalRiskEstimate'] >= risk_range[0]) & 
                     (df['ExternalRiskEstimate'] <= risk_range[1]) &
                     (df['NumSatisfactoryTrades'] >= min_trades)]
    
    if len(filtered_df) > 0:
        X_filtered = filtered_df[features]
        y_filtered = filtered_df['target']
        y_pred = rf_model.predict(X_filtered)
        accuracy = np.mean(y_pred == y_filtered)
        metrics_div = html.Div([
            html.H4("Auswertung"),
            html.P(f"Anzahl der ausgewählten Personen: {len(filtered_df)}"),
            html.P(f"Trefferquote der Vorhersage: {accuracy:.4f} (1.0 = perfekt)")
        ])
    else:
        metrics_div = html.Div([
            html.H4("Auswertung"),
            html.P("Keine Daten nach Filterung übrig")
        ])
    
    title = "Visualisierung"
    
    simple_feature_names = {}
    for f in features:
        simple_feature_names[f] = simplified_terms.get(f, f)
        
    if selected_viz == 'tree':
        title = "Entscheidungsbaum-Ansicht"
        if len(filtered_df) < 10: 
            return html.P("Nicht genügend Daten für die Ansicht nach Filterung."), title, metrics_div

        simplified_feature_names = [simplified_terms.get(feat, feat) for feat in features]
        
        dot_data = export_graphviz(
            dt_model,
            out_file=None,
            feature_names=simplified_feature_names,
            class_names=['Zahlungsprobleme', 'Zuverlässiger Zahler'],
            filled=True,
            rounded=True,
            special_characters=True,
            max_depth=3
        )
        graph = pydotplus.graph_from_dot_data(dot_data)
        img_data = graph.create_png()
        encoded_img = base64.b64encode(img_data).decode('utf-8')
        
        return html.Div([
            html.Img(src=f'data:image/png;base64,{encoded_img}', style={'width': '100%'}),
            html.H5("So lesen Sie den Entscheidungsbaum:", className="mt-4"),
            html.P("Der Baum zeigt, welche Faktoren am wichtigsten für die Kreditentscheidung sind."),
            html.P("Jeder Verzweigungspunkt ist eine Frage (z.B. 'Ist Ihr Kredit-Score über 72?'). Folgen Sie den Zweigen zur Vorhersage: 'Zuverlässiger Zahler' oder 'Zahlungsprobleme'.")
        ]), title, metrics_div
    
    elif selected_viz == 'sankey':
        title = "Entscheidungspfad-Ansicht"
        if len(filtered_df) < 10:
            return html.P("Nicht genügend Daten für die Ansicht nach Filterung."), title, metrics_div
        
        # Top Features aus dem Random Forest extrahieren
        importances = rf_model.feature_importances_
        top_indices = np.argsort(importances)[-3:]  # Top 5 Features
        top_features = [features[i] for i in top_indices]
        
        # Sankey-Diagramm erstellen
        # Labels für das Sankey-Diagramm mit vereinfachten Begriffen
        labels = []
        for feature in top_features:
            simple_feature = simplified_terms.get(feature, feature)
            
            # Spezielle Beschreibungen für bestimmte Feature-Werte
            schlecht_label = "Schlecht"
            mittel_label = "Mittel"
            gut_label = "Gut"
            
            # Spezifische Bezeichnungen je nach Feature anpassen
            if "Score" in simple_feature:
                schlecht_label = "Niedrig"
                gut_label = "Hoch"
            elif "Monate" in simple_feature:
                schlecht_label = "Kürzlich"
                gut_label = "Lange her"
            elif "Anzahl" in simple_feature or "Kreditanfragen" in simple_feature:
                schlecht_label = "Viele"
                gut_label = "Wenige"
            elif "Prozent" in simple_feature:
                schlecht_label = "Gering"
                gut_label = "Hoch"
            elif "Auslastung" in simple_feature:
                schlecht_label = "Hoch"
                gut_label = "Niedrig"
                
            # Für jedes Feature diskretisieren wir die Werte mit angepassten Beschreibungen
            labels.extend([f"{simple_feature} ({schlecht_label})", f"{simple_feature} ({mittel_label})", f"{simple_feature} ({gut_label})"])
        
        # Zielkategorien hinzufügen
        labels.extend(["Zahlungsprobleme wahrscheinlich", "Zuverlässiger Zahler"])
        
        # Quell- und Zielindizes für Links
        source = []
        target = []
        value = []
        
        # Verbindungen zwischen Features erstellen
        sample_size = min(1000, len(filtered_df))
        sampled_df = filtered_df.sample(sample_size) if len(filtered_df) > sample_size else filtered_df
        
        # Verbindungen vom ersten Feature zum zweiten Feature
        for i in range(3):  # Niedrig, Mittel, Hoch des ersten Features
            for j in range(3):  # Niedrig, Mittel, Hoch des zweiten Features
                if i == 0:
                    mask1 = sampled_df[top_features[0]] <= sampled_df[top_features[0]].quantile(0.33)
                elif i == 1:
                    mask1 = (sampled_df[top_features[0]] > sampled_df[top_features[0]].quantile(0.33)) & (sampled_df[top_features[0]] <= sampled_df[top_features[0]].quantile(0.66))
                else:
                    mask1 = sampled_df[top_features[0]] > sampled_df[top_features[0]].quantile(0.66)
                
                q33_2 = sampled_df[top_features[1]].quantile(0.33)
                q66_2 = sampled_df[top_features[1]].quantile(0.66)
                
                if j == 0:
                    mask2 = sampled_df[top_features[1]] <= q33_2
                elif j == 1:
                    mask2 = (sampled_df[top_features[1]] > q33_2) & (sampled_df[top_features[1]] <= q66_2)
                else:
                    mask2 = sampled_df[top_features[1]] > q66_2
                
                count = sum(mask1 & mask2)
                if count > 0:
                    source.append(i)
                    target.append(3 + j)
                    value.append(count)
        
        # Verbindungen vom zweiten zum dritten Feature
        for i in range(3):  # Niedrig, Mittel, Hoch des zweiten Features
            for j in range(3):  # Niedrig, Mittel, Hoch des dritten Features
                q33_2 = sampled_df[top_features[1]].quantile(0.33)
                q66_2 = sampled_df[top_features[1]].quantile(0.66)
                
                if i == 0:
                    mask1 = sampled_df[top_features[1]] <= q33_2
                elif i == 1:
                    mask1 = (sampled_df[top_features[1]] > q33_2) & (sampled_df[top_features[1]] <= q66_2)
                else:
                    mask1 = sampled_df[top_features[1]] > q66_2
                
                q33_3 = sampled_df[top_features[2]].quantile(0.33)
                q66_3 = sampled_df[top_features[2]].quantile(0.66)
                
                if j == 0:
                    mask2 = sampled_df[top_features[2]] <= q33_3
                elif j == 1:
                    mask2 = (sampled_df[top_features[2]] > q33_3) & (sampled_df[top_features[2]] <= q66_3)
                else:
                    mask2 = sampled_df[top_features[2]] > q66_3
                
                count = sum(mask1 & mask2)
                if count > 0:
                    source.append(3 + i)
                    target.append(6 + j)
                    value.append(count)
        
        # Verbindungen vom dritten Feature zur Zielklasse
        for i in range(3):  # Niedrig, Mittel, Hoch des dritten Features
            q33_3 = sampled_df[top_features[2]].quantile(0.33)
            q66_3 = sampled_df[top_features[2]].quantile(0.66)
            
            if i == 0:
                mask = sampled_df[top_features[2]] <= q33_3
            elif i == 1:
                mask = (sampled_df[top_features[2]] > q33_3) & (sampled_df[top_features[2]] <= q66_3)
            else:
                mask = sampled_df[top_features[2]] > q66_3
            
            bad_count = sum(mask & (sampled_df['target'] == 0))
            good_count = sum(mask & (sampled_df['target'] == 1))
            
            if bad_count > 0:
                source.append(6 + i)
                target.append(9)  # "Zahlungsprobleme wahrscheinlich"
                value.append(bad_count)
            
            if good_count > 0:
                source.append(6 + i)
                target.append(10)  # "Zuverlässiger Zahler"
                value.append(good_count)
        
        # Farben für die Links
        link_colors = ['rgba(44, 160, 44, 0.3)'] * len(source)  # Grüntöne für alle Links
        
        # Farben für die Knoten
        node_colors = ['rgba(31, 119, 180, 0.8)'] * 9 + ['rgba(214, 39, 40, 0.8)', 'rgba(44, 160, 44, 0.8)']
        
        # Sankey-Diagramm erstellen
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=node_colors
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=link_colors
            )
        )])
        
        #Zeige eine verständliche Erläuterung an
        feature_explanation = "<br>".join([f"• <b>{simplified_terms.get(feat, feat)}</b>: {get_feature_explanation(feat)}" 
                                         for feat in top_features[:3]])
        
        fig.update_layout(
            title_text="Wie verschiedene Faktoren Ihre Kreditwürdigkeit beeinflussen",
            font_size=12,
            height=600,
            annotations=[
                dict(
                    x=0.5,
                    y=-0.15,
                    showarrow=False,
                    text=f"<b>Die wichtigsten Faktoren:</b><br>{feature_explanation}",
                    xref="paper",
                    yref="paper",
                    align="left"
                )
            ]
        )
        
        return dcc.Graph(figure=fig), title, metrics_div
    
    elif selected_viz == 'roc':
        title = "Genauigkeits-Ansicht"
        if len(filtered_df) < 10:
            return html.P("Nicht genügend Daten für die Ansicht nach Filterung."), title, metrics_div
        
        # ROC-Kurve berechnen
        y_scores = rf_model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'Vorhersage-Qualität (AUC = {roc_auc:.3f})',
            line=dict(color='royalblue', width=2)
        ))
        
        # Zufallslinie hinzufügen
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Zufälliges Raten',
            line=dict(color='firebrick', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Wie gut das System Zahlungsverhalten vorhersagen kann',
            xaxis_title='Anteil fälschlich abgelehnter guter Kunden',
            yaxis_title='Anteil richtig erkannter zuverlässiger Kunden',
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Confusion Matrix
        y_pred_binary = (y_scores > 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred_binary)
        
        # Confusion Matrix als Heatmap mit verständlicheren Beschriftungen
        fig2 = px.imshow(
            cm,
            labels=dict(x="Vorhersage des Systems", y="Tatsächliches Verhalten", color="Anzahl Personen"),
            x=['Wird Probleme haben', 'Wird pünktlich zahlen'],
            y=['Hat Probleme gehabt', 'Hat pünktlich gezahlt'],
            text_auto=True,
            color_continuous_scale='Blues'
        )
        
        fig2.update_layout(
            title='Vorhersage-Genauigkeitstabelle',
            height=400
        )
        
        # Berichte
        report = classification_report(y_test, y_pred_binary, output_dict=True)
        
        # Report als Tabelle formatieren mit alltagssprachlichen Begriffen
        report_table = html.Div([
            html.H5("Wie gut ist die Vorhersage?"),
            html.Table([
                html.Thead(
                    html.Tr([html.Th("Personengruppe"), html.Th("Treffsicherheit"), html.Th("Erkennungsrate"), html.Th("Gesamtwert"), html.Th("Anzahl Fälle")])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td("Personen mit Zahlungsproblemen"),
                        html.Td(f"{report['0']['precision']:.3f}"),
                        html.Td(f"{report['0']['recall']:.3f}"),
                        html.Td(f"{report['0']['f1-score']:.3f}"),
                        html.Td(f"{report['0']['support']}")
                    ]),
                    html.Tr([
                        html.Td("Pünktliche Zahler"),
                        html.Td(f"{report['1']['precision']:.3f}"),
                        html.Td(f"{report['1']['recall']:.3f}"),
                        html.Td(f"{report['1']['f1-score']:.3f}"),
                        html.Td(f"{report['1']['support']}")
                    ]),
                    html.Tr([
                        html.Td("Gesamtgenauigkeit"),
                        html.Td(f"{report['accuracy']:.3f}", colSpan=4)
                    ])
                ])
            ], className="table table-striped")
        ])
        
        # Erläuterung der Metriken für Laien
        metrics_explanation = html.Div([
            html.H5("Was bedeuten diese Zahlen?", className="mt-3"),
            html.P([
                html.Strong("Treffsicherheit"), ": Wenn das Modell jemanden als 'zuverlässig' einstuft, wie oft stimmt das wirklich? (1.0 = perfekt)"
            ]),
            html.P([
                html.Strong("Erkennungsrate"), ": Wie viele gute Kunden werden tatsächlich erkannt und nicht fälschlich abgelehnt? (1.0 = perfekt)"
            ]),
            html.P([
                html.Strong("Gesamtwert"), ": Eine Kombination aus beiden obigen Werten. Höher ist besser. (1.0 = perfekt)"
            ]),
            html.P([
                html.Strong("Die Grafik oben"), ": Je mehr die blaue Linie nach oben links geht, desto besser ist das System. Die rote gestrichelte Linie zeigt, wie gut blindes Raten wäre."
            ]),
            html.P([
                html.Strong("Die Tabelle in der Mitte"), ": Zeigt, wie oft das System richtig und falsch lag. Idealerweise sollten die Zahlen oben links und unten rechts groß sein."
            ])
        ])
        
        return html.Div([
            dcc.Graph(figure=fig),
            dcc.Graph(figure=fig2),
            report_table,
            metrics_explanation
        ]), title, metrics_div