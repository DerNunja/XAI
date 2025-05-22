# Wörterbuch mit vereinfachten Begriffen für alle Spalten im Datensatz
simplified_terms = {
    # Allgemeine Risikobewertung
    "ExternalRiskEstimate": "Kredit-Score",
    "NumSatisfactoryTrades": "Pünktlich bezahlte Rechnungen",
    "NetFractionRevolvingBurden": "Kreditkartenauslastung",
    "NetFractionInstallBurden": "Ratenkreditauslastung",
    "NumRevolvingTradesWithBalance": "Anzahl Kreditkarten mit Schulden",
    "NumInstallTradesWithBalance": "Anzahl laufende Ratenkredite",
    "NumBank2NatlTradesWHighUtilization": "Übernutzte Kreditkonten",
    "PercentTradesNeverDelq": "Prozent nie verspäteter Zahlungen",
    
    # Zahlungshistorie
    "MSinceMostRecentDelq": "Monate seit letzter Zahlungsverspätung",
    "MaxDelq2PublicRecLast12M": "Maximale Zahlungsverspätung (letzte 12 Monate)",
    "MaxDelqEver": "Längste Zahlungsverspätung jemals",
    "AverageMInFile": "Durchschn. Monate in der Kreditakte",
    "NumTotalTrades": "Gesamtzahl Kreditverträge",
    "NumTradesOpeninLast12M": "Neue Kredite (letzte 12 Monate)",
    "PercentInstallTrades": "Anteil Ratenkredite",
    
    # Suchverhalten
    "MSinceMostRecentInqexcl7days": "Monate seit letzter Kreditanfrage",
    "NumInqLast6M": "Kreditanfragen (letzte 6 Monate)",
    "NumInqLast6Mexcl7days": "Kreditanfragen ohne letzte Woche",
    
    # Kreditbüro-Einträge
    "MSinceMostRecentBankcard": "Monate seit letzter Kreditkarte",
    "MSinceMostRecentTradeOpen": "Monate seit letzter Kreditaufnahme",
    "NumTradesOpeninLast12M": "Neue Kredite (letzte 12 Monate)",
    "NumBank2NatlTradesWHighUtilization": "Übernutzte Kreditkonten",
    
    # Zielwerte
    "RiskPerformance": "Zahlungsverlässlichkeit",
    "Good": "Zuverlässiger Kunde",
    "Bad": "Zahlungsprobleme",
    "target": "Zahlungsverlässlichkeit",
    "1": "Zuverlässiger Kunde",
    "0": "Zahlungsprobleme",
    
    # Kategorien
    "Niedrig": "Schlecht",
    "Mittel": "Mittel", 
    "Hoch": "Gut",
    "Schlechtes Risiko": "Zahlungsprobleme wahrscheinlich",
    "Gutes Risiko": "Zuverlässiger Kunde",
    
    # Für Confusion Matrix
    "Schlechte Kreditwürdigkeit": "Zahlungsprobleme",
    "Gute Kreditwürdigkeit": "Zuverlässiger Kunde",

    # Konto- & Kredithistorie
    "MSinceOldestTradeOpen": "Monate seit ältestem Kredit",
    "MSinceMostRecentTradeOpen": "Monate seit letzter Kreditaufnahme",

    # Schwere Zahlungsverspätungen / Negativ­einträge
    "NumTrades60Ever2DerogPubRec": "Anzahl 60-Tage-Zahlungsverzüge",
    "NumTrades90Ever2DerogPubRec": "Anzahl 90-Tage-Zahlungsverzüge",

    # Kontostände / Salden
    "NumRevolvingTradesWBalance": "Anzahl Kreditkarten mit Schulden",
    "NumInstallTradesWBalance":  "Anzahl laufende Ratenkredite",
    "PercentTradesWBalance":     "Menga an Verträge mit Schulden",
}

# Funktion zum Umsetzen der vereinfachten Begriffe
def simplify_term(term):
    for key, value in simplified_terms.items():
        if key in term:
            return term.replace(key, value)
    return term

# Funktion für verständliche Erklärungen zu Features
def get_feature_explanation(feature):
    explanations = {
        "ExternalRiskEstimate": "Je höher dieser Wert, desto besser Ihre allgemeine Kreditbewertung",
        "NumSatisfactoryTrades": "Wie viele Kredite und Rechnungen Sie pünktlich bezahlt haben",
        "NetFractionRevolvingBurden": "Wie stark Ihre Kreditkarten im Verhältnis zum Limit belastet sind",
        "NetFractionInstallBurden": "Wie viel Ihrer Ratenkredite noch offen sind",
        "NumRevolvingTradesWithBalance": "Anzahl Ihrer Kreditkarten mit offenem Saldo",
        "NumInstallTradesWithBalance": "Anzahl Ihrer laufenden Ratenzahlungen",
        "NumBank2NatlTradesWHighUtilization": "Anzahl Kreditkonten, die Sie fast bis zum Limit ausgereizt haben",
        "PercentTradesNeverDelq": "Prozent Ihrer Kredite ohne jegliche Zahlungsverzögerungen",
        "MSinceMostRecentDelq": "Wie viele Monate seit Ihrer letzten Zahlungsverspätung vergangen sind",
        "MaxDelq2PublicRecLast12M": "Längste Zahlungsverspätung im letzten Jahr",
        "MaxDelqEver": "Längste Zahlungsverspätung in Ihrer gesamten Kredithistorie",
        "NumTotalTrades": "Gesamtzahl Ihrer Kreditverträge, Kreditkarten und Darlehen",
        "NumTradesOpeninLast12M": "Wie viele neue Kredite Sie im letzten Jahr aufgenommen haben",
        "MSinceMostRecentInqexcl7days": "Wie lange es her ist, dass jemand Ihre Kreditwürdigkeit überprüft hat",
        "NumInqLast6M": "Wie oft Ihre Kreditwürdigkeit in den letzten 6 Monaten geprüft wurde",
        "MSinceMostRecentBankcard": "Wie lange es her ist, dass Sie eine neue Kreditkarte bekommen haben"
    }
    return explanations.get(feature, "Ein Faktor für Ihre Kreditbewertung")
