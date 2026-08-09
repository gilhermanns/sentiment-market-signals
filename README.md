# Sentiment Market Signals: Robuste Analyse von Finanzmarkt-Sentiment

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Ein umfassendes End-to-End-Projekt zur **rigorosen Bewertung der prädiktiven Kraft von Finanzmarkt-Sentiment** aus Schlagzeilen. Dieses Tool demonstriert eine robuste Methodik zur Signalbewertung, die für **Private Banking, Wealth Management und institutionelle Anlagestrategien** unerlässlich ist, um fundierte Entscheidungen zu treffen und Risiken zu managen.

## Business Relevanz

Für Finanzexperten im **Private Banking, Wealth Management und institutionellen Asset Management** ist die Fähigkeit, quantitative Signale kritisch zu bewerten, von entscheidender Bedeutung. Dieses Projekt bietet einen Rahmen für:

-   **Due Diligence bei Anlagestrategien**: Objektive Bewertung von Strategien, die auf unstrukturierten Daten (wie Nachrichten-Sentiment) basieren, um Überanpassung zu vermeiden und echte Alpha-Quellen zu identifizieren.
-   **Risikomanagement**: Verständnis der Grenzen und potenziellen Fallstricke von Sentiment-basierten Modellen, um Fehlallokationen und unerwartete Verluste zu verhindern.
-   **Informierte Kundenberatung**: Bereitstellung einer fundierten Grundlage für Gespräche mit Kunden über die Rolle von Sentiment-Analysen in ihren Portfolios.
-   **Entwicklung neuer Produkte**: Aufbau modularer und testbarer Pipelines für die Integration alternativer Datenquellen in Anlagestrategien.

Dieses Projekt unterstreicht die Bedeutung einer methodisch sauberen Analyse, selbst wenn die Ergebnisse die anfänglichen Hypothesen nicht bestätigen. Es zeigt die Fähigkeit, komplexe Datenpipelines zu entwerfen, zu implementieren und kritisch zu hinterfragen – eine Kernkompetenz im modernen Finanzwesen.

## Motivation (Angewandte Perspektive)

Die Frage, ob Nachrichten-Sentiment tatsächlich Aktienrenditen oder Volatilität vorhersagen kann, ist im Finanzwesen von großem Interesse. Oftmals führen scheinbar vielversprechende Korrelationen in historischen Daten zu enttäuschenden Ergebnissen in der Praxis, da sie auf Überanpassung oder Look-Ahead-Bias beruhen. Dieses Projekt stellt die ehrlichere und praktisch relevantere Frage:

**Wie kann man die prädiktive Kraft von Sentiment-Signalen methodisch sauber testen, um Scheinkorrelationen zu vermeiden und die Robustheit der Analyse zu gewährleisten?**

Wir bauen eine vollständige NLP-zu-Statistik-Pipeline auf, die Schlagzeilen erfasst, Sentiment auf zwei Arten bewertet, diese mit realen Marktdaten unter strenger No-Lookahead-Regel abgleicht und das resultierende Signal dann umfassenden Tests unterzieht: Lag-Regressionen, Volatilitätstests, Event Studies und einen kostenbereinigten Backtest. Das Ergebnis – dass für diesen Datensatz kein zuverlässiger prädiktiver Wert gefunden wurde – ist selbst ein wichtiges und publizierbares Ergebnis, das die Integrität der Methodik bestätigt und nicht deren Scheitern darstellt.

## Wichtiger Hinweis zur Datenintegrität

**Die in diesem Repository verwendeten Schlagzeilentexte sind synthetisch generiert, nicht von echten Nachrichtenquellen gescrapt.** Da ein Live-Finanznachrichtenarchiv in dieser Entwicklungsumgebung nicht zuverlässig erreichbar war, generiert `src/sentiment_signals/headlines.py` Vorlagen für positive, negative und neutrale Schlagzeilen (mit Vokabularvariationen) und ordnet diese realen Tickern und Handelsdaten zu. Das Sentiment der Schlagzeilen wird **konstruktionsbedingt unabhängig von den realisierten Renditen** zugewiesen. Dies ist ein bewusstes Placebo-Design, um sicherzustellen, dass jedes später gefundene "prädiktive" Ergebnis auf einen echten Pipeline-Fehler zurückzuführen wäre und nicht auf heimlich informative Schlagzeilen. Die **Marktpreisdaten sind real** (siehe unten), und die **Pipeline selbst** – Erfassung → Bereinigung → Bewertung → Abgleich → statistische Tests – ist das eigentliche Ergebnis. Die Pipeline ist modular aufgebaut, sodass `generate_market_headlines` einfach durch einen Feed mit echten Nachrichten (z.B. eine Nachrichten-API oder ein gelabelter Datensatz wie die [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank)) ersetzt werden kann, ohne dass der Rest des Codes betroffen ist.

## Methodik

1.  **Sentiment-Bewertung, zwei Ansätze** (`lexicon.py`, `classifier.py`):
    -   Ein manuell erstelltes Finanzlexikon: Zählt positive/negative Finanzbegriffe pro Schlagzeile (mit einfacher Negationsumkehr) und bewertet im Bereich `[-1, 1]`. Keine Downloads, vollständig überprüfbar.
    -   Ein TF-IDF + Logistische Regression Klassifikator, trainiert auf einem synthetisch gelabelten Korpus von 1.500 Schlagzeilen. Der kontinuierliche Score ist `P(positiv) - P(negativ)`.
    -   **Dokumentierter Upgrade-Pfad**: Der Austausch durch [FinBERT](https://huggingface.co/ProsusAI/finbert) ist ein Drop-in-Ersatz hinter derselben `.predict_label` / `.score`-Schnittstelle, was die Modularität und Erweiterbarkeit des Projekts unterstreicht.
2.  **No-Lookahead-Abgleich** (`alignment.py`): Eine zentrale Funktion, `build_lagged_dataset`, führt alle Vorwärtsverschiebungen intern durch und lehnt nicht-positive Horizonte ab, um sicherzustellen, dass keine zukünftigen Informationen in die Analyse einfließen. Dies ist entscheidend für die Integrität von Backtests.
3.  **Lag-Regressionen** (`lag_regression.py`): OLS-Regressionen der zukünftigen Renditen auf das Sentiment des Tages t, mit Newey-West (HAC) Standardfehlern zur Berücksichtigung von Autokorrelationen.
4.  **Volatilitätsregression** (`volatility.py`): OLS-Regression der nächsten Tagesvolatilität auf den Betrag des Sentiments und die Sentiment-Dispersion.
5.  **Event Study** (`event_study.py`): Analyse abnormaler Renditen um Ereignisse mit starkem Sentiment, um potenzielle Marktreaktionen zu identifizieren.
6.  **Backtest** (`backtest.py`): Eine Sentiment-gesteuerte Long/Flat/Short-Strategie, die mit einer Buy-and-Hold-Strategie verglichen wird, unter Berücksichtigung realistischer Transaktionskosten.
7.  **Dashboard** (`dashboard.py`): Eine statische HTML-Seite zur Visualisierung von Schlagzeilen, Renditen und Schlusskursen, die eine schnelle Überprüfung der Daten ermöglicht.

## Projektergebnisse & Erkenntnisse

Die Analyse, durchgeführt über 5 Jahre realer täglicher Preisdaten (2021-07-02 bis 2026-07-02) für `AAPL`, `MSFT`, `TSLA`, `JPM`, `XOM` und 4.351 synthetische Schlagzeilen, liefert folgende wichtige Erkenntnisse:

-   **Klassifikator-Leistung**: Der Sentiment-Klassifikator erreicht eine hohe Genauigkeit auf synthetischen Daten, was die korrekte Implementierung der Pipeline bestätigt und einen klaren Upgrade-Pfad für reale Daten aufzeigt.
-   **Lag-Regressionen**: Es wurde kein statistisch signifikanter prädiktiver Wert des Sentiments für zukünftige Renditen über verschiedene Zeithorizonte festgestellt. Dies ist ein wichtiges, ehrliches Ergebnis, das die Robustheit der Testmethodik unterstreicht.
-   **Volatilitäts-Regression & Event Study**: Auch hier zeigten sich keine robusten, statistisch signifikanten Zusammenhänge zwischen Sentiment und Volatilität oder abnormalen Renditen, was die Notwendigkeit einer kritischen Bewertung von Sentiment-Signalen betont.
-   **Backtest-Ergebnisse**: Die Sentiment-gesteuerte Strategie konnte die Buy-and-Hold-Strategie nach Berücksichtigung von Transaktionskosten in keinem der getesteten Ticker übertreffen. Dies bestätigt die Ergebnisse der Lag-Regressionen und unterstreicht, dass Signale ohne echte prädiktive Kraft keine Überrenditen generieren können.

Diese Ergebnisse, obwohl sie keine positive prädiktive Kraft des Sentiments zeigen, sind für Finanzprofis äußerst wertvoll. Sie demonstrieren die Fähigkeit, eine komplexe quantitative Analyse durchzuführen, die Ergebnisse kritisch zu interpretieren und die Grenzen von Modellen zu verstehen – Fähigkeiten, die im Private Banking und Wealth Management unerlässlich sind.

*(Beispiel-Charts und Tabellen, die die Klassifikator-Leistung, Lag-Regressionen, Event-Study-CARs und Backtest-Equity-Kurven zeigen, würden hier eingebettet.)*

## Projektstruktur

```text
/sentiment-market-signals/
├── README.md               # Projektdokumentation
├── requirements.txt        # Python-Abhängigkeiten
├── src/sentiment_signals/  # Importierbares Paket mit Kernlogik
│   ├── market_data.py      # Marktpreisdaten-Beschaffung
│   ├── headlines.py        # Synthetische Schlagzeilen-Generierung
│   ├── lexicon.py          # Lexikon-basierte Sentiment-Bewertung
│   ├── classifier.py       # ML-basierte Sentiment-Klassifikation
│   ├── alignment.py        # No-Lookahead-Datenabgleich
│   ├── lag_regression.py   # Lag-Regressionen für prädiktive Analyse
│   ├── volatility.py       # Volatilitäts-Regressionen
│   ├── event_study.py      # Event-Study-Analyse
│   ├── backtest.py         # Strategie-Backtesting
│   └── dashboard.py        # HTML-Dashboard-Generierung
├── tests/                  # Pytest-Suite
├── scripts/                # Skripte zur Pipeline-Ausführung und Chart-Generierung
├── reports/                # Generierte CSV-Outputs und HTML-Dashboard
└── docs/img/               # Eingebettete SVG-Charts
```

## Erste Schritte

### Installation
1.  Klonen Sie das Repository:
    ```bash
    git clone https://github.com/gilhermanns/sentiment-market-signals.git
    cd sentiment-market-signals
    ```
2.  Installieren Sie die Abhängigkeiten:
    ```bash
    pip install -r requirements.txt
    ```
3.  Führen Sie die vollständige Analyse aus:
    ```bash
    python scripts/run_pipeline.py
    ```

## Lizenz & Haftungsausschluss

Dieses Projekt ist unter der MIT-Lizenz lizenziert. Es dient Bildungs- und Forschungszwecken im Bereich Quantitative Finance. Die vorgestellten Modelle und Ergebnisse dienen nur zur Veranschaulichung und stellen keine Finanzberatung dar oder garantieren eine reale Performance. Bei der Handhabung von Finanzinstrumenten ist stets professionelles Urteilsvermögen und eine gründliche Due Diligence erforderlich.

---

*Entwickelt mit Unterstützung von Claude Code (Anthropic).*
