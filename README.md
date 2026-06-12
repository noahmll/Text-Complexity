# Text-Complexity

Für **DIS22a** — Analyse der Textkomplexität wissenschaftlicher Preprints (bioRxiv).
TH Köln · Mai 2026

---

## Forschungsfrage

> Verändert sich die Textkomplexität wissenschaftlicher Preprints zwischen Versionen systematisch?
> Werden Papers von der ersten Preprint-Version (v1) über weitere Revisionen bis zur publizierten Fassung sprachlich **einfacher oder komplexer** — und gibt es dabei Muster, die über einzelne Papers hinaus stabil sind?

### Hypothesen

| | Hypothese |
|---|---|
| **H1** | Die finale Fassung ist sprachlich einfacher als v1 — redaktionelle Überarbeitung glättet die Sprache. |
| **H2** | Spätere Preprint-Versionen werden sukzessive einfacher — Autoren passen den Text an Reviewer-Feedback an. |

---

## Projektstruktur

```
complexity_over_versions.py   # Hauptanalyse: Metriken über die Versionshistorie (v1 → vN)
dois.txt                      # 958 DOIs der untersuchten Preprints
requirements.txt
plots/                        # (gitignored) generierte Abbildungen
data/                         # (gitignored) Rohdaten
    xml/                      # JATS-XML je Paper-Version, Dateiname …_vN.xml
    json/                     # JSON-Export je Paper-Version
legacy/                       # Frühere Projektphase, nicht mehr Teil der Analyse
    complexity_over_text.py   # Komplexität *innerhalb* eines Textes (0–100 % Position)
    text_extraction.py        # JSON laden / Volltext bereinigen (nur vom Legacy-Skript genutzt)
```

> **Legacy:** In der ersten Projektphase wurde — solange noch keine Versionsdaten vorlagen —
> die Komplexitätskurve *innerhalb* einzelner Papers über die relative Textposition (0–100 %)
> analysiert. Dieser Code liegt jetzt in `legacy/` und ist nicht mehr Teil der aktuellen
> Auswertung. Die Forschungsfrage wird ausschließlich über die **Versionsentwicklung** beantwortet.

---

## Installation

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt_tab brown
```

## Ausführen

```bash
python complexity_over_versions.py                 # alle gültigen Papers
python complexity_over_versions.py --n 200         # 200 zufällige Papers
python complexity_over_versions.py --n 200 --seed 42   # reproduzierbares Sample
```

**Abstract und Volltext (Body) werden getrennt analysiert.** Das Skript erzeugt die
Übersicht im Root plus je sechs Sichten pro Metrik-Gruppe und Textquelle:

```
plots/
├── papers_per_version.png          # Anzahl Papers je Version (ganzer Korpus)
├── body/                           # Volltext (ohne Abstract, ohne Refs/Figures/Tables/Formeln)
│   ├── simple_metrics/             # Wortzahl, Sections, Figures, Tables, Refs, Satzlänge
│   │   ├── normalized_over_versions.png    # % Änderung ggü. v1
│   │   ├── stepwise_over_versions.png      # % Änderung ggü. Vorgänger
│   │   ├── direction_over_versions.png     # Richtungs-Anteil je Schritt
│   │   ├── distribution.png                # Verteilung der Netto-Änderung
│   │   ├── scatter.png                     # Netto-Änderung × Volatilität
│   │   └── baseline_change.png             # v1-Ausgangswert × Netto-Änderung
│   └── complexity/                 # Flesch, Wortlänge, TTR, Rare-Word, Nominalisierung, Satzlänge
│       └── (dieselben sechs Dateien)
└── abstract/                       # nur Komplexität (Strukturmetriken im Abstract sinnlos)
    └── complexity/
        └── (dieselben sechs Dateien)
```

**Sechs Sichten auf dieselben Metriken:**

- **Kumulativ (`normalized_over_versions.png`)** — % Änderung gegenüber **v1**: dünne Einzeltrajektorien
  je Paper, überlagert von der fetten Mittelwertlinie über alle Papers und einem ± 1 SD Band
  (Nulllinie = v1). Zeigt den **Netto-Effekt** der gesamten Überarbeitung (→ H1).
- **Schrittweise (`stepwise_over_versions.png`)** — % Änderung gegenüber der **direkten Vorgängerversion**
  (v1→v2, …, v5→v6) als Mittelwert ± 1 SD je Übergang. Zeigt, **wo** im Revisionsprozess
  Änderung passiert (→ H2 „sukzessive").
- **Richtung (`direction_over_versions.png`)** — gestapelte Balken je Übergang mit dem Anteil Papers, die
  steigen / ~gleich bleiben / fallen (`|Δ| < 1 %` gilt als unverändert). Für Komplexitäts-
  metriken nach Bedeutung als *einfacher / unverändert / komplexer* gelabelt, für einfache
  Metriken neutral als *gestiegen / unverändert / gesunken*.
- **Verteilung (`distribution.png`)** — Histogramm der **Netto-Änderung pro Paper**
  (letzte Version vs. v1) mit Median-Linie und Anteil einfacher/komplexer. Deckt auf, ob ein
  Mittelwert nahe 0 echte Stabilität bedeutet oder zwei gegenläufige Teilpopulationen verdeckt.
- **Scatter (`scatter.png`)** — ein Punkt je Paper: x = **Netto-Änderung** (letzte vs. v1),
  y = **Volatilität** = RMSE der Residuen um den linearen Trend (Wert ~ Versionsindex), in
  % des v1-Werts (0 = exakt linear, größer = erratischer). Der lineare Fit absorbiert die
  Gesamtrichtung, sodass y nur das „Gezappel" um den Trend misst — weitgehend unabhängig von
  der Netto-Änderung. Läuft über **alle** Papers (beliebige Versionszahl).
- **Baseline-Change (`baseline_change.png`)** — ein Punkt je Paper: x = **v1-Ausgangswert**,
  y = **Netto-Änderung** (% ggü. v1), mit Fit-Linie und Pearson r. Negative Steigung =
  **Regression zur Mitte** (hoch startende Papers fallen, niedrig startende steigen →
  Konvergenz). Über **alle** Papers.

---

## Datenbasis

Rohdaten liegen unter `data/` (gitignored) und stammen aus bioRxiv im **JATS-XML**-Format.

- `data/xml/` — eine XML-Datei je Paper-**Version**, benannt `…_vN.xml`
- Versionszugehörigkeit wird über das `_vN`-Suffix im Dateinamen gruppiert
- `dois.txt` — 958 DOIs der untersuchten Preprints

**Filterregeln:** Geladen werden alle Papers mit **lückenlosen Versionen ab v1** (≥ 2).
Versionen mit < 300 Zeichen Body-Text werden verworfen. Die **versions-ausgerichteten
Sichten** (kumulativ, schrittweise, Richtung, Verteilung) nutzen daraus die Teilmenge mit
**genau v1–v6** (`N_VERSIONS = 6`, konstantes n je Version); **Scatter und Baseline-Change**
laufen über **alle** Papers (beliebige Versionszahl). Wortzahl-Mindestschwelle für die
Metrik-Berechnung: 200 Wörter (Body), 100 Wörter (Abstract).

**XML-Verarbeitung:** **Abstract und Body werden getrennt** rekursiv eingesammelt. Im Body
werden `fig`, `table-wrap`, `supplementary-material`, `disp-formula` und `inline-formula`
übersprungen, damit Abbildungs-, Tabellen- und Formelinhalte die Sprachmetriken nicht
verzerren. **Inline-Zitationen** (`<xref ref-type="bibr">`, z. B. „12" oder „Smith et al.")
werden in beiden Quellen entfernt. Die Referenzliste liegt im JATS unter `<back>` und ist
damit ohnehin nicht im Body. Zusätzlich werden je Version Strukturzähler erfasst (Body):
Top-Level-Abschnitte, Abbildungen, Tabellen, Referenzen.

---

## Metriken

Pro Version werden je Paper folgende Werte berechnet und als **% Änderung relativ zu v1**
(v1 ≙ 0 %) ausgedrückt. Über alle Papers hinweg wird je Version der **Mittelwert ± SD**
aggregiert und zusammen mit den Einzeltrajektorien dargestellt.

### Einfache Textmetriken
| Metrik | Basis |
|---|---|
| **Word Count** | Wörter gesamt |
| **Top-level Sections** | Anzahl `sec`-Elemente im Body |
| **Figures / Tables / References** | Anzahl `fig` / `table-wrap` / `ref` |
| **Ø Satzlänge** | Wörter / Sätze |

### Linguistische Komplexität
| Metrik | Formel / Basis | Richtung |
|---|---|---|
| **Flesch Reading Ease** | 206.8 − 1.015·(W/S) − 84.6·(Syl/W) | höher = leichter |
| **Ø Wortlänge** | Zeichen / Wörter | höher = schwierigere Wörter |
| **Type-Token Ratio** | Unique Tokens / Tokens gesamt | höher = reicheres Vokabular |
| **Rare-Word Rate** | Wörter ∉ Top-10 k (Brown-Korpus) / Gesamt | höher = mehr Fachvokabular |
| **Nominalisierungsrate** | Wörter mit -tion/-ment/-ity/… / Gesamt | höher = formaler Schreibstil |

Die Metriken werden ohne externe Lesbarkeits-Library selbst berechnet (nur `nltk` für
Satz-Tokenisierung und das Brown-Korpus).

---

## Bekannte Einschränkungen

- Silbenzählung (Flesch) ist eine Vokalgruppen-Heuristik, kein Lexikon.
- Nominalisierung wird rein über Suffixe erkannt (kein POS-Tagging in der Versionsanalyse).
- Die Rare-Word-Rate ist `NaN`, falls das NLTK-Brown-Korpus nicht verfügbar ist.
