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

## Aktueller Stand

Da Versionsdaten (v1/v2/final) noch nicht vorliegen, wird als **erster methodischer Schritt** die Komplexitätskurve *innerhalb* einzelner Papers analysiert: Wie verhalten sich die 6 Metriken entlang des Dokumentverlaufs (0–100 % Textposition)? Dies erlaubt, das Metrik-Verhalten in wissenschaftlichen Texten zu verstehen, bevor Versionsvergleiche möglich sind.

---

## Projektstruktur

```
complexity_over_text.py   # Hauptskript: 6 Metriken als Sliding-Window über den Textverlauf
src/
    text_extraction.py    # JSON laden, Volltext bereinigen, Abschnitt extrahieren
SampleData/
    output/               # 499 JSON-Dateien (bioRxiv-Preprints)
    metadata.json
dois.txt                  # 959 DOIs für Versionsdaten (beantragt)
requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng brown
```

## Ausführen

```bash
python complexity_over_text.py              # Standard: 20 Dokumente
python complexity_over_text.py --n 50       # 50 zufällige Dokumente
python complexity_over_text.py --n all      # alle 499 Dokumente
python complexity_over_text.py --n 30 --seed 7   # reproduzierbares Sample
```

Das Skript erzeugt `complexity_over_text.png` mit 6 Subplots (Metrik × Textposition).

---

## Datenbasis

**499 JSON-Dateien** in `SampleData/output/`, je ein Paper. Relevante Felder pro Datei:

- `text` — Volltext (pipe-delimited, wird von `_deep_clean()` bereinigt)
- `metadata.pub_metadata.preprint_abstract` — Abstracttext
- `metadata.pub_metadata.preprint_category` — Fachgebiet (z. B. `"neuroscience"`, `"bioinformatics"`)
- `metadata.pub_metadata.published_journal` — Zieljournal
- `metadata.pub_metadata.preprint_date` / `published_date` — Zeitstempel

**Top-Fachgebiete:** Neuroscience (96), Bioinformatics (43), Microbiology (34), Cell Biology (30), Cancer Biology (29)

### Bekannte Datenprobleme

- Volltext enthält Zeilennummern und Pipe-Formatierung vom PDF-Extraktor
- Referenzliste ist **nicht** vom Fließtext getrennt → verzerrt Satzlänge und Wortlänge
- Inline-Zitationen (`(Smith et al., 2021)`) erscheinen als Fließtext
- Versionsdaten (v1, v2, publizierte Fassung) liegen noch nicht vor

---

## Methodik: Sliding-Window-Analyse

Das Skript berechnet 6 Komplexitätsmetriken in einem satzweise aufgebauten Sliding Window über den bereinigten Volltext jedes Papers.

| Parameter | Wert | Bedeutung |
|---|---|---|
| `WINDOW_WORDS` | 200 | Wörter pro Fenster |
| `STEP_WORDS` | 50 | Schrittweite (75 % Overlap) |
| `N_BINS` | 50 | Positions-Buckets (0–100 %) |
| `MIN_WORDS` | 800 | Mindestwörter nach Bereinigung |

**Pipeline:**
1. `_deep_clean()` — Pipe-Rows, Zeilennummern, Referenzlisten, Abbildungslegenden & Zitationen entfernen
2. Satzweise Fenster aufbauen → Relative Position = Fenstermitte / Textlänge
3. 6 Metriken je Fenster berechnen, Bin-Index zuweisen
4. Mittelwert ± SD je Bin über alle n Papers, Glättung mit rollendem 5-Bin-Mittelwert

---

## Komplexitätsmetriken

`complexity_over_text.py` berechnet folgende **6 Metriken**:

| Metrik | Formel / Basis | Richtung |
|---|---|---|
| **Flesch Reading Ease** | 206.8 − 1.015·(W/S) − 84.6·(Syl/W) | höher = leichter (wiss. Text ≈ 0–30) |
| **Ø Satzlänge** | Wörter / Sätze | höher = komplexer |
| **Ø Wortlänge** | Zeichen / Wörter | höher = schwierigere Wörter |
| **Type-Token Ratio** | Unique Tokens / Tokens gesamt | höher = reicheres Vokabular |
| **Rare-Word Rate** | Wörter ∉ Top-10 k / Gesamt | höher = mehr Fachvokabular |
| **Nominalisierungsrate** | Nomen mit -tion/-ment/-ity/… / Gesamt | höher = formaler Schreibstil |

> **Hinweis:** Die README einer früheren Projektphase listete 12 Metriken (7× `textstat`, 5× linguistisch). Das Skript implementiert derzeit die obigen 6 selbst berechneten Metriken ohne externe Lesbarkeits-Libraries.

---

## Datenwünsche

Folgende Daten würden die Analyse erheblich verbessern:

1. **Mehrere Preprint-Versionen (v1, v2, …)** — Kern der Forschungsfrage; 959 DOIs bereits angefragt
2. **Strukturierter Text mit Abschnittslabels** — Intro / Methods / Results / Discussion statt Textblob
3. **Referenzliste getrennt vom Fließtext** — Literaturangaben verzerren Satz- und Wortlänge
4. **Text ohne Inline-Zitationen** — `(Smith et al., 2021)` verzerrt mehrere Metriken
