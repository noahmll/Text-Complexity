# Text-Complexity
Für DIS22a — Analyse und Vergleich von Textkomplexität wissenschaftlicher Abhandlungen (bioRxiv-Preprints).

---

## Datenbasis

499 JSON-Dateien in `SampleData/output/`, je ein Paper. Relevante Felder pro Datei:

- `text` — Volltext (pipe-delimited, muss bereinigt werden)
- `metadata.pub_metadata.preprint_abstract` — sauberer Abstracttext
- `metadata.pub_metadata.preprint_category` — Fachgebiet (z.B. "neuroscience")
- `metadata.pub_metadata.published_journal` — Zieljournal
- `metadata.pub_metadata.preprint_date` / `published_date` — Zeitstempel
- `tables`, `figures`, `formulas` — extrahierte Strukturelemente

Bekannte Datenprobleme:
- `local_pdf_path` ist absolut statt relativ
- Volltext enthält Zeilennummern und Pipe-Formatierung vom PDF-Extraktor
- Inline-Zitationen und Referenzliste sind nicht vom Fließtext getrennt

---

## Projektstruktur

```
src/
    text_extraction.py   # JSON laden, Volltext bereinigen, Abschnitt extrahieren
    complexity.py        # analyze_complexity(text) -> dict
prototype.py             # Vergleich Abstract vs. Introduction an einem Paper
requirements.txt
```

### Installation

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng
```

### Ausführen

```bash
python prototype.py
```

---

## Komplexitätsmetriken

`analyze_complexity(text: str) -> dict` gibt folgende Metriken zurück:

**Tier 1 — Klassische Lesbarkeitsformeln (`textstat`)**

| Metrik | Skala | Richtung |
|---|---|---|
| Flesch Reading Ease | 0–100 | höher = leichter |
| Flesch-Kincaid Grade | US-Schulniveau | höher = schwerer |
| Gunning Fog | US-Schulniveau | höher = mehr Jargon |
| SMOG Index | US-Schulniveau | höher = schwerer (min. 30 Sätze nötig) |
| Coleman-Liau | US-Schulniveau (zeichenbasiert) | höher = schwerer |
| ARI | US-Schulniveau (zeichenbasiert) | höher = schwerer |
| Dale-Chall | 0–16 | höher = mehr Fachvokabular |

**Tier 2 — Linguistische Merkmale (pure Python + NLTK)**

| Metrik | Skala | Richtung |
|---|---|---|
| Type-Token Ratio | 0–1 | höher = reicheres Vokabular |
| Avg. Satzlänge | Wörter/Satz | höher = längere Sätze |
| Avg. Wortlänge | Zeichen/Wort | höher = längere Wörter |
| Complex Word Ratio | 0–1, Anteil ≥3-Silben-Wörter | höher = mehr Fachjargon |
| Lexical Density | 0–1, Inhaltswörter/Gesamt | höher = informationsdichter |

Referenzwerte: Roman ~Flesch 70–80, Grade 7–9 / Zeitung ~Flesch 60–70, Grade 10–12 / wissenschaftlicher Text ~Flesch 0–30, Grade 15–20+

---

## Offene Datenwünsche

Folgende Informationen fehlen und würden die Analyse erheblich verbessern:

1. **Strukturierter Text mit Abschnittslabels** — statt eines Textblobs einzelne Felder für Introduction, Methods, Results, Discussion, References. Ohne das mischt jede Analyse sprachlich sehr unterschiedliche Texttypen.

2. **Referenzliste getrennt vom Fließtext** — Literaturangaben verzerren derzeit Satzlänge, Wortlänge und Dale-Chall-Score erheblich.

5. **Text ohne Inline-Zitationen** — `(Smith et al., 2021)` erscheint derzeit als Fließtext und verzerrt mehrere Metriken.

7. **Mehrere Preprint-Versionen (v1, v2, ...)** — würde erlauben zu messen, ob Peer Review Texte verständlicher oder komplexer macht.
