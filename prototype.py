"""
Prototype: Text Complexity Analysis for one bioRxiv paper.

Compares:  Abstract  vs.  Introduction section of full text
Paper:     10.1101_2022.03.14.484320.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from text_extraction import load_paper, clean_text, get_abstract, extract_section
from complexity import analyze_complexity

# --- Config ---
DATA_DIR = Path(__file__).parent / "SampleData" / "output"
TARGET_FILE = DATA_DIR / "10.1101_2022.03.14.484320.json"
INTRO_KEYWORD = "INTRODUCTION"
INTRO_MAX_CHARS = 3000  # ~500 words — comparable length to the abstract

# --- Load and prepare texts ---
paper = load_paper(str(TARGET_FILE))

abstract_text = get_abstract(paper)
cleaned_full = clean_text(paper["text"])
intro_text = extract_section(cleaned_full, INTRO_KEYWORD, max_chars=INTRO_MAX_CHARS)

print(f"\nPaper:    {paper.get('doc_id', TARGET_FILE.stem)}")
print(f"Title:    {paper.get('metadata', {}).get('pub_metadata', {}).get('preprint_title', 'N/A')}")
print(f"Category: {paper.get('metadata', {}).get('pub_metadata', {}).get('preprint_category', 'N/A')}")

# Show which intro section was found (helps debugging)
intro_preview = intro_text[:80].replace("\n", " ")
print(f"\nIntro section starts with: \"{intro_preview}...\"")

# --- Analyze ---
abstract_metrics = analyze_complexity(abstract_text)
intro_metrics = analyze_complexity(intro_text)

# --- Print comparison table ---
# Columns: key, display label, scale/direction, novel/everyday reference
METRIC_ROWS = [
    ("word_count",
     "Word count",
     "words, higher = longer text",
     "novel ~80k-120k total"),
    ("sentence_count",
     "Sentence count",
     "sentences, higher = more sentences",
     "novel ~3k-6k total"),
    ("avg_sentence_length_words",
     "Avg sentence length",
     "words/sent, higher = longer sentences",
     "novel ~15-20, newspaper ~20-25"),
    ("avg_word_length_chars",
     "Avg word length",
     "chars/word, higher = longer words",
     "novel ~4.5-5.0, news ~4.5-5.0"),
    ("flesch_reading_ease",
     "Flesch Reading Ease",
     "0-100, higher = easier to read",
     "novel ~70-80, newspaper ~60-70"),
    ("flesch_kincaid_grade",
     "Flesch-Kincaid Grade",
     "US grade level, higher = harder",
     "novel ~7-9, newspaper ~10-12"),
    ("gunning_fog",
     "Gunning Fog",
     "US grade level, higher = more jargon",
     "novel ~8-10, newspaper ~12"),
    ("smog_index",
     "SMOG Index",
     "US grade level, higher = harder",
     "novel ~8, newspaper ~11-12"),
    ("coleman_liau_index",
     "Coleman-Liau",
     "US grade level (char-based), higher = harder",
     "novel ~7-9, newspaper ~10-11"),
    ("ari",
     "ARI",
     "US grade level (char-based), higher = harder",
     "novel ~7-9, newspaper ~10-12"),
    ("dale_chall",
     "Dale-Chall",
     "score 0-16, higher = more unfamiliar words",
     "novel ~6-8, newspaper ~7-8"),
    ("type_token_ratio",
     "Type-Token Ratio",
     "0-1, higher = richer vocabulary",
     "novel ~0.55-0.70 at ~500 words"),
    ("complex_word_ratio",
     "Complex word ratio",
     "0-1, share of >=3-syllable words",
     "novel ~0.10-0.18, newspaper ~0.15-0.20"),
    ("lexical_density",
     "Lexical density",
     "0-1, higher = more content words (denser info)",
     "novel ~0.45-0.55, speech ~0.35-0.40"),
]

LW = 30   # label column width
VW = 13   # value column width
IW = 48   # info column width

print()
print(f"{'Metric':<{LW}} {'Abstract':>{VW}} {'Introduction':>{VW}}  {'Scale & direction':<{IW}}  Novel / everyday reference")
print("-" * (LW + VW * 2 + IW + 28))

def fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)

for key, label, scale, reference in METRIC_ROWS:
    a = abstract_metrics.get(key)
    i = intro_metrics.get(key)
    print(f"{label:<{LW}} {fmt(a):>{VW}} {fmt(i):>{VW}}  {scale:<{IW}}  {reference}")

print()

# --- Warnings ---
for name, m in [("Abstract", abstract_metrics), ("Introduction", intro_metrics)]:
    if m.get("sentence_count", 0) < 30:
        print(f"  Note: {name} has {m['sentence_count']} sentences -- SMOG index is unreliable (needs >=30).")

for name, m in [("Abstract", abstract_metrics), ("Introduction", intro_metrics)]:
    if m.get("_warning_textstat"):
        print(f"\n  Missing dependency: {m['_warning_textstat']}")
        break

print()
