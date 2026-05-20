"""
complexity_over_text.py

Measures 6 text-complexity metrics as a sliding-window average over the full
body of N scientific preprints. Results are shown as 6 line plots where the
x-axis is relative text position (0 % = start, 100 % = end) and the y-axis
is the cross-document average for each metric.

Usage
-----
    python complexity_over_text.py              # default: 20 documents
    python complexity_over_text.py --n 50       # 50 random documents
    python complexity_over_text.py --n all      # all available documents
    python complexity_over_text.py --n 30 --seed 7   # reproducible sample
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── project paths ─────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
SRC_DIR    = ROOT / "src"
OUTPUT_DIR = ROOT / "SampleData" / "output"
sys.path.insert(0, str(SRC_DIR))

from text_extraction import clean_text  # existing cleaner from src/

# ── analysis parameters ───────────────────────────────────────────────────────
WINDOW_WORDS = 200   # words per sliding window
STEP_WORDS   = 50    # step between windows (overlap = WINDOW_WORDS - STEP_WORDS)
N_BINS       = 50    # position buckets (0 – 100 %)
MIN_WORDS    = 800   # skip documents shorter than this after cleaning

# ── NLTK setup ────────────────────────────────────────────────────────────────
def _setup_nltk() -> bool:
    try:
        import nltk
        needed = [
            ("punkt_tab",                    "tokenizers"),
            ("averaged_perceptron_tagger_eng","taggers"),
            ("brown",                        "corpora"),
        ]
        for name, cat in needed:
            try:
                nltk.data.find(f"{cat}/{name}")
            except LookupError:
                nltk.download(name, quiet=True)
        return True
    except Exception:
        return False

NLTK_OK = _setup_nltk()

# ── common-word list (top 10 000 English words from Brown corpus) ─────────────
def _build_common_words() -> set:
    """Build a set of the ~10 000 most frequent lowercase English words."""
    if not NLTK_OK:
        return set()
    try:
        from collections import Counter
        from nltk.corpus import brown
        freq = Counter(w.lower() for w in brown.words() if w.isalpha())
        return {w for w, _ in freq.most_common(10_000)}
    except Exception:
        return set()

COMMON_WORDS: set = _build_common_words()

# ── nominalization suffixes ───────────────────────────────────────────────────
NOMIN_SUFFIXES = (
    "tion", "sion", "ment", "ness", "ity", "ance", "ence",
    "ure", "al", "age", "ism",
)

# ── text cleaning ─────────────────────────────────────────────────────────────
def _deep_clean(raw: str) -> str:
    """
    Full cleaning pipeline:
    1. Existing clean_text() strips pipe-table rows and lone line numbers.
    2. Reference / acknowledgement sections are removed.
    3. Figure and table captions are removed.
    4. Inline citation markers are removed.
    """
    text = clean_text(raw)

    # Remove everything from a section header that signals non-prose content
    text = re.sub(
        r"\b(REFERENCES|BIBLIOGRAPHY|ACKNOWLEDGEMENTS?|FUNDING|"
        r"SUPPORTING\s+INFORMATION|SUPPLEMENTARY)\b.*",
        "", text, flags=re.IGNORECASE | re.DOTALL,
    )

    # Figure / table captions (e.g. "Figure 3. …sentence…")
    text = re.sub(r"\b(Figure|Fig\.|Table)\s+\S[^.]*\.", "", text, flags=re.IGNORECASE)

    # Numeric citation markers  (1)  [2,3]  [4–6]
    text = re.sub(r"\(\d[\d,\s–\-]*\)|\[\d[\d,\s–\-]*\]", "", text)

    # Author-year citations  (Smith et al., 2020)  (Jones, 2019)
    text = re.sub(r"\([A-Z][a-z]+(?:\s+et\s+al\.?)?,\s*\d{4}\)", "", text)

    # Collapse remaining whitespace
    return re.sub(r"\s{2,}", " ", text).strip()


# ── tokenization helpers ──────────────────────────────────────────────────────
def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text)


def _sentence_count(text: str) -> int:
    """Sentence count using NLTK if available, regex heuristic as fallback."""
    if NLTK_OK:
        try:
            import nltk
            return max(1, len(nltk.sent_tokenize(text)))
        except Exception:
            pass
    return max(1, len(re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())))


def _syllables(word: str) -> int:
    """Vowel-group heuristic with silent-terminal-e correction."""
    w = word.lower().strip("'-.")
    if not w:
        return 0
    count, prev_v = 0, False
    for ch in w:
        v = ch in "aeiouy"
        if v and not prev_v:
            count += 1
        prev_v = v
    if w.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


# ── six complexity metrics ────────────────────────────────────────────────────

def metric_flesch(text: str) -> float:
    """Flesch Reading Ease  (higher = easier, typical science ≈ 0–30)."""
    words = _word_tokens(text)
    if not words:
        return float("nan")
    n_w = len(words)
    n_s = _sentence_count(text)
    n_y = sum(_syllables(w) for w in words)
    return 206.835 - 1.015 * (n_w / n_s) - 84.6 * (n_y / n_w)


def metric_avg_sent_len(text: str) -> float:
    """Average sentence length in words (higher = more complex)."""
    words = _word_tokens(text)
    if not words:
        return float("nan")
    return len(words) / _sentence_count(text)


def metric_avg_word_len(text: str) -> float:
    """Average word length in characters (higher = longer / harder words)."""
    words = _word_tokens(text)
    if not words:
        return float("nan")
    return sum(len(w) for w in words) / len(words)


def metric_ttr(text: str) -> float:
    """
    Type-Token Ratio on the window.
    Because the window size is fixed (200 words), TTR is comparable
    across all positions without additional length correction.
    """
    words = [w.lower() for w in _word_tokens(text)]
    if not words:
        return float("nan")
    return len(set(words)) / len(words)


def metric_rare_word_rate(text: str) -> float:
    """
    Fraction of words outside the 10 000 most common English words.
    High value → many rare / technical terms.
    Falls back to 0 if the common-word list is unavailable.
    """
    if not COMMON_WORDS:
        return float("nan")
    words = [w.lower() for w in _word_tokens(text)]
    if not words:
        return float("nan")
    return sum(1 for w in words if w not in COMMON_WORDS) / len(words)


def metric_nominalization_rate(text: str) -> float:
    """
    Fraction of ALL words that are nominalizations.
    Uses NLTK POS tags to identify nouns first, then checks suffixes.
    Falls back to suffix-only detection on all words if NLTK is unavailable.
    """
    words = _word_tokens(text)
    if not words:
        return float("nan")

    if NLTK_OK:
        try:
            import nltk
            tagged_here = nltk.pos_tag(words)
            nouns = [w.lower() for w, tag in tagged_here if tag.startswith("NN")]
        except Exception:
            nouns = [w.lower() for w in words]
    else:
        nouns = [w.lower() for w in words]

    nominalizations = sum(1 for w in nouns if w.endswith(NOMIN_SUFFIXES))
    return nominalizations / len(words)


# ── sliding-window generator ──────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving punctuation."""
    if NLTK_OK:
        try:
            import nltk
            return nltk.sent_tokenize(text)
        except Exception:
            pass
    # Fallback: split on .!? followed by whitespace + uppercase
    return re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())


def iter_windows(text: str):
    """
    Yield (relative_position, window_text) using a sentence-aware sliding
    window of ~WINDOW_WORDS words.  Each window is built by accumulating
    whole sentences until the word count crosses WINDOW_WORDS, then stepped
    forward by STEP_WORDS words.

    Using whole sentences preserves punctuation so that sentence-counting
    metrics (Flesch, avg. sentence length) remain valid.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return

    # Build a word-offset index: cumulative word count at the start of each sentence
    sent_word_counts = [len(_word_tokens(s)) for s in sentences]
    total_words = sum(sent_word_counts)

    if total_words < MIN_WORDS:
        return

    cumulative = [0]
    for c in sent_word_counts:
        cumulative.append(cumulative[-1] + c)

    step = 0  # word offset of the window start
    while step + WINDOW_WORDS <= total_words:
        # Find the first sentence that starts at or after `step`
        s_start = next(
            (i for i, cum in enumerate(cumulative) if cum >= step),
            len(sentences) - 1,
        )
        # Accumulate sentences until we have at least WINDOW_WORDS words
        s_end = s_start
        words_so_far = 0
        while s_end < len(sentences) and words_so_far < WINDOW_WORDS:
            words_so_far += sent_word_counts[s_end]
            s_end += 1

        win_text = " ".join(sentences[s_start:s_end])
        win_word_count = words_so_far
        centre = (step + win_word_count / 2) / total_words
        centre = max(0.0, min(1.0, centre))

        yield centre, win_text

        step += STEP_WORDS


# ── metric definitions registry ───────────────────────────────────────────────
# Each entry: (display_label, callable(text) -> float)
METRICS: list[tuple[str, callable]] = [
    ("Flesch Reading Ease\n(higher = easier)",
     metric_flesch),
    ("Avg. Sentence Length\n(words/sentence, higher = more complex)",
     metric_avg_sent_len),
    ("Avg. Word Length\n(chars/word, higher = harder words)",
     metric_avg_word_len),
    ("Type-Token Ratio\n(higher = richer vocabulary)",
     metric_ttr),
    ("Rare Word Rate\n(fraction outside top-10k, higher = more jargon)",
     metric_rare_word_rate),
    ("Nominalization Rate\n(fraction of all words, higher = more formal)",
     metric_nominalization_rate),
]


# ── main ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot 6 complexity metrics over text position for N preprints."
    )
    p.add_argument(
        "--n", default="20",
        help="Number of documents to analyse (integer or 'all'). Default: 20",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for document sampling. Default: 42",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── select documents ──────────────────────────────────────────────────────
    all_files = sorted(OUTPUT_DIR.glob("*.json"))
    if not all_files:
        sys.exit(f"No JSON files found in {OUTPUT_DIR}")

    if args.n.lower() == "all":
        selected = all_files
    else:
        n = int(args.n)
        if n > len(all_files):
            print(f"Note: only {len(all_files)} documents available; using all.")
            selected = all_files
        else:
            random.seed(args.seed)
            selected = random.sample(all_files, n)

    print(f"Corpus      : {OUTPUT_DIR}")
    print(f"Documents   : {len(selected)} selected  ({len(all_files)} available)")
    print(f"Window      : {WINDOW_WORDS} words, step {STEP_WORDS} words")
    print(f"Position bins: {N_BINS}")
    if not NLTK_OK:
        print("Warning: NLTK not available — nominalization rate uses suffix-only fallback.")
    if not COMMON_WORDS:
        print("Warning: Common-word list unavailable — rare-word rate will be NaN.")
    print()

    # ── accumulate binned metric values ───────────────────────────────────────
    # bin_data[metric_idx][bin_idx] = list of float values
    bin_data: list[list[list[float]]] = [
        [[] for _ in range(N_BINS)] for _ in METRICS
    ]

    ok, skipped = 0, 0

    for i, path in enumerate(selected, 1):
        print(f"  [{i:>3}/{len(selected)}] {path.name}", end=" … ", flush=True)
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)

            raw = doc.get("text", "")
            if not raw:
                raise ValueError("empty text field")

            text  = _deep_clean(raw)
            words = _word_tokens(text)

            if len(words) < MIN_WORDS:
                raise ValueError(f"too short after cleaning ({len(words)} words < {MIN_WORDS})")

            n_windows = 0
            for centre, win_text in iter_windows(text):
                bin_idx = min(int(centre * N_BINS), N_BINS - 1)
                for m_idx, (_label, fn) in enumerate(METRICS):
                    val = fn(win_text)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        bin_data[m_idx][bin_idx].append(val)
                n_windows += 1

            print(f"{n_windows} windows  ({len(words)} words)")
            ok += 1

        except Exception as e:
            print(f"SKIPPED ({e})")
            skipped += 1

    print(f"\nResult : {ok} analysed, {skipped} skipped.")

    if ok == 0:
        sys.exit("No documents could be analysed.")

    # ── build plot arrays ─────────────────────────────────────────────────────
    x = np.linspace(0, 100, N_BINS)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(
        f"Text Complexity over Document Progress  "
        f"(n = {ok} papers | window = {WINDOW_WORDS} words | step = {STEP_WORDS} words)",
        fontsize=13, fontweight="bold",
    )

    COLORS = ["#2166AC", "#D6604D", "#1A9850", "#762A83", "#E08214", "#4D4D4D"]

    for ax, (label, _fn), color, bins in zip(
        axes.flat, METRICS, COLORS, bin_data
    ):
        means = np.array([np.mean(b) if b else np.nan for b in bins])
        stds  = np.array([np.std(b)  if b else np.nan for b in bins])

        # Smooth with a rolling mean (window = 5 bins ≈ 10 % of text)
        kernel = np.ones(5) / 5
        means_sm = np.convolve(means, kernel, mode="same")
        stds_sm  = np.convolve(stds,  kernel, mode="same")

        ax.plot(x, means_sm, linewidth=2.2, color=color, label="mean (smoothed)")
        ax.fill_between(
            x, means_sm - stds_sm, means_sm + stds_sm,
            alpha=0.18, color=color, label="±1 SD",
        )
        # Also show raw (unsmoothed) mean as a light dotted line
        ax.plot(x, means, linewidth=0.8, color=color, linestyle=":", alpha=0.5)

        ax.set_title(label, fontsize=10, pad=6)
        ax.set_xlabel("Text position (%)", fontsize=9)
        ax.grid(True, alpha=0.35, linestyle="--")
        ax.legend(fontsize=8, loc="upper right")

        # Shade the three structural regions
        for region, (lo, hi), alpha in [
            ("Introduction\nregion", (0,  25), 0.04),
            ("Methods /\nResults",   (25, 75), 0.04),
            ("Discussion /\nConclusion", (75, 100), 0.04),
        ]:
            ax.axvspan(lo, hi, alpha=alpha, color="grey")

        ax.set_xlim(0, 100)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = ROOT / "complexity_over_text.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved -> {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
