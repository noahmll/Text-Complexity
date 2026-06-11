"""
complexity_over_versions.py

Analyses how simple text metrics and linguistic complexity change across
successive versions of bioRxiv preprints (JATS XML, data/xml/).

Papers with version gaps or not starting at v1 are excluded.

Outputs:
    simple_metrics_over_versions.png
    complexity_over_versions.png

Usage:
    python complexity_over_versions.py              # all valid papers
    python complexity_over_versions.py --n 200      # random 200 papers
    python complexity_over_versions.py --n 200 --seed 42
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

ROOT    = Path(__file__).parent
XML_DIR = ROOT / "data" / "xml"
sys.path.insert(0, str(ROOT / "src"))

# ── NLTK setup ────────────────────────────────────────────────────────────────
def _setup_nltk() -> bool:
    try:
        import nltk
        for name, cat in [("punkt_tab", "tokenizers"), ("brown", "corpora")]:
            try:
                nltk.data.find(f"{cat}/{name}")
            except LookupError:
                nltk.download(name, quiet=True)
        return True
    except Exception:
        return False

NLTK_OK = _setup_nltk()

def _build_common_words() -> set:
    if not NLTK_OK:
        return set()
    try:
        from nltk.corpus import brown
        freq = Counter(w.lower() for w in brown.words() if w.isalpha())
        return {w for w, _ in freq.most_common(10_000)}
    except Exception:
        return set()

COMMON_WORDS: set = _build_common_words()

NOMIN_SUFFIXES = (
    "tion", "sion", "ment", "ness", "ity", "ance", "ence", "ure", "ism",
)


# ═══════════════════════════════════════════════════════════════════════════════
# XML PARSING
# ═══════════════════════════════════════════════════════════════════════════════

_SKIP_BODY = frozenset({
    "fig", "table-wrap", "supplementary-material",
    "disp-formula", "inline-formula",
})


def _collect_text(el: ET.Element, skip: frozenset) -> list:
    """Recursively collect text strings, skipping subtrees in `skip`."""
    if el.tag in skip:
        return []
    parts = []
    if el.text and el.text.strip():
        parts.append(el.text.strip())
    for child in el:
        parts.extend(_collect_text(child, skip))
        if child.tail and child.tail.strip():
            parts.append(child.tail.strip())
    return parts


def parse_xml(path: Path) -> Optional[dict]:
    """Parse a JATS XML file; return a data dict or None on failure."""
    m = re.search(r"_v(\d+)$", path.stem)
    if not m:
        return None
    version = int(m.group(1))

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError:
        return None

    doi_el = root.find('.//article-id[@pub-id-type="doi"]')
    if doi_el is None or not doi_el.text:
        return None
    doi = doi_el.text.strip()

    abstract_el = root.find(".//abstract")
    body_el     = root.find(".//body")

    abstract_text = " ".join(_collect_text(abstract_el, frozenset())) if abstract_el is not None else ""
    body_text     = " ".join(_collect_text(body_el, _SKIP_BODY))       if body_el     is not None else ""
    full_text     = re.sub(r"\s+", " ", (abstract_text + " " + body_text).strip())

    n_sections = sum(1 for c in body_el if c.tag == "sec") if body_el is not None else 0
    n_figures  = len(root.findall(".//fig"))
    n_tables   = len(root.findall(".//table-wrap"))
    n_refs     = len(root.findall(".//ref"))

    return {
        "doi":        doi,
        "version":    version,
        "full_text":  full_text,
        "n_sections": n_sections,
        "n_figures":  n_figures,
        "n_tables":   n_tables,
        "n_refs":     n_refs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOADING & FILTERING
# ═══════════════════════════════════════════════════════════════════════════════

def _scan_file_groups(xml_dir: Path) -> dict:
    """
    Group XML paths by DOI-key (filename stem without _vN) using only filenames —
    no parsing. Returns only groups with consecutive versions starting at v1.
    """
    raw_groups: dict = defaultdict(list)
    for p in sorted(xml_dir.glob("*.xml")):
        m = re.search(r"_v(\d+)$", p.stem)
        if not m:
            continue
        doi_key = p.stem[: m.start()]
        raw_groups[doi_key].append((int(m.group(1)), p))

    valid: dict = {}
    for key, entries in raw_groups.items():
        entries.sort()
        nums = [v for v, _ in entries]
        if nums[0] != 1:
            continue
        if nums != list(range(1, len(nums) + 1)):
            continue
        if len(nums) < 2:
            continue
        valid[key] = [p for _, p in entries]
    return valid


def load_papers(xml_dir: Path, n: Optional[int], seed: Optional[int]) -> dict:
    """
    Scan filenames to identify valid papers first, subsample if needed,
    then parse only the selected files.
    """
    print("Scanning XML filenames ...", flush=True)
    file_groups = _scan_file_groups(xml_dir)
    print(f"Valid papers (consecutive from v1, >=2 versions): {len(file_groups)}", flush=True)

    if n is not None and n < len(file_groups):
        rng = random.Random(seed)
        keys = rng.sample(sorted(file_groups), n)
        file_groups = {k: file_groups[k] for k in keys}
        print(f"Subsampled to {n} papers.", flush=True)

    total_files = sum(len(v) for v in file_groups.values())
    print(f"Parsing {total_files} XML files ...", flush=True)

    papers: dict = {}
    for i, (key, paths) in enumerate(file_groups.items()):
        if i and i % 100 == 0:
            print(f"  {i}/{len(file_groups)} papers", flush=True)
        versions = []
        for p in paths:
            d = parse_xml(p)
            if d and len(d["full_text"]) > 300:
                versions.append(d)
        if len(versions) >= 2:
            papers[versions[0]["doi"]] = versions

    return papers


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def _word_tokens(text: str) -> list:
    return re.findall(r"[a-zA-Z']+", text)


def _sentence_count(text: str) -> int:
    if NLTK_OK:
        try:
            import nltk
            return max(1, len(nltk.sent_tokenize(text)))
        except Exception:
            pass
    return max(1, len(re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())))


def _syllables(word: str) -> int:
    w = word.lower().strip("'-.")
    if not w:
        return 1
    count, prev_v = 0, False
    for ch in w:
        v = ch in "aeiouy"
        if v and not prev_v:
            count += 1
        prev_v = v
    if w.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def compute_metrics(text: str) -> Optional[dict]:
    """Return all metrics for a text, or None if too short."""
    words = _word_tokens(text)
    n_w   = len(words)
    if n_w < 200:
        return None

    n_s    = _sentence_count(text)
    avg_sl = n_w / n_s
    avg_wl = sum(len(w) for w in words) / n_w
    flesch = 206.835 - 1.015 * avg_sl - 84.6 * (sum(_syllables(w) for w in words) / n_w)
    wl     = [w.lower() for w in words]
    ttr    = len(set(wl)) / n_w
    rare   = (sum(1 for w in wl if w not in COMMON_WORDS) / n_w) if COMMON_WORDS else float("nan")
    nom    = sum(1 for w in wl if w.endswith(NOMIN_SUFFIXES)) / n_w

    return {
        "word_count":          n_w,
        "avg_sent_len":        avg_sl,
        "avg_word_len":        avg_wl,
        "flesch":              flesch,
        "ttr":                 ttr,
        "rare_word_rate":      rare,
        "nominalization_rate": nom,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

def build_paper_metrics(papers: dict) -> dict:
    """
    Compute all metrics per paper per version.
    Returns {doi: {version: {metric: value}}}.
    """
    result = {}
    for doi, versions in papers.items():
        paper_data = {}
        for vd in versions:
            v = vd["version"]
            entry = {
                "n_sections": vd["n_sections"],
                "n_figures":  vd["n_figures"],
                "n_tables":   vd["n_tables"],
                "n_refs":     vd["n_refs"],
            }
            cx = compute_metrics(vd["full_text"])
            if cx:
                entry.update(cx)
            paper_data[v] = entry
        if len(paper_data) >= 2:
            result[doi] = paper_data
    return result


def _clean_val(v) -> bool:
    return v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))


def aggregate_absolute(paper_metrics: dict) -> tuple:
    """Per-version mean ± SD in absolute values across all papers."""
    by_v: dict = defaultdict(lambda: defaultdict(list))
    for doi, versions in paper_metrics.items():
        for v, metrics in versions.items():
            for k, val in metrics.items():
                if _clean_val(val):
                    by_v[v][k].append(val)
    n_at_v = {v: len(by_v[v].get("n_sections", [])) for v in by_v}
    return {v: dict(d) for v, d in by_v.items()}, n_at_v


def aggregate_normalized(paper_metrics: dict) -> tuple:
    """
    Per-version mean ± SD of % change relative to v1 across all papers.
    v1 is always 0 %. Papers where v1 value is 0 are skipped for that metric.
    """
    by_v: dict = defaultdict(lambda: defaultdict(list))
    for doi, versions in paper_metrics.items():
        v1 = versions.get(1)
        if not v1:
            continue
        for v, metrics in versions.items():
            for k, val in metrics.items():
                baseline = v1.get(k)
                if not _clean_val(baseline) or not _clean_val(val):
                    continue
                if abs(baseline) < 1e-9:
                    continue
                pct = (val - baseline) / abs(baseline) * 100
                by_v[v][k].append(pct if v > 1 else 0.0)
    n_at_v = {v: len(by_v[v].get("n_sections", [])) for v in by_v}
    return {v: dict(d) for v, d in by_v.items()}, n_at_v


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

_SIMPLE_SPECS = [
    ("word_count",   "Word Count",           "words"),
    ("n_sections",   "Top-level Sections",   "count"),
    ("n_figures",    "Figures",              "count"),
    ("n_tables",     "Tables",               "count"),
    ("n_refs",       "References",           "count"),
    ("avg_sent_len", "Avg. Sentence Length", "words / sentence"),
]

_COMPLEX_SPECS = [
    ("flesch",             "Flesch Reading Ease",  "score (higher = easier)"),
    ("avg_word_len",       "Avg. Word Length",     "characters"),
    ("ttr",                "Type-Token Ratio",     "ratio (higher = richer)"),
    ("rare_word_rate",     "Rare-Word Rate",        "fraction"),
    ("nominalization_rate","Nominalization Rate",  "fraction"),
    ("avg_sent_len",       "Avg. Sentence Length", "words / sentence"),
]

_COLORS = ["#2166AC", "#1A9850", "#D6604D", "#762A83", "#E08214", "#4D4D4D"]


def _xtick_labels(xs: list, n_at_v: dict) -> list:
    return [f"v{int(x)}\n(n={n_at_v.get(int(x), '?')})" for x in xs]


def _plot_spaghetti_grid(
    paper_metrics: dict,
    by_v: dict,
    n_at_v: dict,
    specs: list,
    title: str,
    outpath: Path,
    min_n: int = 3,
    max_lines: int = 400,
) -> None:
    """Spaghetti plot: individual paper trajectories (thin, transparent) + bold mean."""
    versions = sorted(v for v in by_v if n_at_v.get(v, 0) >= min_n)

    # Subsample papers for display if corpus is large
    doi_list = list(paper_metrics.keys())
    if len(doi_list) > max_lines:
        doi_list = random.Random(0).sample(doi_list, max_lines)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for idx, ((key, label, unit), color) in enumerate(zip(specs, _COLORS)):
        ax = axes[idx]

        # Individual paper lines
        for doi in doi_list:
            pv = paper_metrics[doi]
            xs_p = sorted(v for v in pv if _clean_val(pv[v].get(key)))
            if len(xs_p) < 2:
                continue
            ys_p = [pv[v][key] for v in xs_p]
            ax.plot(xs_p, ys_p, lw=0.5, alpha=0.08, color=color)

        # Bold mean line
        xs, means = [], []
        for v in versions:
            vals = [x for x in by_v[v].get(key, []) if _clean_val(x)]
            if vals:
                xs.append(v)
                means.append(np.mean(vals))

        if not xs:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(label, fontsize=10, fontweight="bold")
            continue

        ax.plot(xs, means, "o-", lw=2.5, ms=5, color=color, zorder=5, label="mean")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylabel(unit, fontsize=8)
        ax.set_xlabel("Version", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(_xtick_labels(xs, n_at_v), fontsize=7)
        ax.grid(True, alpha=0.3, linestyle="--")
        if idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved -> {outpath}")


def _plot_normalized_grid(
    by_v_norm: dict,
    n_at_v: dict,
    specs: list,
    title: str,
    outpath: Path,
    min_n: int = 3,
) -> None:
    """% change from v1: mean line + shaded ±1 SD band."""
    versions = sorted(v for v in by_v_norm if n_at_v.get(v, 0) >= min_n)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for idx, ((key, label, _unit), color) in enumerate(zip(specs, _COLORS)):
        ax = axes[idx]
        xs, means, stds = [], [], []
        for v in versions:
            vals = [x for x in by_v_norm[v].get(key, []) if _clean_val(x)]
            if vals:
                xs.append(v)
                means.append(np.mean(vals))
                stds.append(np.std(vals))

        if not xs:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(label, fontsize=10, fontweight="bold")
            continue

        xs, means, stds = np.array(xs), np.array(means), np.array(stds)

        ax.axhline(0, color="gray", lw=1, linestyle="--", alpha=0.6, zorder=1)
        ax.plot(xs, means, "o-", lw=2, ms=5, color=color, zorder=5, label="mean % change")
        ax.fill_between(xs, means - stds, means + stds, alpha=0.18, color=color, label="+/-1 SD")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylabel("% change from v1", fontsize=8)
        ax.set_xlabel("Version", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(_xtick_labels(xs, n_at_v), fontsize=7)
        ax.grid(True, alpha=0.3, linestyle="--")
        if idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved -> {outpath}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n",    default="all", help='papers to use, or "all" (default: all)')
    ap.add_argument("--seed", type=int, default=None, help="random seed for subsampling")
    args = ap.parse_args()

    n = None if args.n.lower() == "all" else int(args.n)
    papers = load_papers(XML_DIR, n, args.seed)

    if not papers:
        sys.exit("No valid papers found.")

    if not COMMON_WORDS:
        print("Warning: NLTK Brown corpus unavailable -- rare-word rate will be NaN.")

    print("Computing metrics ...", flush=True)
    paper_metrics           = build_paper_metrics(papers)
    by_v,      n_at_v      = aggregate_absolute(paper_metrics)
    by_v_norm, n_at_v_norm = aggregate_normalized(paper_metrics)
    print(f"Papers per version: { {v: n_at_v[v] for v in sorted(n_at_v)} }")

    n_total = len(paper_metrics)

    # ── Spaghetti plots (absolute values + bold mean) ─────────────────────────
    _plot_spaghetti_grid(
        paper_metrics, by_v, n_at_v, _SIMPLE_SPECS,
        title   = f"Simple Text Metrics — Individual Trajectories  (n={n_total} papers)",
        outpath = ROOT / "simple_metrics_over_versions.png",
    )
    _plot_spaghetti_grid(
        paper_metrics, by_v, n_at_v, _COMPLEX_SPECS,
        title   = f"Linguistic Complexity — Individual Trajectories  (n={n_total} papers)",
        outpath = ROOT / "complexity_over_versions.png",
    )

    # ── Normalised plots (% change from v1) ───────────────────────────────────
    _plot_normalized_grid(
        by_v_norm, n_at_v_norm, _SIMPLE_SPECS,
        title   = f"Simple Text Metrics — Change from v1 (%)  (n={n_total} papers)",
        outpath = ROOT / "simple_metrics_normalized.png",
    )
    _plot_normalized_grid(
        by_v_norm, n_at_v_norm, _COMPLEX_SPECS,
        title   = f"Linguistic Complexity — Change from v1 (%)  (n={n_total} papers)",
        outpath = ROOT / "complexity_normalized.png",
    )

    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
