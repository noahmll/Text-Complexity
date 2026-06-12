"""
complexity_over_versions.py

Analyses how simple text metrics and linguistic complexity change across
successive versions of bioRxiv preprints (JATS XML, data/xml/). Abstract and
body are analysed SEPARATELY; inline bibliographic citations are stripped.

Papers with version gaps or not starting at v1 are excluded.

Outputs (written to plots/):
    papers_per_version.png            bar chart of papers per version
    body/simple_metrics/              full text — six views (see below)
    body/complexity/                  full text — six views
    abstract/complexity/              abstract  — six views (no structural metrics)

    Each <group>/ folder contains the same six views:
    normalized_over_versions.png      % change vs v1 (trajectories + mean ±1 SD)
    stepwise_over_versions.png        % change vs previous version (mean ±1 SD)
    direction_over_versions.png       share up / ~equal / down per step
    distribution.png                  histogram of per-paper net change (last vs v1)
    scatter.png                       net change vs volatility (one point/paper)
    baseline_change.png               v1 baseline vs net change (regression to mean)

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
from tqdm import tqdm

ROOT      = Path(__file__).parent
XML_DIR   = ROOT / "data" / "xml"
PLOTS_DIR = ROOT / "plots"

# Only analyse papers with exactly these versions (v1 .. vN_VERSIONS).
N_VERSIONS = 6

# ── NLTK setup ────────────────────────────────────────────────────────────────
def _ensure_resource(name: str, category: str) -> bool:
    """Find an NLTK resource, downloading it if missing, and verify it loaded.

    Returns True only if the resource is actually available afterwards, so a
    silently failed download (e.g. offline) is reported instead of swallowed.
    """
    import nltk
    path = f"{category}/{name}"
    try:
        nltk.data.find(path)
        return True
    except LookupError:
        nltk.download(name, quiet=True)
    try:
        nltk.data.find(path)
        return True
    except LookupError:
        return False


def _setup_nltk() -> bool:
    try:
        import nltk  # noqa: F401
    except ImportError:
        return False
    # punkt_tab improves sentence tokenisation; brown backs the rare-word list.
    ok = True
    for name, cat in [("punkt_tab", "tokenizers"), ("brown", "corpora")]:
        ok = _ensure_resource(name, cat) and ok
    return ok

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
    """
    Recursively collect text strings, skipping subtrees in `skip`. Inline
    bibliographic citations (<xref ref-type="bibr">…</xref>, e.g. "12" or
    "Smith et al.") are dropped so they don't distort the metrics; the text
    *after* the citation (its tail) is kept by the parent's loop.
    """
    if el.tag in skip:
        return []
    if el.tag == "xref" and el.get("ref-type") == "bibr":
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

    # Abstract and body are kept SEPARATE so each can be analysed on its own.
    abstract_text = re.sub(r"\s+", " ", " ".join(_collect_text(abstract_el, frozenset())).strip()) if abstract_el is not None else ""
    body_text     = re.sub(r"\s+", " ", " ".join(_collect_text(body_el, _SKIP_BODY)).strip())       if body_el     is not None else ""

    n_sections = sum(1 for c in body_el if c.tag == "sec") if body_el is not None else 0
    n_figures  = len(root.findall(".//fig"))
    n_tables   = len(root.findall(".//table-wrap"))
    n_refs     = len(root.findall(".//ref"))

    return {
        "doi":           doi,
        "version":       version,
        "abstract_text": abstract_text,
        "body_text":     body_text,
        "n_sections":    n_sections,
        "n_figures":     n_figures,
        "n_tables":      n_tables,
        "n_refs":        n_refs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOADING & FILTERING
# ═══════════════════════════════════════════════════════════════════════════════

def _scan_version_distribution(xml_dir: Path) -> tuple[dict, int]:
    """
    Papers-per-version counts across the WHOLE corpus, independent of the
    exact-v1..vN analysis filter — used only for the overview bar chart so it
    also reflects papers with fewer or more versions. Counts every DOI group
    whose versions run consecutively from v1 (>=2 versions); filename-only.
    """
    raw: dict = defaultdict(list)
    for p in sorted(xml_dir.glob("*.xml")):
        m = re.search(r"_v(\d+)$", p.stem)
        if not m:
            continue
        raw[p.stem[: m.start()]].append(int(m.group(1)))

    counts: dict = defaultdict(int)
    n_papers = 0
    for nums in raw.values():
        nums = sorted(nums)
        if len(nums) < 2 or nums != list(range(1, len(nums) + 1)):
            continue
        n_papers += 1
        for v in nums:
            counts[v] += 1
    return dict(counts), n_papers


def _scan_file_groups(xml_dir: Path, exact: bool = True) -> dict:
    """
    Group XML paths by DOI-key (filename stem without _vN) using only filenames —
    no parsing.

    exact=True  → only groups that have exactly versions v1..vN_VERSIONS.
    exact=False → any group whose versions run consecutively from v1 (>=2),
                  i.e. papers with fewer or more versions are also included.
    """
    raw_groups: dict = defaultdict(list)
    for p in sorted(xml_dir.glob("*.xml")):
        m = re.search(r"_v(\d+)$", p.stem)
        if not m:
            continue
        doi_key = p.stem[: m.start()]
        raw_groups[doi_key].append((int(m.group(1)), p))

    expected = list(range(1, N_VERSIONS + 1))
    valid: dict = {}
    for key, entries in raw_groups.items():
        entries.sort()
        nums = [v for v, _ in entries]
        if exact:
            if nums != expected:
                continue
        else:
            if len(nums) < 2 or nums != list(range(1, len(nums) + 1)):
                continue
        valid[key] = [p for _, p in entries]
    return valid


def load_papers(xml_dir: Path, n: Optional[int], seed: Optional[int],
                exact: bool = True) -> dict:
    """
    Scan filenames to identify valid papers first, subsample if needed,
    then parse only the selected files. With exact=False, papers with any
    number of consecutive-from-v1 versions (>=2) are loaded.
    """
    print("Scanning XML filenames ...", flush=True)
    file_groups = _scan_file_groups(xml_dir, exact=exact)
    label = (f"exactly v1..v{N_VERSIONS}" if exact
             else "consecutive from v1, >=2 versions")
    print(f"Valid papers ({label}): {len(file_groups)}", flush=True)

    if n is not None and n < len(file_groups):
        rng = random.Random(seed)
        keys = rng.sample(sorted(file_groups), n)
        file_groups = {k: file_groups[k] for k in keys}
        print(f"Subsampled to {n} papers.", flush=True)

    total_files = sum(len(v) for v in file_groups.values())
    print(f"Parsing {total_files} XML files ...", flush=True)

    papers: dict = {}
    for key, paths in tqdm(file_groups.items(), total=len(file_groups),
                           unit="papers", leave=False):
        versions = []
        for p in paths:
            d = parse_xml(p)
            if d and len(d["body_text"]) > 300:
                versions.append(d)
        # A version may drop out if its text is too short; keep papers that
        # still have enough versions for the requested mode.
        if (exact and len(versions) == N_VERSIONS) or (not exact and len(versions) >= 2):
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


def compute_metrics(text: str, min_words: int = 200) -> Optional[dict]:
    """Return all metrics for a text, or None if shorter than min_words.
    Abstracts are much shorter than bodies, so they use a lower threshold."""
    words = _word_tokens(text)
    n_w   = len(words)
    if n_w < min_words:
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

def build_paper_metrics(papers: dict, text_key: str = "body_text",
                        with_structure: bool = True, min_words: int = 200) -> dict:
    """
    Compute all metrics per paper per version from the chosen text source
    (text_key = "body_text" or "abstract_text").
    Structural counts (sections/figures/tables/refs) are document-level and only
    added when with_structure=True (they are meaningless for the abstract).
    Returns {doi: {version: {metric: value}}}.
    """
    result = {}
    for doi, versions in tqdm(papers.items(), total=len(papers),
                              unit="papers", leave=False):
        paper_data = {}
        for vd in versions:
            v = vd["version"]
            entry = {}
            if with_structure:
                entry.update({
                    "n_sections": vd["n_sections"],
                    "n_figures":  vd["n_figures"],
                    "n_tables":   vd["n_tables"],
                    "n_refs":     vd["n_refs"],
                })
            cx = compute_metrics(vd[text_key], min_words=min_words)
            if cx:
                entry.update(cx)
            paper_data[v] = entry
        if len(paper_data) >= 2:
            result[doi] = paper_data
    return result


def _clean_val(v) -> bool:
    return v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))


def build_normalized_paper_metrics(paper_metrics: dict) -> dict:
    """
    Per-paper % change of each metric relative to that paper's own v1 (v1 = 0 %).
    Returns {doi: {version: {metric: pct_change}}}. Metrics whose v1 baseline is
    missing or zero are omitted for that paper, mirroring aggregate_normalized().
    """
    result: dict = {}
    for doi, versions in paper_metrics.items():
        v1 = versions.get(1)
        if not v1:
            continue
        pdata: dict = {}
        for v, metrics in versions.items():
            entry = {}
            for k, val in metrics.items():
                baseline = v1.get(k)
                if not _clean_val(baseline) or not _clean_val(val):
                    continue
                if abs(baseline) < 1e-9:
                    continue
                entry[k] = 0.0 if v == 1 else (val - baseline) / abs(baseline) * 100
            pdata[v] = entry
        result[doi] = pdata
    return result


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


def aggregate_stepwise(paper_metrics: dict) -> dict:
    """
    Per-step % change relative to the IMMEDIATELY PRECEDING version.
    Returns {v: {metric: [pct change from v-1 to v]}} for v = 2..N.
    Complements aggregate_normalized() (which is always relative to v1).
    """
    by_step: dict = defaultdict(lambda: defaultdict(list))
    for doi, versions in paper_metrics.items():
        for v in sorted(versions):
            if v == 1:
                continue
            prev, cur = versions.get(v - 1), versions.get(v)
            if not prev or not cur:
                continue
            for k, val in cur.items():
                base = prev.get(k)
                if not _clean_val(base) or not _clean_val(val) or abs(base) < 1e-9:
                    continue
                by_step[v][k].append((val - base) / abs(base) * 100)
    return {v: dict(d) for v, d in by_step.items()}


def build_net_change(paper_metrics: dict) -> dict:
    """
    Per-paper NET % change of each metric: last version vs v1.
    Returns {metric: [pct change, ...]} across papers — the raw material for the
    distribution view, so we can see whether a near-zero mean hides two opposing
    sub-populations rather than genuine stability.
    """
    out: dict = defaultdict(list)
    for doi, versions in paper_metrics.items():
        v1   = versions.get(1)
        last = versions.get(max(versions)) if versions else None
        if not v1 or not last:
            continue
        for k, val in last.items():
            base = v1.get(k)
            if not _clean_val(base) or not _clean_val(val) or abs(base) < 1e-9:
                continue
            out[k].append((val - base) / abs(base) * 100)
    return dict(out)


def build_trajectory_features(paper_metrics: dict, keys: list) -> dict:
    """
    Per-paper trajectory summary for each metric, across ALL papers (>=2 versions,
    any version count):
      - base : the v1 (baseline) value in the metric's natural units
      - net  : net % change of the last version vs v1  (signed magnitude)
      - vol  : volatility = RMSE of the residuals around a linear trend
               (value vs version index), expressed as % of the v1 baseline.
               The linear fit absorbs the overall direction, so this measures
               only the scatter *around* the trend and is largely independent of
               the net change. 0 = trajectory lies exactly on a straight line;
               larger = more erratic. (Papers with only 2 versions are 0 by
               construction, since a line fits two points exactly.)
      - nver : number of versions of the paper
    Returns {metric: {"base": [...], "net": [...], "vol": [...], "nver": [...]}}.
    """
    feats = {k: {"base": [], "net": [], "vol": [], "nver": []} for k in keys}
    for doi, versions in paper_metrics.items():
        vs = sorted(versions)
        if len(vs) < 2:
            continue
        for k in keys:
            seq = [versions[v].get(k) for v in vs]
            if any(not _clean_val(x) for x in seq):
                continue
            first, last = seq[0], seq[-1]
            if abs(first) < 1e-9:
                continue
            yv = np.asarray(seq, dtype=float)
            xv = np.arange(yv.size, dtype=float)
            if yv.size >= 3:
                slope, intercept = np.polyfit(xv, yv, 1)
                resid = yv - (slope * xv + intercept)
                vol = float(np.sqrt(np.mean(resid ** 2)) / abs(first) * 100.0)
            else:
                vol = 0.0  # 2 versions: a line fits exactly → no residual
            feats[k]["base"].append(first)
            feats[k]["net"].append((last - first) / abs(first) * 100)
            feats[k]["vol"].append(vol)
            feats[k]["nver"].append(len(vs))
    return feats


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

# Direction-of-change config (step view).
_STEP_THRESHOLD = 1.0  # |% change| below this counts as "unchanged"

# For complexity metrics: does an INCREASE make the text simpler (-1) or more
# complex (+1)? Used to label the direction bars semantically.
_COMPLEXITY_POLARITY = {
    "flesch":              -1,  # higher Flesch Reading Ease = easier to read
    "avg_word_len":        +1,
    "ttr":                 +1,
    "rare_word_rate":      +1,
    "nominalization_rate": +1,
    "avg_sent_len":        +1,
}


def _xtick_labels(xs: list) -> list:
    return [f"v{int(x)}" for x in xs]


def _step_labels(steps: list) -> list:
    return [f"v{int(v) - 1}→v{int(v)}" for v in steps]


def _plot_normalized_spaghetti_grid(
    norm_paper_metrics: dict,
    by_v_norm: dict,
    n_at_v: dict,
    specs: list,
    title: str,
    outpath: Path,
    min_n: int = 3,
    max_lines: int = 400,
) -> None:
    """
    Combined view per metric: individual paper trajectories as % change from v1
    (thin, transparent) overlaid with the bold cross-paper mean and a ±1 SD band.
    """
    versions = sorted(v for v in by_v_norm if n_at_v.get(v, 0) >= min_n)

    # Subsample papers for display if corpus is large
    doi_list = list(norm_paper_metrics.keys())
    if len(doi_list) > max_lines:
        doi_list = random.Random(0).sample(doi_list, max_lines)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for idx, ((key, label, _unit), color) in enumerate(zip(specs, _COLORS)):
        ax = axes[idx]
        ax.axhline(0, color="gray", lw=1, linestyle="--", alpha=0.6, zorder=1)

        # Individual normalized paper lines (collect values for robust y-limits)
        line_vals: list = []
        for doi in doi_list:
            pv = norm_paper_metrics[doi]
            xs_p = sorted(v for v in pv if v in versions and _clean_val(pv[v].get(key)))
            if len(xs_p) < 2:
                continue
            ys_p = [pv[v][key] for v in xs_p]
            line_vals.extend(ys_p)
            ax.plot(xs_p, ys_p, lw=0.5, alpha=0.08, color=color, zorder=2)

        # Bold mean line + ±1 SD band
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
        ax.fill_between(xs, means - stds, means + stds, alpha=0.18,
                        color=color, zorder=3, label="±1 SD")
        ax.plot(xs, means, "o-", lw=2.5, ms=5, color=color, zorder=5,
                label="mean % change")

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylabel("% change from v1", fontsize=8)
        ax.set_xlabel("Version", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(_xtick_labels(xs), fontsize=7)
        ax.grid(True, alpha=0.3, linestyle="--")
        if idx == 0:
            ax.legend(fontsize=8)

        # Robust y-limits: clip extreme individual outliers (1st–99th pct) but
        # always keep the mean ± SD band fully visible.
        if line_vals:
            lo, hi = np.percentile(line_vals, [1, 99])
            lo = min(lo, float((means - stds).min()))
            hi = max(hi, float((means + stds).max()))
            pad = max(1.0, 0.05 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")


def _plot_version_distribution(n_at_v: dict, n_total: int, outpath: Path) -> None:
    """Bar chart of how many papers contribute at each version."""
    versions = sorted(n_at_v)
    counts   = [n_at_v[v] for v in versions]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([f"v{v}" for v in versions], counts,
                  color="#2166AC", edgecolor="white")
    ax.bar_label(bars, fontsize=8, padding=2)

    ax.set_title(f"Papers per Version  (n={n_total} papers)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Version", fontsize=9)
    ax.set_ylabel("Number of papers", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")


def _plot_stepwise_grid(
    by_step: dict,
    specs: list,
    title: str,
    outpath: Path,
) -> None:
    """Mean % change vs the preceding version per transition, with ±1 SD band."""
    steps = sorted(by_step)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for idx, ((key, label, _unit), color) in enumerate(zip(specs, _COLORS)):
        ax = axes[idx]
        ax.axhline(0, color="gray", lw=1, linestyle="--", alpha=0.6, zorder=1)

        xs, means, stds = [], [], []
        for v in steps:
            vals = [x for x in by_step[v].get(key, []) if _clean_val(x)]
            if vals:
                xs.append(v)
                means.append(np.mean(vals))
                stds.append(np.std(vals))

        if not xs:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(label, fontsize=10, fontweight="bold")
            continue

        pos = np.arange(len(xs))
        means, stds = np.array(means), np.array(stds)
        ax.fill_between(pos, means - stds, means + stds, alpha=0.18,
                        color=color, zorder=3, label="±1 SD")
        ax.plot(pos, means, "o-", lw=2.5, ms=5, color=color, zorder=5,
                label="mean % change")

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylabel("% change vs previous", fontsize=8)
        ax.set_xlabel("Version step", fontsize=8)
        ax.set_xticks(pos)
        ax.set_xticklabels(_step_labels(xs), fontsize=7)
        ax.grid(True, alpha=0.3, linestyle="--")
        if idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")


def _plot_direction_grid(
    by_step: dict,
    specs: list,
    title: str,
    outpath: Path,
    semantic: bool,
    threshold: float = _STEP_THRESHOLD,
) -> None:
    """
    Stacked bars per transition: share of papers that go up / stay ~equal / go
    down (|Δ| < threshold counts as unchanged). When `semantic`, up/down are
    relabelled as simpler / more complex via _COMPLEXITY_POLARITY.
    """
    steps = sorted(by_step)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for idx, (key, label, _unit) in enumerate(specs):
        ax = axes[idx]

        xs, up, flat, down = [], [], [], []
        for v in steps:
            vals = [x for x in by_step[v].get(key, []) if _clean_val(x)]
            if not vals:
                continue
            n = len(vals)
            u = sum(1 for x in vals if x > threshold) / n * 100
            d = sum(1 for x in vals if x < -threshold) / n * 100
            xs.append(v)
            up.append(u)
            down.append(d)
            flat.append(100 - u - d)

        if not xs:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(label, fontsize=10, fontweight="bold")
            continue

        pos = np.arange(len(xs))
        up, flat, down = np.array(up), np.array(flat), np.array(down)

        if semantic and _COMPLEXITY_POLARITY.get(key, 1) < 0:
            simpler, harder = up, down           # increase = simpler (e.g. Flesch)
        elif semantic:
            simpler, harder = down, up           # increase = more complex
        if semantic:
            ax.bar(pos, simpler, color="#1A9850", label="simpler")
            ax.bar(pos, flat, bottom=simpler, color="#BBBBBB", label="~unchanged")
            ax.bar(pos, harder, bottom=simpler + flat, color="#D6604D",
                   label="more complex")
        else:
            ax.bar(pos, down, color="#E08214", label="decreased")
            ax.bar(pos, flat, bottom=down, color="#BBBBBB", label="~unchanged")
            ax.bar(pos, up, bottom=down + flat, color="#2166AC", label="increased")

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylabel("% of papers", fontsize=8)
        ax.set_xlabel("Version step", fontsize=8)
        ax.set_xticks(pos)
        ax.set_xticklabels(_step_labels(xs), fontsize=7)
        ax.set_ylim(0, 100)
        if idx == 0:
            ax.legend(fontsize=7, loc="lower center", ncol=3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")


def _plot_netchange_distribution(
    net_change: dict,
    specs: list,
    title: str,
    outpath: Path,
    semantic: bool,
    threshold: float = _STEP_THRESHOLD,
) -> None:
    """
    Histogram of the per-paper net % change (last version vs v1) for each metric.
    A single peak at 0 means genuine stability; a wide or bimodal spread means a
    near-zero mean is hiding opposing sub-populations. Median line + share of
    papers that end up simpler / more complex (or up / down) are annotated.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for idx, ((key, label, _unit), color) in enumerate(zip(specs, _COLORS)):
        ax = axes[idx]
        vals = np.array([x for x in net_change.get(key, []) if _clean_val(x)])

        if vals.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(label, fontsize=10, fontweight="bold")
            continue

        # Clip extreme outliers (1st–99th pct) for a readable x-range.
        lo, hi = np.percentile(vals, [1, 99])
        if hi <= lo:
            lo, hi = vals.min() - 1, vals.max() + 1
        clipped = vals[(vals >= lo) & (vals <= hi)]

        ax.hist(clipped, bins=30, range=(lo, hi),
                color=color, alpha=0.75, edgecolor="white")
        ax.axvline(0, color="gray", lw=1.2, linestyle="--", zorder=4)
        median = float(np.median(vals))
        ax.axvline(median, color="black", lw=1.6, zorder=5,
                   label=f"median {median:+.1f}%")

        # Share of papers per direction (|Δ| < threshold = unchanged).
        n = vals.size
        up   = (vals > threshold).sum()  / n * 100
        down = (vals < -threshold).sum() / n * 100
        if semantic and _COMPLEXITY_POLARITY.get(key, 1) < 0:
            simpler, harder = up, down
        else:
            simpler, harder = down, up
        note = (f"{simpler:.0f}% simpler\n{harder:.0f}% complex" if semantic
                else f"{up:.0f}% up\n{down:.0f}% down")
        ax.text(0.97, 0.95, note, transform=ax.transAxes, fontsize=8,
                ha="right", va="top",
                bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.85))

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("net % change (last vs v1)", fontsize=8)
        ax.set_ylabel("papers", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")


def _padded_limits(lo: float, hi: float, frac: float = 0.05) -> tuple:
    """Expand a [lo, hi] range by `frac` on each side so points never sit on the
    axis border. Falls back to a small symmetric pad when the range is degenerate."""
    span = hi - lo
    pad = frac * span if span > 0 else (abs(hi) * frac or 1.0)
    return lo - pad, hi + pad


def _plot_trajectory_scatter(
    feats: dict,
    specs: list,
    title: str,
    outpath: Path,
) -> None:
    """
    Scatter per metric: net % change (last vs v1, x) versus volatility (y) —
    the RMSE of the trajectory's residuals around its linear trend, in % of the
    v1 baseline (0 = perfectly linear, larger = more erratic). One point per
    paper, across all papers regardless of version count.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.flatten()
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for idx, ((key, label, _unit), color) in enumerate(zip(specs, _COLORS)):
        ax = axes[idx]
        xs = np.array(feats[key]["net"])
        ys = np.array(feats[key]["vol"])

        if xs.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(label, fontsize=10, fontweight="bold")
            continue

        # Clip to 1st–99th pct (x) / 99th pct (y) so outliers don't squash the cloud.
        lo, hi = np.percentile(xs, [1, 99])
        if hi <= lo:
            lo, hi = xs.min() - 1, xs.max() + 1
        yhi = float(np.percentile(ys, 99)) or 1.0
        m = (xs >= lo) & (xs <= hi)

        ax.scatter(xs[m], ys[m], s=10, alpha=0.35, color=color, edgecolors="none")
        ax.axvline(0, color="gray", lw=1.2, linestyle="--", zorder=1)
        median = float(np.median(xs))
        ax.axvline(median, color="black", lw=1.4, zorder=2,
                   label=f"median Δ {median:+.1f}%")

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("net % change (last vs v1)", fontsize=8)
        ax.set_ylabel("volatility: RMSE around trend (% of v1)", fontsize=8)
        ax.set_xlim(*_padded_limits(lo, hi))
        ax.set_ylim(*_padded_limits(0.0, yhi))
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc="upper right")

    plt.savefig(outpath, dpi=150, bbox_inches="tight")


def _plot_baseline_change_scatter(
    feats: dict,
    specs: list,
    title: str,
    outpath: Path,
) -> None:
    """
    Regression-to-the-mean view per metric: v1 baseline value (x) versus net %
    change (y), one point per paper (all version counts), coloured by version
    count. A downward trend (negative correlation) means papers that start high
    tend to drop and papers that start low tend to rise — convergence. A fitted
    line and Pearson r are annotated.

    Caveat: measurement noise alone induces some negative baseline-change
    correlation, so a mild slope is not necessarily a real effect.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.flatten()
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for idx, ((key, label, unit), color) in enumerate(zip(specs, _COLORS)):
        ax = axes[idx]
        bx = np.array(feats[key]["base"])
        ny = np.array(feats[key]["net"])

        if bx.size < 3:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(label, fontsize=10, fontweight="bold")
            continue

        # Correlation on full data; clip both axes (1–99 pct) for display only.
        r = float(np.corrcoef(bx, ny)[0, 1])
        xlo, xhi = np.percentile(bx, [1, 99])
        ylo, yhi = np.percentile(ny, [1, 99])
        m = (bx >= xlo) & (bx <= xhi) & (ny >= ylo) & (ny <= yhi)

        ax.scatter(bx[m], ny[m], s=10, alpha=0.35, color=color, edgecolors="none")
        ax.axhline(0, color="gray", lw=1.2, linestyle="--", zorder=1)

        # Least-squares trend line over the displayed range.
        if xhi > xlo:
            slope, intercept = np.polyfit(bx, ny, 1)
            xx = np.array([xlo, xhi])
            ax.plot(xx, slope * xx + intercept, color="black", lw=1.6, zorder=4,
                    label=f"fit (r={r:+.2f})")

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel(f"v1 baseline ({unit})", fontsize=8)
        ax.set_ylabel("net % change (last vs v1)", fontsize=8)
        ax.set_xlim(*_padded_limits(xlo, xhi))
        ax.set_ylim(*_padded_limits(ylo, yhi))
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc="upper right")

    plt.savefig(outpath, dpi=150, bbox_inches="tight")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def generate_source_views(paper_metrics: dict, outdir: Path,
                          spec_groups: list, source_label: str) -> None:
    """
    Produce all views for one text source (body or abstract). Derives the
    exact-v1..vN subset for the version-aligned views, computes the aggregates
    once, then writes each group's six figures into outdir/<group>/.

    spec_groups: list of (subdir, group_title, specs, semantic).
    """
    expected_versions = set(range(1, N_VERSIONS + 1))
    pm_exact = {doi: pv for doi, pv in paper_metrics.items()
                if set(pv) == expected_versions}

    by_v_norm, n_at_v_norm = aggregate_normalized(pm_exact)
    norm_paper_metrics     = build_normalized_paper_metrics(pm_exact)
    by_step                = aggregate_stepwise(pm_exact)
    net_change             = build_net_change(pm_exact)
    n_total = len(pm_exact)
    n_traj  = len(paper_metrics)

    for sub, group_title, specs, semantic in spec_groups:
        d = outdir / sub
        d.mkdir(parents=True, exist_ok=True)
        feats = build_trajectory_features(paper_metrics, [k for k, *_ in specs])
        tag = f"{group_title} ({source_label})"

        _plot_normalized_spaghetti_grid(
            norm_paper_metrics, by_v_norm, n_at_v_norm, specs,
            title   = f"{tag} — Change from v1 (%)  (n={n_total} papers)",
            outpath = d / "normalized_over_versions.png",
        )
        _plot_stepwise_grid(
            by_step, specs,
            title   = f"{tag} — Change vs Previous Version (%)  (n={n_total} papers)",
            outpath = d / "stepwise_over_versions.png",
        )
        _plot_direction_grid(
            by_step, specs, semantic=semantic,
            title   = f"{tag} — Direction of Change per Step  (n={n_total} papers)",
            outpath = d / "direction_over_versions.png",
        )
        _plot_netchange_distribution(
            net_change, specs, semantic=semantic,
            title   = f"{tag} — Distribution of Net Change (last vs v1)  (n={n_total} papers)",
            outpath = d / "distribution.png",
        )
        _plot_trajectory_scatter(
            feats, specs,
            title   = f"{tag} — Net Change vs Volatility  (n={n_traj} papers, all version counts)",
            outpath = d / "scatter.png",
        )
        _plot_baseline_change_scatter(
            feats, specs,
            title   = f"{tag} — Baseline vs Net Change  (n={n_traj} papers, all version counts)",
            outpath = d / "baseline_change.png",
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n",    default="all", help='papers to use, or "all" (default: all)')
    ap.add_argument("--seed", type=int, default=None, help="random seed for subsampling")
    args = ap.parse_args()

    n = None if args.n.lower() == "all" else int(args.n)
    # Load the broad set (>=2 consecutive versions from v1); the exactly-v1..vN
    # subset for the version-aligned views is derived inside generate_source_views.
    papers = load_papers(XML_DIR, n, args.seed, exact=False)

    if not papers:
        sys.exit("No valid papers found.")

    if not COMMON_WORDS:
        print("Warning: NLTK Brown corpus unavailable -- rare-word rate will be NaN.")

    print("Computing metrics ...", flush=True)
    # Abstract and body are analysed separately; abstracts are shorter, so they
    # use a lower word threshold and carry no structural metrics.
    pm_body     = build_paper_metrics(papers, text_key="body_text",
                                      with_structure=True, min_words=200)
    pm_abstract = build_paper_metrics(papers, text_key="abstract_text",
                                      with_structure=False, min_words=100)

    expected_versions = set(range(1, N_VERSIONS + 1))
    n_exact = sum(1 for pv in pm_body.values() if set(pv) == expected_versions)
    print(f"Papers: {len(pm_body)} total (>=2 versions), "
          f"{n_exact} with exactly v1..v{N_VERSIONS}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Corpus overview (source-independent).
    version_counts, n_all = _scan_version_distribution(XML_DIR)
    _plot_version_distribution(version_counts, n_all,
                               outpath=PLOTS_DIR / "papers_per_version.png")

    # Body: full set of views (simple + complexity).
    generate_source_views(
        pm_body, PLOTS_DIR / "body",
        spec_groups=[
            ("simple_metrics", "Simple Text Metrics",   _SIMPLE_SPECS,  False),
            ("complexity",     "Linguistic Complexity", _COMPLEX_SPECS, True),
        ],
        source_label="body",
    )

    # Abstract: only the linguistic-complexity views (structural metrics N/A).
    generate_source_views(
        pm_abstract, PLOTS_DIR / "abstract",
        spec_groups=[
            ("complexity", "Linguistic Complexity", _COMPLEX_SPECS, True),
        ],
        source_label="abstract",
    )

    print(f'Plots saved to "{PLOTS_DIR.name}/" '
          f'(papers_per_version.png + body/ + abstract/)')


if __name__ == "__main__":
    main()
