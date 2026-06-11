import json
import re
from pathlib import Path


def load_paper(json_path: str) -> dict:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def clean_text(raw_text: str) -> str:
    """
    Convert the raw `text` field to plain prose.

    Handles two formats found in the corpus:
      A) Pipe-delimited rows:  | N | cell content |
      B) Standalone line numbers embedded among prose paragraphs

    Also collapses multi-space PDF-extraction artifacts.
    """
    lines = raw_text.split("\n")
    parts = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Separator row like |-----|
        if re.match(r"^\|[-\s|]+\|$", stripped):
            continue
        # Pipe-delimited data row: | N | content |
        if stripped.startswith("|"):
            m = re.match(r"^\|\s*\d+\s*\|\s*(.*?)\s*\|?\s*$", stripped)
            if m:
                cell = m.group(1).strip()
                if cell and re.search(r"[a-zA-Z]", cell):
                    parts.append(cell)
            continue
        # Standalone page/line number — discard
        if re.match(r"^\d+$", stripped):
            continue
        # Narrative prose (must contain at least one letter)
        if re.search(r"[a-zA-Z]", stripped):
            parts.append(stripped)

    joined = " ".join(parts)
    # Collapse multi-space PDF artifacts
    return re.sub(r" {2,}", " ", joined).strip()


def get_abstract(paper: dict) -> str:
    return paper.get("metadata", {}).get("pub_metadata", {}).get("preprint_abstract", "")


def extract_section(cleaned_text: str, section_keyword: str, max_chars: int = 3000) -> str:
    """
    Find a named section heading and return up to max_chars characters from there.
    Falls back to the first max_chars characters if the section is not found.
    """
    idx = cleaned_text.upper().find(section_keyword.upper())
    if idx == -1:
        return cleaned_text[:max_chars]
    return cleaned_text[idx: idx + max_chars]
