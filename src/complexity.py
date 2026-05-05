import re


# ---------------------------------------------------------------------------
# NLTK setup (lazy, graceful fallback)
# ---------------------------------------------------------------------------

def _setup_nltk() -> bool:
    """Download required NLTK data on first use. Returns False on failure."""
    try:
        import nltk
        for resource, category in [
            ("punkt_tab", "tokenizers"),
            ("averaged_perceptron_tagger_eng", "taggers"),
        ]:
            try:
                nltk.data.find(f"{category}/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Syllable counting (pure Python, vowel-group heuristic)
# ---------------------------------------------------------------------------

def _count_syllables(word: str) -> int:
    word = word.lower().strip("'-.")
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # Silent terminal 'e' (e.g. "membrane" -> 2 not 3)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------

def analyze_complexity(text: str) -> dict:
    """
    Compute a flat dict of text complexity metrics.

    Tier 1 (textstat — classic readability formulas):
        flesch_reading_ease, flesch_kincaid_grade, gunning_fog,
        smog_index, coleman_liau_index, ari, dale_chall

    Tier 2 (pure Python + optional NLTK):
        type_token_ratio, avg_sentence_length_words, avg_word_length_chars,
        complex_word_ratio, lexical_density

    Metadata:
        word_count, sentence_count, char_count
    """
    metrics = {}

    # --- Basic counts ---
    word_tokens = re.findall(r"[a-zA-Z']+", text)
    word_count = len(word_tokens)
    char_count = len(text)

    # Simple sentence splitter: .!? followed by space + uppercase letter
    sentence_boundaries = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    sentence_count = max(1, len(sentence_boundaries))

    metrics["word_count"] = word_count
    metrics["sentence_count"] = sentence_count
    metrics["char_count"] = char_count

    # --- Tier 1: textstat ---
    try:
        import textstat
        metrics["flesch_reading_ease"] = textstat.flesch_reading_ease(text)
        metrics["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(text)
        metrics["gunning_fog"] = textstat.gunning_fog(text)
        metrics["smog_index"] = textstat.smog_index(text)
        metrics["coleman_liau_index"] = textstat.coleman_liau_index(text)
        metrics["ari"] = textstat.automated_readability_index(text)
        metrics["dale_chall"] = textstat.dale_chall_readability_score(text)
    except ImportError:
        for key in ("flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog",
                    "smog_index", "coleman_liau_index", "ari", "dale_chall"):
            metrics[key] = None
        metrics["_warning_textstat"] = "pip install textstat"

    # --- Tier 2a: pure-Python lexical metrics ---
    lowercase_tokens = [w.lower() for w in word_tokens]

    metrics["type_token_ratio"] = (
        round(len(set(lowercase_tokens)) / word_count, 4) if word_count > 0 else 0.0
    )
    metrics["avg_word_length_chars"] = (
        round(sum(len(w) for w in word_tokens) / word_count, 2) if word_count > 0 else 0.0
    )
    metrics["avg_sentence_length_words"] = (
        round(word_count / sentence_count, 2) if sentence_count > 0 else 0.0
    )

    complex_words = [w for w in word_tokens if _count_syllables(w) >= 3]
    metrics["complex_word_ratio"] = (
        round(len(complex_words) / word_count, 4) if word_count > 0 else 0.0
    )

    # --- Tier 2b: lexical density via NLTK POS tagging ---
    CONTENT_POS = {
        "NN", "NNS", "NNP", "NNPS",                        # nouns
        "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",          # verbs
        "JJ", "JJR", "JJS",                                 # adjectives
        "RB", "RBR", "RBS",                                 # adverbs
    }

    if _setup_nltk():
        try:
            import nltk
            all_tagged = []
            for sent in nltk.sent_tokenize(text):
                all_tagged.extend(nltk.pos_tag(nltk.word_tokenize(sent)))
            total = len(all_tagged)
            content = sum(1 for _, tag in all_tagged if tag in CONTENT_POS)
            metrics["lexical_density"] = round(content / total, 4) if total > 0 else 0.0
        except Exception as e:
            metrics["lexical_density"] = None
            metrics["_warning_lexical_density"] = str(e)
    else:
        metrics["lexical_density"] = None

    return metrics
