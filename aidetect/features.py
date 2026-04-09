"""Feature extraction for AI text detection."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import List, Dict, Any

import numpy as np


# Model-specific vocabulary signatures (common phrases/patterns)
MODEL_SIGNATURES: Dict[str, List[str]] = {
    "gpt-4": [
        "delve", "it's worth noting", "it's important to note",
        "in the realm of", "navigating", "landscape", "multifaceted",
        "holistic", "synergy", "leverage", "paradigm", "tapestry",
        "intricate", "comprehensive", "nuanced", "furthermore",
        "additionally", "in conclusion", "to summarize",
        "let me", "i'd be happy to", "great question",
    ],
    "claude": [
        "i appreciate", "that said", "i should note", "i want to be",
        "straightforward", "genuinely", "honestly", "i think",
        "reasonable", "fair to say", "it seems", "i'd suggest",
        "thoughtful", "certainly", "absolutely", "i understand",
        "helpful", "happy to help", "let me think",
        "i should mention", "worth considering",
    ],
    "gemini": [
        "sure", "here's", "here is", "let's break",
        "key takeaway", "in essence", "essentially",
        "crucial", "significant", "notable", "compelling",
        "fascinating", "remarkable", "underscores",
        "it's crucial to", "plays a vital role",
        "a wide range of", "in today's world",
    ],
}

# Filler / hedge words common in AI text
AI_FILLER = {
    "moreover", "furthermore", "additionally", "consequently",
    "nevertheless", "nonetheless", "notwithstanding", "henceforth",
    "thereby", "wherein", "thereof", "hereby",
}

# Transition phrases over-used by AI
AI_TRANSITIONS = [
    "in addition to", "on the other hand", "as a result",
    "in light of", "with regard to", "in terms of",
    "it is important to", "it should be noted",
    "plays a crucial role", "is a testament to",
]


def _word_tokenize(text: str) -> List[str]:
    """Simple word tokenizer."""
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())


def _sentence_split(text: str) -> List[str]:
    """Split text into sentences."""
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]


def extract_features(text: str) -> Dict[str, Any]:
    """Extract statistical and pattern-based features from text."""
    words = _word_tokenize(text)
    if not words:
        return _empty_features()

    n_words = len(words)
    n_chars = len(text)
    unique_words = set(words)

    # Basic stats
    avg_word_len = np.mean([len(w) for w in words])
    vocab_richness = len(unique_words) / n_words if n_words else 0

    # Sentence-level stats
    sentences = _sentence_split(text)
    sent_lengths = [len(_word_tokenize(s)) for s in sentences]
    avg_sent_len = np.mean(sent_lengths) if sent_lengths else 0
    sent_len_std = np.std(sent_lengths) if len(sent_lengths) > 1 else 0

    # Burstiness: how variable sentence lengths are (humans are burstier)
    burstiness = sent_len_std / avg_sent_len if avg_sent_len > 0 else 0

    # Punctuation density
    punct_count = sum(1 for c in text if c in string.punctuation)
    punct_density = punct_count / n_chars if n_chars else 0

    # Comma density (AI tends to use more commas)
    comma_density = text.count(",") / n_words if n_words else 0

    # Word frequency distribution (entropy as perplexity proxy)
    freq = Counter(words)
    probs = np.array(list(freq.values())) / n_words
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    # AI filler word ratio
    filler_count = sum(1 for w in words if w in AI_FILLER)
    filler_ratio = filler_count / n_words

    # Transition phrase count
    text_lower = text.lower()
    transition_count = sum(1 for t in AI_TRANSITIONS if t in text_lower)
    transition_density = transition_count / max(len(sentences), 1)

    # Model signature scores
    signature_scores = {}
    for model, phrases in MODEL_SIGNATURES.items():
        hits = sum(1 for p in phrases if p in text_lower)
        signature_scores[model] = hits / len(phrases)

    # Repetition: how often bigrams repeat
    if n_words > 1:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(n_words - 1)]
        bigram_freq = Counter(bigrams)
        repeated_bigrams = sum(1 for c in bigram_freq.values() if c > 1)
        bigram_repetition = repeated_bigrams / len(bigrams)
    else:
        bigram_repetition = 0

    # Starts with capital ratio (AI almost always starts sentences with caps)
    cap_starts = sum(1 for s in sentences if s and s[0].isupper())
    cap_start_ratio = cap_starts / max(len(sentences), 1)

    return {
        "n_words": n_words,
        "avg_word_len": float(avg_word_len),
        "vocab_richness": float(vocab_richness),
        "avg_sent_len": float(avg_sent_len),
        "sent_len_std": float(sent_len_std),
        "burstiness": float(burstiness),
        "punct_density": float(punct_density),
        "comma_density": float(comma_density),
        "entropy": float(entropy),
        "filler_ratio": float(filler_ratio),
        "transition_density": float(transition_density),
        "bigram_repetition": float(bigram_repetition),
        "cap_start_ratio": float(cap_start_ratio),
        "signature_scores": signature_scores,
    }


def _empty_features() -> Dict[str, Any]:
    return {
        "n_words": 0, "avg_word_len": 0, "vocab_richness": 0,
        "avg_sent_len": 0, "sent_len_std": 0, "burstiness": 0,
        "punct_density": 0, "comma_density": 0, "entropy": 0,
        "filler_ratio": 0, "transition_density": 0,
        "bigram_repetition": 0, "cap_start_ratio": 0,
        "signature_scores": {"gpt-4": 0, "claude": 0, "gemini": 0},
    }


def feature_vector(features: Dict[str, Any]) -> np.ndarray:
    """Convert feature dict to numeric vector for classification."""
    sig = features["signature_scores"]
    return np.array([
        features["avg_word_len"],
        features["vocab_richness"],
        features["avg_sent_len"],
        features["burstiness"],
        features["punct_density"],
        features["comma_density"],
        features["entropy"],
        features["filler_ratio"],
        features["transition_density"],
        features["bigram_repetition"],
        features["cap_start_ratio"],
        sig.get("gpt-4", 0),
        sig.get("claude", 0),
        sig.get("gemini", 0),
    ])
