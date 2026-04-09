"""Core detector: sentence-level AI text attribution."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

from aidetect.features import extract_features, MODEL_SIGNATURES


@dataclass
class SentenceResult:
    text: str
    start: int
    end: int
    label: str  # "human", "gpt-4", "claude", "gemini"
    confidence: float
    features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("features", None)
        return d


@dataclass
class DocumentResult:
    text: str
    sentences: List[SentenceResult]
    aggregate_score: float  # 0=human, 1=AI
    dominant_model: Optional[str]
    model_distribution: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aggregate_score": round(self.aggregate_score, 3),
            "dominant_model": self.dominant_model,
            "model_distribution": {k: round(v, 3) for k, v in self.model_distribution.items()},
            "sentences": [s.to_dict() for s in self.sentences],
        }


def _split_sentences(text: str) -> List[tuple]:
    """Split text into (sentence, start, end) tuples."""
    results = []
    for m in re.finditer(r'[^.!?\n]+[.!?]*', text):
        s = m.group().strip()
        if s and len(s.split()) >= 2:
            results.append((s, m.start(), m.end()))
    return results


class Detector:
    """Sentence-level AI text detector with model attribution."""

    MODELS = ["gpt-4", "claude", "gemini"]

    def __init__(self):
        # Thresholds tuned for heuristic detection
        self._ai_threshold = 0.45

    def analyze(self, text: str) -> DocumentResult:
        """Analyze text and return sentence-level AI attribution."""
        sent_tuples = _split_sentences(text)
        if not sent_tuples:
            return DocumentResult(
                text=text, sentences=[], aggregate_score=0.0,
                dominant_model=None, model_distribution={},
            )

        results = []
        for sent_text, start, end in sent_tuples:
            result = self._analyze_sentence(sent_text, start, end)
            results.append(result)

        return self._aggregate(text, results)

    def _analyze_sentence(self, text: str, start: int, end: int) -> SentenceResult:
        features = extract_features(text)
        sig_scores = features["signature_scores"]

        # Compute AI probability from multiple signals
        ai_signals = []

        # Low burstiness = more AI-like (within a sentence, check uniformity)
        if features["avg_word_len"] > 4.8:
            ai_signals.append(0.6)
        if features["comma_density"] > 0.08:
            ai_signals.append(0.5)
        if features["filler_ratio"] > 0.02:
            ai_signals.append(0.7)
        if features["cap_start_ratio"] >= 1.0 and features["n_words"] > 5:
            ai_signals.append(0.3)

        # Signature matches are strong signals
        max_sig_model = max(sig_scores, key=sig_scores.get)
        max_sig_score = sig_scores[max_sig_model]

        if max_sig_score > 0.05:
            ai_signals.append(min(0.9, max_sig_score * 5))

        # Transition density
        if features["transition_density"] > 0.3:
            ai_signals.append(0.6)

        # Average word length (AI tends to use longer words)
        if features["avg_word_len"] > 5.2:
            ai_signals.append(0.5)

        if not ai_signals:
            ai_prob = 0.15  # baseline
        else:
            ai_prob = 1 - (1 - max(ai_signals)) * (1 - sum(ai_signals) / len(ai_signals) * 0.5)
            ai_prob = min(ai_prob, 0.98)

        # Determine label
        if ai_prob < self._ai_threshold:
            label = "human"
            confidence = 1 - ai_prob
        else:
            # Attribute to specific model
            if max_sig_score > 0.01:
                label = max_sig_model
            else:
                # Default to GPT-4 as most common
                label = "gpt-4"
            confidence = ai_prob

        return SentenceResult(
            text=text, start=start, end=end,
            label=label, confidence=round(confidence, 3),
            features=features,
        )

    def _aggregate(self, text: str, sentences: List[SentenceResult]) -> DocumentResult:
        if not sentences:
            return DocumentResult(text=text, sentences=[], aggregate_score=0.0,
                                  dominant_model=None, model_distribution={})

        ai_sentences = [s for s in sentences if s.label != "human"]
        ai_ratio = len(ai_sentences) / len(sentences)

        # Model distribution
        model_counts: Dict[str, int] = {}
        for s in sentences:
            model_counts[s.label] = model_counts.get(s.label, 0) + 1

        total = len(sentences)
        model_dist = {k: v / total for k, v in model_counts.items()}

        # Dominant model (excluding human)
        ai_models = {k: v for k, v in model_counts.items() if k != "human"}
        dominant = max(ai_models, key=ai_models.get) if ai_models else None

        # Aggregate score: weighted by confidence
        if ai_sentences:
            avg_conf = sum(s.confidence for s in ai_sentences) / len(ai_sentences)
            agg_score = ai_ratio * avg_conf
        else:
            agg_score = 0.0

        return DocumentResult(
            text=text, sentences=sentences,
            aggregate_score=round(agg_score, 3),
            dominant_model=dominant,
            model_distribution=model_dist,
        )
