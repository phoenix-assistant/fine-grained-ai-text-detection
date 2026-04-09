"""Tests for aidetect."""

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from aidetect import Detector, __version__
from aidetect.cli import main
from aidetect.features import extract_features, feature_vector, _sentence_split
from aidetect.detector import DocumentResult, SentenceResult


# --- Fixtures ---

HUMAN_TEXT = """I went to the store yesterday. Bought some milk and bread. 
The dog was barking at nothing again. Kids are driving me crazy today.
Forgot to call mom back. Whatever, I'll do it tomorrow."""

AI_TEXT_GPT = """It's worth noting that the landscape of artificial intelligence has become increasingly multifaceted. 
Furthermore, navigating the intricate tapestry of modern technology requires a comprehensive and nuanced understanding. 
Additionally, leveraging synergies across the paradigm of innovation is essential for holistic growth. 
In conclusion, the realm of AI continues to present fascinating opportunities for advancement."""

AI_TEXT_CLAUDE = """I appreciate that this is a complex topic, and I want to be straightforward about the limitations. 
That said, I think it's reasonable to consider multiple perspectives here. 
I should note that there are genuinely important considerations worth exploring. 
I'd suggest taking a thoughtful approach to this, and I understand if you have concerns."""

AI_TEXT_GEMINI = """Sure, here's a breakdown of the key takeaways from this analysis. 
In essence, this plays a vital role in today's world of technological advancement. 
Let's break down the crucial and significant aspects that make this compelling. 
It's crucial to understand a wide range of factors that underscore these remarkable findings."""


@pytest.fixture
def detector():
    return Detector()


@pytest.fixture
def tmp_dir(tmp_path):
    (tmp_path / "human.txt").write_text(HUMAN_TEXT)
    (tmp_path / "ai.txt").write_text(AI_TEXT_GPT)
    return tmp_path


# --- Detector Tests ---

class TestDetector:
    def test_analyze_returns_document_result(self, detector):
        result = detector.analyze(HUMAN_TEXT)
        assert isinstance(result, DocumentResult)
        assert isinstance(result.sentences, list)
        assert len(result.sentences) > 0

    def test_human_text_low_score(self, detector):
        result = detector.analyze(HUMAN_TEXT)
        assert result.aggregate_score < 0.7

    def test_ai_text_detected(self, detector):
        result = detector.analyze(AI_TEXT_GPT)
        assert result.aggregate_score > 0.2

    def test_gpt4_attribution(self, detector):
        result = detector.analyze(AI_TEXT_GPT)
        ai_sents = [s for s in result.sentences if s.label == "gpt-4"]
        assert len(ai_sents) > 0

    def test_claude_attribution(self, detector):
        result = detector.analyze(AI_TEXT_CLAUDE)
        claude_sents = [s for s in result.sentences if s.label == "claude"]
        assert len(claude_sents) > 0

    def test_gemini_attribution(self, detector):
        result = detector.analyze(AI_TEXT_GEMINI)
        gemini_sents = [s for s in result.sentences if s.label == "gemini"]
        assert len(gemini_sents) > 0

    def test_empty_text(self, detector):
        result = detector.analyze("")
        assert result.aggregate_score == 0.0
        assert result.sentences == []

    def test_sentence_spans(self, detector):
        result = detector.analyze("Hello world. This is a test.")
        for s in result.sentences:
            assert s.start >= 0
            assert s.end > s.start

    def test_to_dict(self, detector):
        result = detector.analyze(AI_TEXT_GPT)
        d = result.to_dict()
        assert "aggregate_score" in d
        assert "sentences" in d
        assert "model_distribution" in d


# --- Feature Tests ---

class TestFeatures:
    def test_extract_features(self):
        f = extract_features("This is a simple test sentence.")
        assert f["n_words"] > 0
        assert 0 <= f["vocab_richness"] <= 1

    def test_feature_vector_shape(self):
        f = extract_features("Hello world, this is a test.")
        vec = feature_vector(f)
        assert vec.shape == (14,)

    def test_empty_features(self):
        f = extract_features("")
        assert f["n_words"] == 0


# --- CLI Tests ---

class TestCLI:
    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert __version__ in result.output

    def test_analyze_text(self):
        runner = CliRunner()
        result = runner.invoke(main, ["analyze", "This is a test sentence for analysis."])
        assert result.exit_code == 0

    def test_analyze_json(self):
        runner = CliRunner()
        result = runner.invoke(main, ["analyze", "-j", "This is a test."])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "aggregate_score" in data

    def test_batch(self, tmp_dir):
        runner = CliRunner()
        result = runner.invoke(main, ["batch", str(tmp_dir)])
        assert result.exit_code == 0

    def test_batch_csv(self, tmp_dir):
        runner = CliRunner()
        result = runner.invoke(main, ["batch", str(tmp_dir), "-f", "csv"])
        assert result.exit_code == 0
        assert "file,aggregate_score" in result.output
