# aidetect

[![CI](https://github.com/phoenix-assistant/fine-grained-ai-text-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/phoenix-assistant/fine-grained-ai-text-detection/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Fine-grained AI text detection with sentence-level model attribution.** Not just "is this AI?" — but "which AI wrote this sentence?"

## What It Does

- 🔍 **Sentence-level analysis** — each sentence gets its own AI/human label
- 🏷️ **Model attribution** — identifies GPT-4, Claude, or Gemini patterns
- 📊 **Confidence scores** — per-sentence and aggregate document scores
- 🚀 **No API required** — runs entirely offline using statistical features

## Quick Start

```bash
pip install .

# Analyze text directly
aidetect analyze "It's worth noting that the landscape of AI is multifaceted."

# Analyze a file
aidetect analyze paper.txt

# JSON output
aidetect analyze -j paper.txt

# Batch analyze a directory
aidetect batch ./documents/ -f csv -o results.csv

# Start MCP server
pip install ".[mcp]"
aidetect serve
```

## Python API

```python
from aidetect import Detector

detector = Detector()
result = detector.analyze("""
It's worth noting that the landscape of artificial intelligence 
has become increasingly multifaceted. Furthermore, navigating the 
intricate tapestry of modern technology requires a comprehensive 
and nuanced understanding.
""")

print(f"AI Score: {result.aggregate_score:.1%}")
print(f"Dominant Model: {result.dominant_model}")

for sent in result.sentences:
    print(f"  [{sent.label}] ({sent.confidence:.0%}) {sent.text[:60]}")
```

## Methodology

aidetect uses a multi-signal approach combining:

1. **Statistical features** — vocabulary richness, sentence length variance (burstiness), word length distribution, entropy as a perplexity proxy
2. **Pattern matching** — model-specific vocabulary signatures (e.g., GPT-4's overuse of "delve", "multifaceted", "tapestry"; Claude's hedging patterns; Gemini's "sure, here's")
3. **Structural analysis** — punctuation density, comma patterns, transition phrase frequency, bigram repetition

Each sentence is independently scored and attributed. Document-level scores aggregate sentence results weighted by confidence.

### Detection Signals by Model

| Signal | GPT-4 | Claude | Gemini |
|--------|-------|--------|--------|
| "delve", "multifaceted" | ✅ | | |
| "I appreciate", hedging | | ✅ | |
| "Sure, here's" | | | ✅ |
| High formality | ✅ | | ✅ |
| Self-referential caution | | ✅ | |

## Accuracy Benchmarks

> ⚠️ **v0.1.0 — Heuristic baseline.** This release uses statistical features and pattern matching without ML training data. Accuracy will improve significantly with trained classifiers in future releases.

| Metric | Value |
|--------|-------|
| Human text (true negative) | ~70-80% |
| AI text detection (true positive) | ~60-75% |
| Model attribution accuracy | ~40-60% |

These are estimates. Formal benchmarks with labeled datasets coming in v0.2.0.

## Comparison

| Feature | aidetect | GPTZero | Originality.ai |
|---------|----------|---------|-----------------|
| Sentence-level | ✅ | ✅ | ❌ |
| Model attribution | ✅ | ❌ | ❌ |
| Offline/local | ✅ | ❌ | ❌ |
| Open source | ✅ | ❌ | ❌ |
| API required | ❌ | ✅ | ✅ |
| ML-trained | ❌ (v0.1) | ✅ | ✅ |
| Free | ✅ | Freemium | Paid |

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Roadmap

- [ ] Train classifiers on labeled AI/human text corpora
- [ ] Add more model signatures (Llama, Mistral, etc.)
- [ ] Perplexity scoring with local language models
- [ ] Browser extension
- [ ] Confidence calibration with ground truth

## License

MIT
