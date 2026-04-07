# Fine-Grained AI Text Detection

> **One-line pitch:** Not "AI or human" but "which kind of AI-human collaboration" — 4-class detection for the nuanced reality of modern content creation.

## Problem

**Who feels the pain:**
- **Universities** — Academic integrity crisis, but current tools cry wolf (false positives) or miss hybrid content
- **Enterprises** — Compliance/legal need to know if contracts, disclosures were AI-drafted
- **Publishers** — Editorial standards require knowing human involvement level
- **Recruiters/HR** — Screening AI-generated applications vs. AI-polished vs. human
- **Legal/regulatory** — SEC, FDA submissions need content provenance

**How bad:**
- **Academic:** 50%+ of students admit to using AI; Turnitin has 1-4% false positive rate (wrongly accusing humans)
- **Enterprise:** No way to verify "human-written" claims in contracts, disclosures, regulatory filings
- **Scale:** GPT writes 10%+ of new internet content; detection stuck on binary classification
- **Policy gap:** Different content types need different policies (AI-polished résumé ≠ AI-written thesis)
- **Current tools fail:** GPTZero, Turnitin optimize for binary — miss the nuanced middle ground

**The core insight:**
Binary "AI vs human" is wrong. Reality is 4 classes with different policy implications:
1. **Pure human** — Entirely human-written
2. **Pure LLM** — Entirely AI-generated  
3. **LLM-polished human** — Human draft, AI edited/improved
4. **Humanized LLM** — AI draft, human edited to evade detection

Each class has DIFFERENT policy implications. A student using Grammarly AI to polish their essay ≠ submitting pure ChatGPT output.

## Solution

**What we build:**
4-class AI text detector based on RACE methodology (ACL 2026) with policy-aware output.

**Core features:**
1. **4-Class Classification** — Confidence scores for each class, not binary output
2. **Segment-Level Analysis** — Which paragraphs/sentences are which class
3. **Policy Engine** — Configurable rules per content type (academic, legal, editorial)
4. **Explanation Layer** — Why the classification (linguistic markers, patterns)
5. **Batch Processing** — Enterprise-scale analysis (thousands of docs)
6. **API + Integrations** — LMS (Canvas, Blackboard), ATS (Greenhouse, Lever), CMS

**How it works:**
- Fine-tuned transformer model on RACE dataset (multi-class labeled)
- Ensemble approach (stylometric + perplexity + learned features)
- Calibrated confidence scores (not just softmax)
- Continual learning pipeline (new models, new evasion techniques)
- Human-in-the-loop for edge cases

**Key output:**
```json
{
  "classification": "llm_polished_human",
  "confidence": 0.78,
  "breakdown": {
    "pure_human": 0.12,
    "pure_llm": 0.05,
    "llm_polished_human": 0.78,
    "humanized_llm": 0.05
  },
  "segments": [
    {"text": "First paragraph...", "class": "pure_human", "confidence": 0.85},
    {"text": "Second paragraph...", "class": "llm_polished_human", "confidence": 0.72}
  ],
  "policy_recommendation": "acceptable_with_disclosure"
}
```

## Why Now

1. **RACE paper just published (ACL 2026)** — Methodology validated by top venue
2. **Detection arms race intensifying** — Binary tools failing, market needs sophistication
3. **Policy catching up** — Universities, enterprises developing AI use policies that need nuance
4. **Enterprise AI adoption exploding** — Every company using AI, need to audit output
5. **Regulatory pressure** — SEC considering AI disclosure rules for financial filings
6. **Trust deficit** — People don't trust binary "99% AI" verdicts anymore

## Market Landscape

**TAM:** $3B (content authentication + academic integrity + enterprise compliance)
**SAM:** $500M (AI-specific text detection, enterprise + education)
**SOM Year 1:** $3-8M (200-400 enterprise/education customers at $15-20K average)

**Competitors:**

| Company | What They Do | Gap |
|---------|--------------|-----|
| **Turnitin** | Plagiarism + AI detection | Binary only, high false positives, education-only |
| **GPTZero** | AI detection | Binary, consumer-focused, no enterprise features |
| **Originality.ai** | AI detection | Binary, content marketing focus |
| **Copyleaks** | Plagiarism + AI | Binary, weak on hybrid content |
| **Sapling** | AI detection | Binary, limited accuracy |
| **Writer.com** | Enterprise AI platform | Detection is minor feature, not core |
| **Pangram Labs** | Academic integrity | Binary, early stage |

**No one is doing:** Multi-class detection with policy-aware output and enterprise features.

## Competitive Advantages

1. **Technical moat** — RACE methodology (4-class) is state-of-art; competitors stuck on binary
2. **Policy integration** — Not just detection but "what should we do about it"
3. **Enterprise-grade** — SOC 2, batch processing, integrations vs. consumer tools
4. **Segment-level analysis** — Pinpoint exactly which parts are AI vs. human
5. **Continual learning** — Pipeline to adapt as new models/evasion techniques emerge
6. **First-mover on nuance** — "AI-polished" vs "AI-written" distinction is our brand

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Processing                          │
│  - Text normalization                                       │
│  - Chunking (sentence, paragraph, document level)           │
│  - Language detection                                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Feature Extraction                        │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│ Perplexity  │ Stylometric │ Burstiness  │ Learned         │
│ (per-model) │ Features    │ Analysis    │ Embeddings      │
└──────┬──────┴──────┬──────┴──────┬──────┴────────┬─────────┘
       │             │             │               │
       ▼             ▼             ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│              Ensemble Classification Model                   │
│  - Fine-tuned RoBERTa backbone                              │
│  - Multi-class head (4 classes)                             │
│  - Calibration layer (reliable confidence scores)           │
│  - Segment-level + document-level outputs                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Policy Engine                             │
│  - Customer-configurable rules                              │
│  - Context-aware recommendations                            │
│  - Threshold management                                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output Layer                              │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│ API         │ Dashboard   │ Integrations│ Reports          │
│ (REST/gRPC) │ (Review UI) │ (LMS, ATS)  │ (PDF, CSV)       │
└─────────────┴─────────────┴─────────────┴──────────────────┘
```

**Tech stack:**
- PyTorch model serving (TorchServe or custom)
- FastAPI for API layer
- PostgreSQL for results/audit trail
- Redis for caching/rate limiting
- React dashboard for review queue
- Kubernetes for scaling

**Model training:**
- RACE dataset as foundation
- Synthetic data augmentation (controlled AI-human mixing)
- Continual learning pipeline (weekly retraining on new samples)
- Red team for evasion testing

## Build Plan

### Phase 1: Core Model (Months 1-4)
- [ ] Implement RACE methodology (4-class detection)
- [ ] Build evaluation benchmark (accuracy, calibration, segment-level)
- [ ] Train on RACE dataset + augmented data
- [ ] Basic API with classification output
- [ ] Land 5 design partners (2 universities, 3 enterprises)
- [ ] Accuracy target: 85%+ on 4-class, 95%+ on binary
- **Milestone:** Working model, 5 pilots, $50K pilot revenue

### Phase 2: Enterprise Features (Months 5-9)
- [ ] Segment-level analysis (which sentences are which)
- [ ] Policy engine (configurable rules)
- [ ] LMS integrations (Canvas, Blackboard, Moodle)
- [ ] Batch processing (100K+ docs/day)
- [ ] Dashboard for review queue
- [ ] SOC 2 Type 1 certification
- [ ] Sales team (2-3 reps for education + enterprise)
- **Milestone:** $400K ARR, 50 paying customers, top 3 LMS integrations

### Phase 3: Market Expansion (Months 10-18)
- [ ] ATS integrations (Greenhouse, Lever, Workday)
- [ ] Legal/compliance vertical (contract review, regulatory filings)
- [ ] Publisher vertical (editorial workflow integration)
- [ ] Multi-language support (Spanish, Chinese, French)
- [ ] On-premise deployment option for regulated industries
- [ ] SOC 2 Type 2 + FERPA compliance
- **Milestone:** $1.5M ARR, 150+ customers, recognized as "nuanced detection" leader

## Risks & Challenges

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Detection arms race (evasion improves)** | High | Continual learning, red team, rapid iteration |
| **RACE methodology doesn't generalize** | High | Augmented training data, ensemble approach, custom fine-tuning |
| **Turnitin copies 4-class approach** | Medium | Move fast, lock in integrations, policy engine differentiation |
| **False positive PR disasters** | High | Conservative thresholds, human review layer, clear confidence scores |
| **Education sales cycle (slow, bureaucratic)** | Medium | Enterprise focus first, education through channel partners |
| **Model costs at scale** | Medium | Efficient serving, tiered pricing, caching |

## Monetization

**Pricing tiers:**

| Tier | Price | Includes |
|------|-------|----------|
| **Starter (Education)** | $5K/year | 10K documents/year, basic dashboard, email support |
| **Professional** | $15K/year | 100K documents/year, API access, integrations, policy engine |
| **Enterprise** | $50K+/year | Unlimited, custom integrations, SLA, dedicated support |
| **API-only** | $0.02/document | Pay-as-you-go for developers |

**Path to $1M ARR:**
- 30 Professional ($15K) = $450K
- 8 Enterprise ($50K) = $400K  
- API usage = $150K
- **Total: $1M ARR in 12-14 months**

**Expansion revenue:**
- Upsell to enterprise from professional (natural expansion)
- New verticals (legal, publishing) at higher price points
- Custom model training for large customers

## Verdict

### 🟢 BUILD

**Rationale:**

1. **Clear technical differentiation** — 4-class detection is genuinely better than binary; RACE paper validates approach
2. **Urgent market need** — Academic integrity crisis, enterprise compliance demands growing
3. **Strong monetization** — Education and enterprise both pay well for detection tools
4. **Defensible moat** — Continual learning + policy engine + integrations compound
5. **Timing** — Binary tools are failing; market ready for sophistication
6. **Acquirable** — Turnitin, Grammarly, or enterprise compliance players would pay for this

**Concerns:**
- Arms race is real — need to invest in continual learning
- Education sales can be slow — lead with enterprise
- RACE paper is new — need to validate methodology works at scale

**Recommendation:** Start with enterprise (faster sales, higher ACV), prove 4-class approach works, then expand to education through partnerships. The "nuanced detection" positioning is defensible and differentiated.

**Key insight:** The market doesn't need better binary detection — it needs detection that matches how people actually use AI. That's 4 classes, not 2.
