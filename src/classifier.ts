import { FeatureVector, TextClass } from './types';

/**
 * Heuristic scoring for each class based on feature thresholds.
 * Each function returns 0-1 confidence.
 */

function clamp(v: number): number {
  return Math.max(0, Math.min(1, v));
}

function sigmoid(x: number, center: number, steepness: number): number {
  return 1 / (1 + Math.exp(-steepness * (x - center)));
}

export function scoreHuman(f: FeatureVector): number {
  // Human text: high burstiness, high sentence variance, lower formality, high vocab diversity variance
  const burstFactor = sigmoid(f.burstiness, 1.5, 1.5);
  const varianceFactor = sigmoid(f.sentenceLengthVariance, 15, 0.1);
  const informalFactor = 1 - sigmoid(f.formality, 0.95, 10);
  const lowRepetition = 1 - sigmoid(f.repetitionScore, 0.1, 10);
  const lowTransition = 1 - sigmoid(f.transitionScore, 0.02, 80);
  return clamp((burstFactor * 0.25 + varianceFactor * 0.2 + informalFactor * 0.2 + lowRepetition * 0.2 + lowTransition * 0.15));
}

export function scoreLLMGenerated(f: FeatureVector): number {
  // LLM text: low burstiness, uniform sentence length, high formality, high transitions, low perplexity
  const lowBurst = 1 - sigmoid(f.burstiness, 1.5, 1.5);
  const uniformSent = 1 - sigmoid(f.sentenceLengthVariance, 20, 0.08);
  const formal = sigmoid(f.formality, 0.9, 10);
  const highTransition = sigmoid(f.transitionScore, 0.015, 80);
  const repetitive = sigmoid(f.repetitionScore, 0.05, 15);
  return clamp((lowBurst * 0.25 + uniformSent * 0.2 + formal * 0.2 + highTransition * 0.2 + repetitive * 0.15));
}

export function scoreAIPolished(f: FeatureVector): number {
  // AI-polished: moderate burstiness (some human variance remains), high formality, moderate transitions
  const midBurst = 1 - Math.abs(f.burstiness - 1.2) / 2;
  const midVariance = 1 - Math.abs(f.sentenceLengthVariance - 12) / 30;
  const formal = sigmoid(f.formality, 0.88, 8);
  const midTransition = 1 - Math.abs(f.transitionScore - 0.012) / 0.03;
  const vocabHigh = sigmoid(f.vocabularyDiversity, 0.55, 8);
  return clamp((clamp(midBurst) * 0.2 + clamp(midVariance) * 0.2 + formal * 0.2 + clamp(midTransition) * 0.2 + vocabHigh * 0.2));
}

export function scoreAIHumanized(f: FeatureVector): number {
  // AI-humanized: artificially injected variance, slightly elevated burstiness but still structured
  const fakeBurst = sigmoid(f.burstiness, 1.0, 2) * (1 - sigmoid(f.burstiness, 2.5, 2));
  const fakeVariance = sigmoid(f.sentenceLengthVariance, 10, 0.1) * (1 - sigmoid(f.sentenceLengthVariance, 40, 0.05));
  const stillFormal = sigmoid(f.formality, 0.85, 6);
  const lowPunct = 1 - sigmoid(f.punctuationDiversity, 0.5, 5);
  const midRepetition = sigmoid(f.repetitionScore, 0.03, 20) * (1 - sigmoid(f.repetitionScore, 0.15, 10));
  return clamp((fakeBurst * 0.25 + fakeVariance * 0.2 + stillFormal * 0.2 + lowPunct * 0.15 + midRepetition * 0.2));
}

export function classifyFromFeatures(features: FeatureVector): Record<TextClass, number> {
  const raw = {
    [TextClass.Human]: scoreHuman(features),
    [TextClass.LLMGenerated]: scoreLLMGenerated(features),
    [TextClass.AIPolished]: scoreAIPolished(features),
    [TextClass.AIHumanized]: scoreAIHumanized(features),
  };

  // Normalize to sum = 1
  const total = Object.values(raw).reduce((a, b) => a + b, 0);
  if (total === 0) {
    return { [TextClass.Human]: 0.25, [TextClass.LLMGenerated]: 0.25, [TextClass.AIPolished]: 0.25, [TextClass.AIHumanized]: 0.25 };
  }
  return {
    [TextClass.Human]: raw[TextClass.Human] / total,
    [TextClass.LLMGenerated]: raw[TextClass.LLMGenerated] / total,
    [TextClass.AIPolished]: raw[TextClass.AIPolished] / total,
    [TextClass.AIHumanized]: raw[TextClass.AIHumanized] / total,
  };
}
