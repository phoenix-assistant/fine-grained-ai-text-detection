import { extractFeatures } from './features';
import { classifyFromFeatures } from './classifier';
import { ClassificationResult, TextClass } from './types';

export function detect(text: string): ClassificationResult {
  const features = extractFeatures(text);
  const scores = classifyFromFeatures(features);

  let best: TextClass = TextClass.Human;
  let bestScore = 0;
  for (const [cls, score] of Object.entries(scores)) {
    if (score > bestScore) {
      bestScore = score;
      best = cls as TextClass;
    }
  }

  return {
    label: best,
    confidence: bestScore,
    scores,
    features,
  };
}

export function detectBatch(texts: string[]): ClassificationResult[] {
  return texts.map(detect);
}
