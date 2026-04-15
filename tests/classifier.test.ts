import { detect } from '../src/ensemble';
import { extractFeatures } from '../src/features';
import { TextClass } from '../src/types';

describe('Feature extraction', () => {
  it('returns all feature fields', () => {
    const f = extractFeatures('This is a simple test sentence. Here is another one.');
    expect(f).toHaveProperty('perplexityProxy');
    expect(f).toHaveProperty('burstiness');
    expect(f).toHaveProperty('vocabularyDiversity');
    expect(f).toHaveProperty('sentenceLengthVariance');
    expect(f).toHaveProperty('formality');
    expect(typeof f.perplexityProxy).toBe('number');
  });

  it('handles empty text', () => {
    const f = extractFeatures('');
    expect(f.vocabularyDiversity).toBe(0);
  });
});

describe('Ensemble detection', () => {
  it('returns a valid classification', () => {
    const result = detect('The quick brown fox jumps over the lazy dog.');
    expect(Object.values(TextClass)).toContain(result.label);
    expect(result.confidence).toBeGreaterThan(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
  });

  it('scores sum to 1', () => {
    const result = detect('Furthermore, it is important to consider the implications of this approach. Moreover, the methodology demonstrates significant improvements across all metrics.');
    const sum = Object.values(result.scores).reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 5);
  });

  it('detects highly formal LLM-like text', () => {
    const llmText = 'Furthermore, it is essential to acknowledge that the aforementioned methodology yields consistently superior results. Moreover, the implementation demonstrates remarkable efficiency. Additionally, the comprehensive analysis reveals significant improvements. Consequently, we can conclude that this approach is optimal. Nevertheless, further research is warranted to validate these findings.';
    const result = detect(llmText);
    // LLM or AI-polished should score high for very formal, transition-heavy text
    expect([TextClass.LLMGenerated, TextClass.AIPolished, TextClass.AIHumanized]).toContain(result.label);
  });

  it('provides features in result', () => {
    const result = detect('Hello world this is a test.');
    expect(result.features).toBeDefined();
    expect(result.features.vocabularyDiversity).toBeGreaterThan(0);
  });
});
