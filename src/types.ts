export enum TextClass {
  Human = 'human',
  LLMGenerated = 'llm-generated',
  AIPolished = 'ai-polished',
  AIHumanized = 'ai-humanized',
}

export interface FeatureVector {
  perplexityProxy: number;
  burstiness: number;
  vocabularyDiversity: number;
  sentenceLengthVariance: number;
  avgSentenceLength: number;
  repetitionScore: number;
  transitionScore: number;
  formality: number;
  punctuationDiversity: number;
  wordLengthVariance: number;
}

export interface ClassificationResult {
  label: TextClass;
  confidence: number;
  scores: Record<TextClass, number>;
  features: FeatureVector;
}

export interface BatchResult {
  file: string;
  result: ClassificationResult;
}
