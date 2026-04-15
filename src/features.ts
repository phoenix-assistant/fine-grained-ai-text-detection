import { FeatureVector } from './types';

function tokenize(text: string): string[] {
  return text.toLowerCase().match(/\b[a-z']+\b/g) || [];
}

function sentences(text: string): string[] {
  return text.split(/[.!?]+/).map(s => s.trim()).filter(Boolean);
}

function mean(arr: number[]): number {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

function variance(arr: number[]): number {
  const m = mean(arr);
  return arr.length > 1 ? arr.reduce((s, v) => s + (v - m) ** 2, 0) / (arr.length - 1) : 0;
}

/** Perplexity proxy: inverse of bigram predictability */
export function perplexityProxy(tokens: string[]): number {
  if (tokens.length < 2) return 0;
  const bigrams = new Map<string, number>();
  const unigrams = new Map<string, number>();
  for (let i = 0; i < tokens.length - 1; i++) {
    const bg = `${tokens[i]} ${tokens[i + 1]}`;
    bigrams.set(bg, (bigrams.get(bg) || 0) + 1);
    unigrams.set(tokens[i], (unigrams.get(tokens[i]) || 0) + 1);
  }
  let logProb = 0;
  for (let i = 0; i < tokens.length - 1; i++) {
    const bg = `${tokens[i]} ${tokens[i + 1]}`;
    const p = (bigrams.get(bg) || 1) / (unigrams.get(tokens[i]) || 1);
    logProb += Math.log2(p);
  }
  const avgLogProb = logProb / (tokens.length - 1);
  return Math.pow(2, -avgLogProb);
}

/** Burstiness: variance-to-mean ratio of word frequencies */
export function burstiness(tokens: string[]): number {
  if (!tokens.length) return 0;
  const freq = new Map<string, number>();
  tokens.forEach(t => freq.set(t, (freq.get(t) || 0) + 1));
  const counts = [...freq.values()];
  const m = mean(counts);
  const v = variance(counts);
  return m > 0 ? v / m : 0;
}

/** Vocabulary diversity: unique/total ratio */
export function vocabularyDiversity(tokens: string[]): number {
  if (!tokens.length) return 0;
  return new Set(tokens).size / tokens.length;
}

/** Sentence length variance */
export function sentenceLengthVariance(text: string): number {
  const s = sentences(text);
  const lengths = s.map(sent => tokenize(sent).length);
  return variance(lengths);
}

/** Transition word density */
export function transitionScore(text: string): number {
  const transitions = ['however', 'moreover', 'furthermore', 'additionally', 'consequently',
    'therefore', 'nevertheless', 'nonetheless', 'in conclusion', 'in addition',
    'on the other hand', 'as a result', 'in contrast', 'similarly', 'meanwhile'];
  const lower = text.toLowerCase();
  const tokens = tokenize(lower);
  if (!tokens.length) return 0;
  let count = 0;
  transitions.forEach(t => {
    const re = new RegExp(`\\b${t}\\b`, 'g');
    const matches = lower.match(re);
    if (matches) count += matches.length;
  });
  return count / tokens.length;
}

/** Repetition: ratio of repeated n-grams */
export function repetitionScore(tokens: string[]): number {
  if (tokens.length < 3) return 0;
  const trigrams = new Map<string, number>();
  for (let i = 0; i < tokens.length - 2; i++) {
    const tg = `${tokens[i]} ${tokens[i + 1]} ${tokens[i + 2]}`;
    trigrams.set(tg, (trigrams.get(tg) || 0) + 1);
  }
  const repeated = [...trigrams.values()].filter(v => v > 1).length;
  return repeated / trigrams.size;
}

/** Formality: ratio of formal markers */
export function formalityScore(text: string): number {
  const tokens = tokenize(text);
  if (!tokens.length) return 0;
  const contractions = text.match(/\b\w+'\w+\b/g)?.length || 0;
  const slang = ['gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'yeah', 'nah', 'lol', 'omg'];
  const slangCount = tokens.filter(t => slang.includes(t)).length;
  return 1 - (contractions + slangCount) / tokens.length;
}

/** Punctuation diversity */
export function punctuationDiversity(text: string): number {
  const puncts = text.match(/[^\w\s]/g) || [];
  if (!puncts.length) return 0;
  return new Set(puncts).size / puncts.length;
}

/** Word length variance */
export function wordLengthVariance(tokens: string[]): number {
  const lengths = tokens.map(t => t.length);
  return variance(lengths);
}

export function extractFeatures(text: string): FeatureVector {
  const tokens = tokenize(text);
  const sents = sentences(text);
  const sentLengths = sents.map(s => tokenize(s).length);

  return {
    perplexityProxy: perplexityProxy(tokens),
    burstiness: burstiness(tokens),
    vocabularyDiversity: vocabularyDiversity(tokens),
    sentenceLengthVariance: sentenceLengthVariance(text),
    avgSentenceLength: mean(sentLengths),
    repetitionScore: repetitionScore(tokens),
    transitionScore: transitionScore(text),
    formality: formalityScore(text),
    punctuationDiversity: punctuationDiversity(text),
    wordLengthVariance: wordLengthVariance(tokens),
  };
}
