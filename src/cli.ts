#!/usr/bin/env node
import * as fs from 'fs';
import { detect } from './ensemble';

function printResult(label: string, result: ReturnType<typeof detect>) {
  console.log(`\n📝 ${label}`);
  console.log(`   Classification: ${result.label}`);
  console.log(`   Confidence:     ${(result.confidence * 100).toFixed(1)}%`);
  console.log('   Scores:');
  for (const [cls, score] of Object.entries(result.scores)) {
    const bar = '█'.repeat(Math.round(score * 20));
    console.log(`     ${cls.padEnd(16)} ${(score * 100).toFixed(1).padStart(5)}% ${bar}`);
  }
}

function main() {
  const args = process.argv.slice(2);
  if (!args.length) {
    console.log('Usage: ai-detect <text>');
    console.log('       ai-detect scan <file>');
    process.exit(1);
  }

  if (args[0] === 'scan') {
    const file = args[1];
    if (!file) { console.error('Missing file path'); process.exit(1); }
    if (!fs.existsSync(file)) { console.error(`File not found: ${file}`); process.exit(1); }
    const content = fs.readFileSync(file, 'utf-8');
    const result = detect(content);
    printResult(file, result);
  } else {
    const text = args.join(' ');
    const result = detect(text);
    printResult('Input text', result);
  }
}

main();
