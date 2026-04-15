import express from 'express';
import { detect, detectBatch } from './ensemble';

export function createApp() {
  const app = express();
  app.use(express.json({ limit: '5mb' }));

  app.get('/health', (_req, res) => res.json({ status: 'ok' }));

  app.post('/detect', (req, res) => {
    const { text } = req.body;
    if (!text || typeof text !== 'string') {
      return res.status(400).json({ error: 'Missing "text" field' });
    }
    res.json(detect(text));
  });

  app.post('/detect/batch', (req, res) => {
    const { texts } = req.body;
    if (!Array.isArray(texts) || !texts.every(t => typeof t === 'string')) {
      return res.status(400).json({ error: 'Missing "texts" array of strings' });
    }
    res.json(detectBatch(texts));
  });

  return app;
}

if (require.main === module) {
  const port = parseInt(process.env.PORT || '3000', 10);
  createApp().listen(port, () => console.log(`AI Text Detection API on port ${port}`));
}
