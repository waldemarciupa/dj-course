import Anthropic from '@anthropic-ai/sdk/index.js';
import { countTokens, getTokenizer } from '@anthropic-ai/tokenizer';

import * as dotenv from 'dotenv';
dotenv.config();

if (!process.env.ANTHROPIC_API_KEY) {
  throw new Error('Please set the ANTHROPIC_API_KEY environment variable.');
} else {
  console.log(
    `env var \"ANTHROPIC_API_KEY\" is: ${
      process.env.ANTHROPIC_API_KEY.slice(0, 4) +
      '...' +
      process.env.ANTHROPIC_API_KEY.slice(-4)
    }`
  );
}

const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const model = 'claude-3-5-haiku-latest';
// const model = 'claude-haiku-4-5';
// const model = 'claude-opus-4-1';
// const model = 'claude-sonnet-4-5';

const conversationHistory = [
  {
    role: 'user',
    content:
      'Proszę, napisz mi świetny dowcip programistyczny z którego wszyscy się będą śmiali, nawet moja babcia.',
  },
  {
    role: 'assistant',
    content:
      'Dlaczego programista nie lubi chodzić do restauracji?\nBo zawsze sprawdza "if" (warunek) przed zamówieniem.',
  },
  {
    role: 'user',
    content:
      'Ten żart był głupi i nudny. Poproszę o lepszy żart, który będzie faktycznie zabawny.',
  },
];

const MAX_TOKENS = 256;
const tokenizer = getTokenizer(model);
// Tokenizuj każdą wiadomość i wyświetl tokeny
// conversationHistory.forEach((msg) => {
//   const tokens = tokenizer.encode(msg.content);
//   console.log(`${msg.role}: ${msg.content}`);
//   console.log(`> tokenized: ${tokens.join(',')}\n`);
// });

const response = await client.messages.create({
  max_tokens: MAX_TOKENS,
  messages: [
    {
      role: 'user',
      content: 'Napisz jaka osba jest elon musk',
    },
  ],
  model,
  system: 'You are an evil prank-addict. You are never serious',
});

console.log('Response', response.content[0].text);

// Tokenizuj odpowiedź
// const responseText = response.content[0].text;
// const responseTokens = tokenizer.encode(responseText);
// console.log(`assistant: ${responseText}`);
// console.log(`> tokenized: ${responseTokens.join(',')}\n`);

// // Podsumowanie
// const inputTokens = conversationHistory.reduce(
//   (sum, msg) => sum + countTokens(msg.content),
//   0
// );
// const outputTokens = countTokens(responseText);
// console.log({ inputTokens, outputTokens, usage: response.usage });
