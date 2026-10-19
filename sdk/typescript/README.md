# @lansar/sentinel-client

Official TypeScript and Node.js client library for the **Sentinel Enterprise AI & RAG Gateway**.

[![npm version](https://img.shields.io/npm/v/@lansar/sentinel-client.svg)](https://www.npmjs.com/package/@lansar/sentinel-client)
[![GitHub](https://img.shields.io/badge/GitHub-sentinelai--org%2Fsentinel-181717?logo=github)](https://github.com/sentinelai-org/sentinel)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/sentinelai-org/sentinel/blob/main/LICENSE)

---

## Installation

```bash
npm install @lansar/sentinel-client
# or
pnpm add @lansar/sentinel-client
# or
yarn add @lansar/sentinel-client
```

---

## Quickstart

### 1. Synchronous Chat

```typescript
import { SentinelClient } from '@lansar/sentinel-client';

const sentinel = new SentinelClient({
  baseUrl: 'https://api.sentinel.yourdomain.com',
  apiKey: process.env.SENTINEL_API_KEY,
});

const response = await sentinel.chat({
  question: 'How do Kubernetes Deployments handle rolling updates?',
  sessionId: 'tenant_acme_user_101',
});

console.log(response.answer);
console.log(`Grounded: ${response.meta.groundedness_passed}`);
console.log(`Cost: $${response.meta.cost_usd}`);
```

### 2. Real-Time Streaming with `for await`

```typescript
import { SentinelClient } from '@lansar/sentinel-client';

const sentinel = new SentinelClient({ apiKey: process.env.SENTINEL_API_KEY });

const stream = sentinel.chatStream({
  question: 'Explain Kubernetes Pods in detail.',
});

for await (const chunk of stream) {
  if (chunk.type === 'token') {
    process.stdout.write(chunk.token);
  } else if (chunk.type === 'meta') {
    console.log(`\nLatency: ${chunk.meta.latency_ms}ms`);
  }
}
```

### 3. React Chat Hook (`useSentinelChat`)

```tsx
import React, { useState } from 'react';
import { useSentinelChat } from '@lansar/sentinel-client/react';

export function ChatWidget() {
  const [input, setInput] = useState('');
  const { messages, sendMessage, isStreaming, error } = useSentinelChat({
    baseUrl: 'https://api.sentinel.yourdomain.com',
    apiKey: 'sk-live-enterprise-key',
  });

  return (
    <div>
      <div className="messages">
        {messages.map((m) => (
          <div key={m.id} className={m.role}>
            {m.content}
          </div>
        ))}
        {isStreaming && <span>Generating answer...</span>}
      </div>

      <form onSubmit={(e) => { e.preventDefault(); sendMessage(input); setInput(''); }}>
        <input value={input} onChange={(e) => setInput(e.target.value)} />
        <button type="submit" disabled={isStreaming}>Send</button>
      </form>
    </div>
  );
}
```

---

## License

Apache-2.0
