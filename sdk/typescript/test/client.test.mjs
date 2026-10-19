import test from 'node:test';
import assert from 'node:assert';
import { SentinelClient, AuthenticationError, RateLimitError, PayloadTooLargeError } from '../dist/index.js';

test('SentinelClient initialization', () => {
  const client = new SentinelClient({
    baseUrl: 'https://api.sentinel.yourdomain.com',
    apiKey: 'sk-live-test-123',
  });

  assert.strictEqual(typeof client.chat, 'function');
  assert.strictEqual(typeof client.chatStream, 'function');
  assert.strictEqual(typeof client.ingest, 'function');
  assert.strictEqual(typeof client.getJobStatus, 'function');
  assert.strictEqual(typeof client.submitFeedback, 'function');
});

test('SentinelClient custom fetch mocking', async () => {
  const mockFetch = async (url, options) => {
    assert.strictEqual(url, 'https://api.sentinel.yourdomain.com/v1/chat');
    assert.strictEqual(options.method, 'POST');
    assert.strictEqual(options.headers['Authorization'], 'Bearer sk-live-test-123');

    return new Response(JSON.stringify({
      answer: 'Mocked answer',
      sources: [{ source: 'doc.md', content: 'snippet', score: 0.9 }],
      meta: {
        latency_ms: 50,
        tokens_in: 10,
        tokens_out: 5,
        provider: 'gemini',
        model: 'gemini-2.5-flash',
        cache_hit: false,
        groundedness_passed: true,
        guardrail_passed: true,
        request_id: 'req_123',
      }
    }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  };

  const client = new SentinelClient({
    baseUrl: 'https://api.sentinel.yourdomain.com',
    apiKey: 'sk-live-test-123',
    fetch: mockFetch,
  });

  const res = await client.chat({ question: 'What is a deployment?' });
  assert.strictEqual(res.answer, 'Mocked answer');
  assert.strictEqual(res.meta.groundedness_passed, true);
  assert.strictEqual(res.sources[0].source, 'doc.md');
});

test('SentinelClient error mappings', async () => {
  const mockAuthFail = async () => new Response(JSON.stringify({ message: 'Invalid key' }), { status: 401 });
  const authClient = new SentinelClient({ fetch: mockAuthFail });

  await assert.rejects(
    async () => await authClient.chat({ question: 'Hi' }),
    (err) => err instanceof AuthenticationError && err.statusCode === 401
  );

  const mockRateLimit = async () => new Response(JSON.stringify({ message: 'Throttled' }), {
    status: 429,
    headers: { 'Retry-After': '45' },
  });
  const rateLimitClient = new SentinelClient({ fetch: mockRateLimit });

  await assert.rejects(
    async () => await rateLimitClient.chat({ question: 'Hi' }),
    (err) => err instanceof RateLimitError && err.statusCode === 429 && err.retryAfter === 45
  );
});

test('React subpath export verification', async () => {
  const { useSentinelChat } = await import('../dist/react/index.js');
  assert.strictEqual(typeof useSentinelChat, 'function');
});

