/**
 * Official Sentinel TypeScript & Node.js Client.
 */

import {
  AuthenticationError,
  PayloadTooLargeError,
  RateLimitError,
  SentinelError,
  ServiceUnavailableError,
} from './errors.js';
import type {
  ChatOptions,
  ChatResponse,
  FeedbackOptions,
  FeedbackResponse,
  IngestDocument,
  IngestJobResponse,
  IngestJobStatus,
  SentinelClientOptions,
  SessionHistoryResponse,
  StreamChunk,
} from './types.js';

export class SentinelClient {
  private baseUrl: string;
  private apiKey?: string;
  private timeoutMs: number;
  private customHeaders: Record<string, string>;
  private fetchFn: typeof fetch;

  constructor(options: SentinelClientOptions = {}) {
    this.baseUrl = (options.baseUrl || 'http://localhost:8000').replace(/\/$/, '');
    this.apiKey = options.apiKey;
    this.timeoutMs = options.timeoutMs ?? 30000;
    this.customHeaders = options.headers || {};
    this.fetchFn = options.fetch || globalThis.fetch;
  }

  private get headers(): Record<string, string> {
    const h: Record<string, string> = {
      'Content-Type': 'application/json',
      'User-Agent': '@sentinel-ai/client-ts/0.1.0',
      ...this.customHeaders,
    };
    if (this.apiKey) {
      h['Authorization'] = `Bearer ${this.apiKey}`;
    }
    return h;
  }

  private async handleErrors(response: Response): Promise<void> {
    if (response.ok) return;

    let errorData: any = null;
    let message = `Sentinel request failed with HTTP ${response.status}`;
    try {
      errorData = await response.json();
      message = errorData.message || errorData.detail || errorData.error || message;
    } catch {
      try {
        message = await response.text();
      } catch {}
    }

    if (response.status === 401) {
      throw new AuthenticationError(message, errorData);
    } else if (response.status === 413) {
      throw new PayloadTooLargeError(message, errorData);
    } else if (response.status === 429) {
      const retryHeader = response.headers.get('Retry-After');
      const retrySeconds = retryHeader ? parseInt(retryHeader, 10) : undefined;
      throw new RateLimitError(message, retrySeconds, errorData);
    } else if (response.status === 503) {
      throw new ServiceUnavailableError(message, errorData);
    } else {
      throw new SentinelError(message, response.status, errorData);
    }
  }

  /**
   * Submit a synchronous question to the Sentinel RAG Gateway.
   */
  async chat(options: ChatOptions): Promise<ChatResponse> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const res = await this.fetchFn(`${this.baseUrl}/v1/chat`, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify({
          question: options.question,
          session_id: options.sessionId,
        }),
        signal: controller.signal,
      });

      await this.handleErrors(res);
      return (await res.json()) as ChatResponse;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Stream generated answer tokens in real-time via Server-Sent Events (SSE).
   */
  async *chatStream(options: ChatOptions): AsyncGenerator<StreamChunk, void, unknown> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs);

    let res: Response;
    try {
      res = await this.fetchFn(`${this.baseUrl}/v1/chat/stream`, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify({
          question: options.question,
          session_id: options.sessionId,
        }),
        signal: controller.signal,
      });

      await this.handleErrors(res);
    } finally {
      clearTimeout(timeoutId);
    }

    if (!res.body) {
      throw new SentinelError('Response body is empty', res.status);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const payload = line.replace('data: ', '').trim();
          if (payload === '[DONE]') return;

          try {
            const parsed = JSON.parse(payload);
            if (parsed.token) {
              yield { type: 'token', token: parsed.token };
            } else if (parsed.type === 'sources') {
              yield { type: 'sources', sources: parsed.sources };
            } else if (parsed.type === 'meta') {
              yield { type: 'meta', meta: parsed.meta };
            }
          } catch {
            // Ignore non-JSON heartbeat lines
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  /**
   * Submit document batch for asynchronous background chunking and indexing via Redis Streams.
   */
  async ingest(documents: IngestDocument[]): Promise<IngestJobResponse> {
    const res = await this.fetchFn(`${this.baseUrl}/v1/ingest/jobs`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({ documents }),
    });

    await this.handleErrors(res);
    return (await res.json()) as IngestJobResponse;
  }

  /**
   * Check status and progress of an asynchronous ingestion job.
   */
  async getJobStatus(jobId: string): Promise<IngestJobStatus> {
    const res = await this.fetchFn(`${this.baseUrl}/v1/ingest/jobs/${jobId}`, {
      headers: this.headers,
    });

    await this.handleErrors(res);
    return (await res.json()) as IngestJobStatus;
  }

  /**
   * Submit user satisfaction rating, feedback comment, or ground-truth correction.
   */
  async submitFeedback(options: FeedbackOptions): Promise<FeedbackResponse> {
    const res = await this.fetchFn(`${this.baseUrl}/v1/feedback`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({
        request_id: options.requestId,
        rating: options.rating,
        comment: options.comment,
        correction: options.correction,
        metadata: options.metadata,
      }),
    });

    await this.handleErrors(res);
    return (await res.json()) as FeedbackResponse;
  }

  /**
   * Retrieve chronological message history for a conversation thread.
   */
  async getSessionHistory(sessionId: string): Promise<SessionHistoryResponse> {
    const res = await this.fetchFn(`${this.baseUrl}/v1/chat/sessions/${sessionId}`, {
      method: 'GET',
      headers: this.headers,
    });

    await this.handleErrors(res);
    return (await res.json()) as SessionHistoryResponse;
  }

  /**
   * Clear all message history for a given conversation thread.
   */
  async clearSession(sessionId: string): Promise<boolean> {
    const res = await this.fetchFn(`${this.baseUrl}/v1/chat/sessions/${sessionId}`, {
      method: 'DELETE',
      headers: this.headers,
    });

    await this.handleErrors(res);
    const data = (await res.json()) as { success?: boolean };
    return Boolean(data.success ?? true);
  }

  /**
   * Returns true if the Sentinel API gateway process is responding to liveness checks.
   */
  async isHealthy(): Promise<boolean> {
    try {
      const res = await this.fetchFn(`${this.baseUrl}/health`);
      return res.status === 200;
    } catch {
      return false;
    }
  }

  /**
   * Returns true if all downstream stateful dependencies (Qdrant & Redis) are ready.
   */
  async isReady(): Promise<boolean> {
    try {
      const res = await this.fetchFn(`${this.baseUrl}/ready`);
      return res.status === 200;
    } catch {
      return false;
    }
  }
}
