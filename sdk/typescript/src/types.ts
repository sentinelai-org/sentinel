/**
 * Type definitions for Sentinel TypeScript Client.
 */

export interface SentinelClientOptions {
  /**
   * Base URL of the Sentinel Gateway API (e.g. 'https://api.sentinel.yourdomain.com').
   * Defaults to 'http://localhost:8000'.
   */
  baseUrl?: string;

  /**
   * Bearer token or API key for authenticating with Sentinel.
   */
  apiKey?: string;

  /**
   * Request timeout in milliseconds (default: 30000ms).
   */
  timeoutMs?: number;

  /**
   * Custom HTTP headers attached to every request.
   */
  headers?: Record<string, string>;

  /**
   * Optional custom fetch implementation (defaults to global fetch).
   */
  fetch?: typeof fetch;
}

export interface ChatOptions {
  /**
   * The question or prompt to ask Sentinel.
   */
  question: string;

  /**
   * Optional conversation or tenant session ID (e.g. 'tenant_acme_user_123').
   */
  sessionId?: string;
}

export interface SourceItem {
  source: string;
  content: string;
  score: number;
}

export interface ChatMetadata {
  latency_ms: number;
  tokens_in: number;
  tokens_out: number;
  provider: string;
  model: string;
  cache_hit: boolean;
  cache_type?: string | null;
  cost_usd?: number | null;
  groundedness_score?: number | null;
  groundedness_passed: boolean;
  guardrail_passed: boolean;
  request_id: string;
  trace_id?: string | null;
  span_id?: string | null;
}

export interface ChatResponse {
  answer: string;
  sources: SourceItem[];
  meta: ChatMetadata;
}

export type StreamChunk =
  | { type: 'token'; token: string }
  | { type: 'sources'; sources: SourceItem[] }
  | { type: 'meta'; meta: ChatMetadata };

export interface IngestDocument {
  content: string;
  source?: string;
  metadata?: Record<string, unknown>;
}

export interface IngestJobResponse {
  job_id: string;
  status: string;
  total_documents: number;
  created_at: string;
}

export interface IngestJobStatus {
  job_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  total_documents: number;
  processed_documents: number;
  progress_percent: number;
  created_at?: string | null;
  completed_at?: string | null;
  error?: string | null;
}

export interface FeedbackOptions {
  requestId: string;
  rating: 1 | -1;
  comment?: string;
  correction?: string;
  metadata?: Record<string, unknown>;
}

export interface FeedbackResponse {
  status: string;
  request_id: string;
  feedback_id: string;
}
