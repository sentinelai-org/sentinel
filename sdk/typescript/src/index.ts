/**
 * @sentinel-ai/client
 * Official TypeScript client for Sentinel Enterprise AI & RAG Gateway.
 */

export { SentinelClient } from './client.js';
export {
  SentinelError,
  AuthenticationError,
  RateLimitError,
  PayloadTooLargeError,
  ServiceUnavailableError,
} from './errors.js';
export type {
  SentinelClientOptions,
  ChatOptions,
  ChatResponse,
  SourceItem,
  ChatMetadata,
  StreamChunk,
  IngestDocument,
  IngestJobResponse,
  IngestJobStatus,
  FeedbackOptions,
  FeedbackResponse,
} from './types.js';
export type { ChatMessage, UseSentinelChatOptions } from './react/useSentinelChat.js';
