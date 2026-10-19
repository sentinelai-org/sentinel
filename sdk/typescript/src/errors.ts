/**
 * Error classes thrown by the Sentinel TypeScript Client.
 */

export class SentinelError extends Error {
  public statusCode?: number;
  public details?: unknown;

  constructor(message: string, statusCode?: number, details?: unknown) {
    super(message);
    this.name = 'SentinelError';
    this.statusCode = statusCode;
    this.details = details;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export class AuthenticationError extends SentinelError {
  constructor(message = 'Missing or invalid API key (HTTP 401)', details?: unknown) {
    super(message, 401, details);
    this.name = 'AuthenticationError';
  }
}

export class RateLimitError extends SentinelError {
  public retryAfter?: number;

  constructor(message = 'Rate limit quota exceeded (HTTP 429)', retryAfter?: number, details?: unknown) {
    super(message, 429, details);
    this.name = 'RateLimitError';
    this.retryAfter = retryAfter;
  }
}

export class PayloadTooLargeError extends SentinelError {
  constructor(message = 'Request payload exceeds maximum allowed size (HTTP 413)', details?: unknown) {
    super(message, 413, details);
    this.name = 'PayloadTooLargeError';
  }
}

export class ServiceUnavailableError extends SentinelError {
  constructor(message = 'Sentinel Gateway or dependencies unavailable (HTTP 503)', details?: unknown) {
    super(message, 503, details);
    this.name = 'ServiceUnavailableError';
  }
}
