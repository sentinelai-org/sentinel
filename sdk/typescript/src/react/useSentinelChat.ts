/**
 * React hook for streaming chat with Sentinel.
 */

import { useCallback, useMemo, useState } from 'react';
import { SentinelClient } from '../client.js';
import type { ChatMetadata, SourceItem } from '../types.js';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceItem[];
  meta?: ChatMetadata;
}

export interface UseSentinelChatOptions {
  client?: SentinelClient;
  baseUrl?: string;
  apiKey?: string;
  sessionId?: string;
}

export function useSentinelChat(options: UseSentinelChatOptions = {}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [error, setError] = useState<Error | null>(null);

  const client = useMemo(
    () =>
      options.client ||
      new SentinelClient({
        baseUrl: options.baseUrl,
        apiKey: options.apiKey,
      }),
    [options.client, options.baseUrl, options.apiKey]
  );

  const sendMessage = useCallback(
    async (question: string) => {
      if (!question.trim() || isStreaming) return;

      setError(null);
      setIsStreaming(true);

      const userMsgId = `user_${Date.now()}`;
      const assistantMsgId = `asst_${Date.now()}`;

      setMessages((prev: ChatMessage[]) => [
        ...prev,
        { id: userMsgId, role: 'user', content: question },
        { id: assistantMsgId, role: 'assistant', content: '' },
      ]);

      try {
        const stream = client.chatStream({
          question,
          sessionId: options.sessionId,
        });

        let accumulated = '';
        let sources: SourceItem[] | undefined;
        let meta: ChatMetadata | undefined;

        for await (const chunk of stream) {
          if (chunk.type === 'token') {
            accumulated += chunk.token;
            setMessages((prev: ChatMessage[]) =>
              prev.map((msg: ChatMessage) =>
                msg.id === assistantMsgId ? { ...msg, content: accumulated } : msg
              )
            );
          } else if (chunk.type === 'sources') {
            sources = chunk.sources;
            setMessages((prev: ChatMessage[]) =>
              prev.map((msg: ChatMessage) =>
                msg.id === assistantMsgId ? { ...msg, sources } : msg
              )
            );
          } else if (chunk.type === 'meta') {
            meta = chunk.meta;
            setMessages((prev: ChatMessage[]) =>
              prev.map((msg: ChatMessage) =>
                msg.id === assistantMsgId ? { ...msg, meta } : msg
              )
            );
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err : new Error(String(err)));
      } finally {
        setIsStreaming(false);
      }
    },
    [client, isStreaming, options.sessionId]
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  return {
    messages,
    sendMessage,
    isStreaming,
    error,
    clearMessages,
  };
}
