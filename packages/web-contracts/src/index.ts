/** Transport-only contracts. An identity snapshot never grants a capability. */
export type Locale = 'ru' | 'en';
export interface SessionIdentity {
  user_id: string;
  paid_level: string;
}
export interface ClientBootstrap {
  locale: Locale;
  subject: string;
}
export type ApiFailureKind = 'unauthenticated' | 'forbidden' | 'not-found' |
  'validation' | 'conflict' | 'rate-limited' | 'unavailable' | 'transport' | 'invalid-response';
export interface FieldIssue {
  path: string;
  code: string;
  /** Render as text only, never HTML; localize by code when available. */
  message: string;
}
