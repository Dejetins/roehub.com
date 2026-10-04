import type {ClientBootstrap} from '@roehub/web-contracts';
/** Candidate-only presentation routes; the accepted platform bootstrap stays unchanged. */
export type NavigatorBootstrap = Omit<ClientBootstrap, 'client_routes'> & {client_routes?: string[]};
