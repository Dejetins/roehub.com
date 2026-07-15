export const PLUGIN_RPC_VERSION = "roehub.plugin.rpc/v1alpha1" as const;

export type PluginCapability =
  | "app.action"
  | "data.read"
  | "notification.send"
  | "panel.describe";

export interface PluginContext {
  organizationId: string;
  instanceId: string;
  packageDigest: string;
  packageVersion: string;
  capability: PluginCapability;
}

export interface PluginResponse<T extends Record<string, unknown>> {
  contract: "PluginResponse/v1alpha1";
  status: string;
  data: T;
}

export function requireIdempotencyKey(value: string | undefined): string {
  if (value === undefined || value.length < 8 || value.length > 128) {
    throw new Error("Idempotency-Key must contain 8 to 128 characters");
  }
  return value;
}
