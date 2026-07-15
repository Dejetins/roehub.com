# Roehub Plugin SDK v1alpha1

The Python and TypeScript folders publish the same versioned wire types. Plugin
containers communicate only through `roehub.plugin.rpc/v1alpha1`; they do not
import Roehub application modules. The management API is asynchronous and the
RPC surface exposes fixed capability routes rather than a generic execution
endpoint.

These scaffolds are intentionally dependency-light. The conformance source of
truth is `schemas/plugins/plugin-rpc-v1alpha1.openapi.yaml`.
