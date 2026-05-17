# Backtest AI Configurator LM Studio Tools Cleanup Readiness

This record inventories the retired single-shot `/backtests` AI Configurator
runtime layer after the reset. The `single-shot prompt/blob contract retired`
status is still active: the useful API, storage and validator foundation is
kept, while the old runtime contract is not an implementation target.

Date: 2026-05-17

## Gate Markers

- accepted: false
- blocking_reason: tool-agent runtime is not implemented yet; active runtime is
  intentionally blocked by `tool_agent_pending`
- next_prompt_allowed: true

## Cleanup Inventory

| Probe | Classification | Result |
| --- | --- | --- |
| `BacktestConfigLLMGateway` | historical doc only | No active `src`, `apps`, `scripts`, `tests`, or `configs` hit. Retained only in retirement/reset evidence. |
| `generate_config` / `repair_config` | retired single-shot semantics | No active runtime contract hit. |
| `LMStudioOpenAICompatibleAdapter` | historical doc only | No active adapter implementation hit. Historical files mark it retired. |
| `TRUSTED_CAPABILITIES` | retired model-visible blob | No active prompt/profile builder hit. Historical docs preserve why it was retired. |
| `response_format` / `choices[0].message.content` | historical single-shot smoke evidence | Current hits are in historical 2026-05-13 evidence only. |
| `.codex/agents/generated/backtest-ai-configurator-mlx-v1` | tombstone prompt pack | Prompt files are `status: superseded` and `do_not_execute: true`. |
| ignored cache/scratch search | stale bytecode or scratch | Targeted binary/hidden search found no old gateway/adapter/prompt-blob symbol hit. |

## Active Boundary

The active code boundary is now `BacktestConfigAgentGateway` with
`run_config_session`. `BacktestConfigLLMGateway`, single-shot
`generate_config` / `repair_config`, full prompt-envelope builders, and the
`LMStudioOpenAICompatibleAdapter` `messages + response_format` path are not
active code contracts.

Production runtime config accepts only `runtime: lm_studio_tools`. The worker
wires `DisabledBacktestConfigAgentGateway`, and readiness includes an explicit
`tool_agent_pending` check. That means the feature remains disabled/blocked
until the backend-owned tool-agent adapter exists.

## Preserved Foundation

Preserved surfaces:

- public API routes, status flow and SSE shell;
- job storage, quota, idempotency and lease/audit storage;
- deterministic input gate and output gate;
- catalog resolver and artifact coverage discovery;
- indicator bounds and supported catalog checks;
- `BacktestAiConfigValidator` and `BacktestPreflightService` as final backend
  validation gates.

## Contract Impact

| Dimension | Classification | Note |
| --- | --- | --- |
| Public API contract | compatible-change | API/status/SSE shell remains present; runtime jobs still fail until the agent exists. |
| Port contract | breaking-change | Runtime contract intentionally moved away from single-shot `BacktestConfigLLMGateway` semantics to backend-owned agent sessions. |
| DTO schema | none | No API DTO shape changed in this cleanup. |
| Persisted schema | none | Storage and audit tables are retained. |
| Config schema/defaults | compatible-change | `runtime: lm_studio_tools` is the only accepted runtime literal and is a blocked placeholder. |
| Request hash/cache/persistence identity | none | No identity or cache key behavior changed. |
| Rollout gate | breaking-change | Historical LM Studio single-shot acceptance is not valid rollout evidence. New tool-agent evidence is required. |

## Handoff

The next prompt may proceed to design or implement the LM Studio
OpenAI-compatible tools contract. It must not reuse the retired prompt/blob
adapter as current acceptance and must keep backend validation authoritative
before any `ready` state or load action.
