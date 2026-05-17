# Backtest AI Configurator Tools Cleanup Readiness

Date: 2026-05-17

## Gate Markers

- accepted: false
- blocking_reason: tool_agent_pending; single-shot prompt/blob contract retired
- next_prompt_allowed: true

This cleanup is a repository-boundary readiness record only. It is not model
runtime acceptance and does not implement the tool-agent pipeline.

## Inventory

| Area | Classification | Result |
| --- | --- | --- |
| `src`, `apps`, `scripts`, `tests`, `configs` search for `BacktestConfigLLMGateway`, `LMStudioOpenAICompatibleAdapter`, `generate_config`, `repair_config`, `TRUSTED_CAPABILITIES`, `response_format` | active code check | No active single-shot gateway or adapter contract found. Retained storage audit fields remain historical schema/audit foundation. |
| `src/trading/contexts/backtest/application/ai_configurator/ports/agent_gateway.py` | active code | Current port is `BacktestConfigAgentGateway.run_config_session`; it forbids full catalog prompt blobs and arbitrary filesystem access. |
| `apps/api/wiring/modules/backtest.py` | active code | API composition keeps storage, quota, catalog resolver, validator, artifact capability discovery and uses `DisabledBacktestConfigAgentGateway` by default. |
| `apps/worker/backtest_ai_configurator/wiring/modules.py` | active code | Worker readiness includes `tool_agent_pending`, so production runtime remains blocked. |
| `.codex/agents/generated/backtest-ai-configurator-mlx-v1` | tombstone prompts | Old prompt pack is marked `do_not_execute: true`; Iterations 01-03 are now historical retained-foundation records, not executable prompts. |
| `docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery` | historical docs | Old LM Studio `messages` plus `response_format` evidence remains only as retired/historical evidence. |
| `.codex/tmp/iteration_08_ai_configurator_smoke` | ignored scratch | Search found no remaining full old single-shot prompt/data blobs in the ignored scratch files. |
| `__pycache__` under AI configurator adapter boundary | stale bytecode | Removed stale deleted-adapter bytecode for `lmstudio_tools.py`. |

## Removed Or Retired

- Active single-shot prompt/blob gateway semantics remain absent from active
  code: no `BacktestConfigLLMGateway`, no `generate_config` /
  `repair_config` port, and no `LMStudioOpenAICompatibleAdapter`.
- Old generated prompts in
  `.codex/agents/generated/backtest-ai-configurator-mlx-v1` are tombstones and
  must not be used as executable implementation instructions.
- The reset document now points to this cleanup evidence and states that
  `runtime: lm_studio_tools` is wired to the disabled gateway while
  `tool_agent_pending` blocks readiness.

## Preserved Foundation

- Public API routes, status/SSE shell, quota and idempotency remain intact.
- Postgres job storage and attempt audit fields remain intact.
- `BacktestAiCatalogResolver`, artifact coverage discovery, indicator bounds,
  `BacktestAiInputGate`, `BacktestAiOutputGate`,
  `BacktestAiConfigValidator`, and `BacktestPreflightService` remain intact.
- Final backend validation remains authoritative before any future `ready`
  state can expose `Load configuration`.

## Contract Impact

| Dimension | Classification | Note |
| --- | --- | --- |
| public API contract | none | Routes and payload shell are preserved; feature remains disabled/blocked. |
| port contract | breaking-change | Retired `BacktestConfigLLMGateway.generate_config` / `repair_config`; current port is backend-owned `BacktestConfigAgentGateway.run_config_session`. |
| DTO schema | none | No DTO shape change in this cleanup. |
| persisted schema | none | Historical audit columns are intentionally retained. |
| config schema/defaults | compatible-change | Current accepted literal is blocked `runtime: lm_studio_tools`; config still validates loopback-only model settings. |
| request hash, cache key or persistence identity | none | No idempotency or storage identity changes. |
| rollout gate | breaking-change | Old single-shot acceptance is invalid for rollout; fresh tool-agent evidence is required. |

## Runtime State

Runtime is intentionally not accepted:

- accepted: false
- blocking_reason: tool_agent_pending; single-shot prompt/blob contract retired
- next_prompt_allowed: true

The next prompt may proceed only to design or implement the LM Studio tools
pipeline. It must not reuse historical single-shot evidence as production
acceptance.
