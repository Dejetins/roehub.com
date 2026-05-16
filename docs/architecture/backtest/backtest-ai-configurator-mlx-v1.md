# Backtest AI Configurator LM Studio v1

Current production architecture for the `/backtests` AI configurator.

Status: current MVP contract after LM Studio serving, adapter, service lifecycle
and one-user pipeline readiness were accepted.

Date: 2026-05-16.

## Gate Markers

- accepted: true
- blocking_reason: null
- next_prompt_allowed: true

This document intentionally keeps the historical file path so older prompt
packs and handoffs still resolve, but the active runtime contract below is LM
Studio only for the MVP.

## Current Runtime Contract

- runtime: `lm_studio`
- adapter: `LMStudioOpenAICompatibleAdapter`
- model id: `gemma-4-e2b-it-4bit`
- serving API: `POST /v1/chat/completions` on loopback only
- prompt transport: text in `messages[].content`
- structured output: `response_format.type=json_schema`
- parser: backend reads `choices[0].message.content` and parses that content as
  JSON
- schema rule: JSON Schema type values must be strings; do not use
  `type: ["string", "null"]`
- concurrency default: `active_generations: 1`
- queue default: conservative `max_queue_size`, with public quotas enforced by
  backend admission checks

LM Studio lifecycle is outside the worker process. On Mac Studio it is managed
by:

- `infra/macos/launchd/com.roehub.lmstudio-backtest-ai-runtime.plist`
- `infra/scripts/monit/roehub-lmstudio-backtest-ai-runtime.monitrc`
- `scripts/macos/ensure_lmstudio_backtest_ai_runtime.sh`
- `scripts/macos/smoke_lmstudio_backtest_ai_runtime.sh`

The `launchd` unit is a one-shot ensure action and the Monit entry is a
`check program`. It is not a long-running process wrapper around the LM Studio
server command.

## Trust Boundary

The model never reads repository source code, database state, private paths, raw
artifact manifests, secrets, runtime config files, or system topology directly.
The backend produces the only trusted model context:

- `TRUSTED_CAPABILITIES`
- `externalized_runtime_capabilities`
- compact JSON Schema for the expected output
- trusted request interpretation derived by backend code

Assistant-controlled text remains untrusted until output gate, schema
validation and business validation pass. Browser rendering must keep assistant
text inert and must not stream raw model JSON into the page.

## Capability Source Of Truth

Backend validation remains the source of truth:

- indicator IDs and parameter windows come from backend executable signal
  support and `configs/prod/indicators.yaml`;
- period/timeframe requests are clipped to artifact publisher coverage;
- `BacktestPreflightService` remains the business validation boundary;
- unsupported or ambiguous requests resolve to clarification/correction states,
  not loadable configs.

The model maps natural language to a candidate config. It does not own business
semantics and cannot make unsupported capabilities valid.

## Externalized Operator Inputs

Operators may provide additional policy outside the repository:

- `ROEHUB_BACKTEST_AI_SYSTEM_PROMPT_PATH`
- `ROEHUB_BACKTEST_AI_SECURITY_GATES_PATH`

Repository defaults are local/dev fail-safe behavior. Production policy files
are runtime inputs and must not be treated as model-readable repo source.

## Historical And Deleted Runtime Classification

| Item | Classification | Current decision |
| --- | --- | --- |
| `mlx_lm.server` | historical failure evidence | Not accepted for the MVP runtime. Keep only in historical evidence or explicit non-current cleanup notes. |
| `MLXOpenAICompatibleAdapter` | deleted/stale adapter name | Not present in active source. Current adapter is `LMStudioOpenAICompatibleAdapter`. |
| `mlx_lm_server` | stale rejected config value | Not allowed in current config or current tests. Config loader accepts only `runtime: lm_studio`. |
| `MLX generate` | historical failure wording | May appear only in failed Iteration 08 evidence or cleanup classification. |
| `MLX repair` | historical failure wording | May appear only in failed Iteration 08 evidence or cleanup classification. |
| Old prompt files 01-09 | historical prompt artifacts | Do not edit for this cleanup. |
| Iteration 08 benchmark evidence | historical failure evidence | Preserve as non-acceptance evidence; do not use as current runtime target. |
| LM Studio launchd and Monit files | intentionally retained | Current accepted lifecycle path. |

## Iteration Evidence

Current accepted evidence:

- `docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/lmstudio_serving_gate.md`
- `docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/lmstudio_adapter_acceptance.md`
- `docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/lmstudio_service_lifecycle.md`
- `docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/security_pipeline_readiness.md`

Historical failed evidence:

- `docs/architecture/backtest/benchmark_iterations/2026-05-12_iteration_08_ai_configurator_load_security/`

## Rollback Note

Rollback is operational, not a model contract change:

1. Restore the previous repository commit.
2. Disable or drain `com.roehub.backtest-ai-configurator-worker`.
3. Stop LM Studio runtime through
   `scripts/macos/lmstudio_backtest_ai_runtime.py stop`.
4. Rerun `scripts/macos/smoke_prod.sh` after service reload.
