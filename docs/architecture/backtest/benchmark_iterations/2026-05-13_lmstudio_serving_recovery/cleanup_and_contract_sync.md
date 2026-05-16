# Backtest AI Configurator Cleanup And Contract Sync

Cleanup evidence for stale MLX runtime assumptions after LM Studio serving,
adapter, service lifecycle and security pipeline readiness were accepted.

## Gate Markers

- accepted: true
- blocking_reason: null
- next_prompt_allowed: true
- current_active_hit_count: 0

## Current / Historical / Deleted Classification

| Item | Classification | Evidence |
| --- | --- | --- |
| `runtime: lm_studio` | current | `configs/prod/backtest_ai_configurator.yaml`; config loader accepts only this runtime. |
| `LMStudioOpenAICompatibleAdapter` | current | Active worker wiring imports this adapter. |
| `POST /v1/chat/completions` | current | Adapter posts structured chat-completions requests on loopback. |
| `TRUSTED_CAPABILITIES` | current | Backend-produced sanitized capability envelope; model does not read repo files, DB state, private paths, raw artifact manifests, or runtime config directly. |
| `externalized_runtime_capabilities` | current | Prompt envelope capability source label. |
| `ROEHUB_BACKTEST_AI_SYSTEM_PROMPT_PATH` | current operator input | Optional absolute external JSON policy path; repo default is local/dev fail-safe. |
| `ROEHUB_BACKTEST_AI_SECURITY_GATES_PATH` | current operator input | Optional absolute external JSON gate path; repo default is local/dev fail-safe. |
| `mlx_lm.server` | historical failure evidence | Allowed only in Iteration 08/10 historical notes and this cleanup classification; not an active runtime target. |
| `MLXOpenAICompatibleAdapter` | deleted/stale adapter name | No tracked active source file or import remains. |
| `mlx_lm_server` | deleted/stale config literal | Current config and current tests no longer use this rejected runtime value. |
| `MLX generate` | historical failure evidence | Allowed only in failed Iteration 08 evidence and this classification. |
| `MLX repair` | historical failure evidence | Allowed only in failed Iteration 08 evidence and this classification. |
| LM Studio launchd/Monit files | intentionally retained | They are current accepted lifecycle control, not dead rejected-runtime process files. |

## Current Runtime Contract

The MVP runtime contract is LM Studio only:

- model id: `gemma-4-e2b-it-4bit`
- runtime: `lm_studio`
- adapter: `LMStudioOpenAICompatibleAdapter`
- request API: `POST /v1/chat/completions`
- response format: `response_format.type=json_schema`
- prompt text: `messages[].content`
- parsed output: `choices[0].message.content` as JSON
- JSON Schema rule: JSON Schema type values must be strings; do not use
  `type: ["string", "null"]`
- concurrency: `active_generations: 1`
- queue: conservative `max_queue_size`

## Dependency Inventory

- `configs/prod/backtest_ai_configurator.yaml`,
  `configs/dev/backtest_ai_configurator.yaml` and
  `configs/test/backtest_ai_configurator.yaml`: temporary giant queue and
  ultra-tier literals were replaced with conservative disabled-runtime limits.
- `pyproject.toml`: no MLX-serving package is declared. `httpx` is retained for
  API/Web clients, LM Studio adapter and runtime smokes. `jsonschema` is
  retained for adapter schema validation and business validation.
- `uv.lock`: no dependency removal is required; retained `httpx` and
  `jsonschema` are still imported by current source and tests.
- imports: no current production import of `MLXOpenAICompatibleAdapter` or
  `mlx_openai_compatible` remains.

## Process And Ops Files

- Deleted rejected-runtime process files: none found in current `infra`.
- Retained current process files:
  - `infra/macos/launchd/com.roehub.lmstudio-backtest-ai-runtime.plist`
  - `infra/scripts/monit/roehub-lmstudio-backtest-ai-runtime.monitrc`
  - `infra/macos/launchd/com.roehub.backtest-ai-configurator-worker.plist`
  - `infra/scripts/monit/roehub-backtest-ai-configurator.monitrc`

## Contract Impact

- public API contract: none
- port contract: none
- DTO schema: none
- persisted schema: none
- config schema: none
- config defaults: compatible-change; temporary giant queue/concurrency literals
  were replaced with conservative disabled-runtime defaults
- request hash, cache key, persistence identity semantics: none
- browser-visible behavior: none

## Rollback

1. Restore the previous commit.
2. Disable/drain `com.roehub.backtest-ai-configurator-worker`.
3. Stop LM Studio runtime through
   `scripts/macos/lmstudio_backtest_ai_runtime.py stop`.
4. Run `scripts/macos/smoke_prod.sh` after service reload.
