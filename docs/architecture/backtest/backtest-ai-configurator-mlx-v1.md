# Backtest AI Configurator Runtime Reset

Historical cleanup state for the retired `/backtests` AI Configurator path.

Status: superseded. Single-shot LM Studio prompt contract and tool-agent runtime
are retired from the current path.

Date: 2026-05-16.

Latest cleanup evidence:
`docs/architecture/backtest/benchmark_iterations/2026-05-17_lmstudio_tools_contract/cleanup_readiness.md`.

## Gate Markers

- accepted: false
- blocking_reason: tool-agent runtime is not implemented yet after retiring the
  single-shot prompt/blob contract
- next_prompt_allowed: true

This document intentionally keeps the historical `mlx-v1` path so old prompt
artifacts still resolve. It no longer describes an accepted production runtime.

## Retired Runtime Contract

The following contract is no longer active and must not be used as the target
for new implementation, benchmark, or rollout prompts:

- `BacktestConfigLLMGateway` with `generate_config` / `repair_config`;
- `LMStudioOpenAICompatibleAdapter` that sends only `messages` plus
  `response_format`;
- prompt profiles that assemble a full trusted/untrusted prompt envelope;
- model-visible full `TRUSTED_CAPABILITIES` payloads;
- benchmark acceptance based on `choices[0].message.content` structured output
  without tools.

Historical evidence for that path remains useful only as failure/context
evidence. It is not rollout acceptance for the next implementation.

## Historical Code Boundary

The retained foundation is:

- AI config job storage, quota, idempotency, status and SSE shell;
- deterministic input gate and output gate;
- catalog resolver as backend-owned source inventory;
- `BacktestAiConfigValidator` and `BacktestPreflightService` as final gates;
- indicator executable support, indicator window bounds, and artifact coverage
  checks.

This boundary was kept only as historical context. Iteration 01 of assistant v1
removed `runtime: lm_studio_tools`, the old one-shot AI job API, and mode-button
UI from current code/config/browser paths.

## Superseded Target Direction

The following tool-agent direction was superseded by
`backtest-ai-configurator-assistant-v1.md` and must not be used as current
implementation guidance:

1. The model classifies whether the user asks for a `/backtests` configuration.
2. The model may request backend-owned tools such as context-source reads,
   indicator specs, artifact coverage, config templates, and validation.
3. The backend executes only allowlisted tools, with path/resource validation,
   redaction, size limits, and audit records.
4. The model returns a candidate config or a nearest valid alternative.
5. Backend validation remains authoritative before `ready` and before the UI
   exposes `Load configuration` / `Загрузить конфигурацию`.

The model must not receive arbitrary filesystem access. It may request tools;
the backend decides what can be read or executed.

## Required New Evidence

All previous LM Studio single-shot evidence is superseded for rollout. The new
tool-agent path requires fresh evidence:

- LM Studio tools preflight;
- one real API job reaching `ready`;
- tool security eval with unauthorized tool actions equal to zero;
- S1/S5/S10/S50/S100 load evidence after the smaller gates pass;
- docs and JSON evidence with `accepted`, `blocking_reason`, and
  `next_prompt_allowed`.

## Rollback / Safety

Until assistant v1 runtime is implemented and accepted:

1. Keep `backtest_ai_configurator.enabled: false` in production config.
2. Keep the worker drained or not deployed for public use.
3. Do not use historical single-shot benchmark evidence for rollout.
4. Preserve validator/catalog/security code as the final safety gate for the
   future tool-agent pipeline.
