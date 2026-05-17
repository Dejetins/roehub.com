# Backtest AI Configurator Runtime Reset

Current cleanup state for the `/backtests` AI Configurator.

Status: single-shot LM Studio prompt contract retired. Tool-agent runtime is
pending implementation.

Date: 2026-05-17.

## Gate Markers

- accepted: false
- blocking_reason: tool-agent runtime is not implemented yet after retiring the
  single-shot prompt/blob contract
- next_prompt_allowed: true

This document intentionally keeps the historical `mlx-v1` path so old prompt
artifacts still resolve. It no longer describes an accepted production runtime.

Latest cleanup evidence:
`docs/architecture/backtest/benchmark_iterations/2026-05-17_lmstudio_tools_contract/cleanup_readiness.md`.

Canonical next target:
`docs/architecture/backtest/backtest-ai-configurator-tool-agent-v1.md`.

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

## Current Code Boundary

The retained foundation is:

- AI config job storage, quota, idempotency, status and SSE shell;
- deterministic input gate and output gate;
- catalog resolver as backend-owned source inventory;
- `BacktestAiConfigValidator` and `BacktestPreflightService` as final gates;
- indicator executable support, indicator window bounds, and artifact coverage
  checks.

The active runtime placeholder is `runtime: lm_studio_tools`. It is disabled by
default and wired to a pending tool-agent gateway until the new adapter exists.

## Target Direction

The next implementation must use LM Studio OpenAI-compatible tools:

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

The tool-agent design is documented in
`docs/architecture/backtest/backtest-ai-configurator-tool-agent-v1.md`. That
document is the implementation target for Prompts 03-07 and keeps this reset
state at `accepted: false` until implementation and Mac Studio acceptance pass.

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

Until the tool-agent runtime is implemented and accepted:

1. Keep `backtest_ai_configurator.enabled: false` in production config.
2. Keep the worker drained or not deployed for public use.
3. Do not use historical single-shot benchmark evidence for rollout.
4. Preserve validator/catalog/security code as the final safety gate for the
   future tool-agent pipeline.
