# Backtest AI Configurator - LM Studio Serving Recovery And Runtime Reset

Historical recovery evidence for the `/backtests` AI configurator local serving
gate on the real Mac Studio host, plus the later reset that retired the
single-shot prompt/blob adapter contract.

## Scope

- Checked LM Studio daemon, server, loaded model state and loopback port on
  Mac Studio using `/Users/daniildegtyarev/.lmstudio/bin/lms`.
- Loaded `gemma-4-e2b-it` as `gemma-4-e2b-it-4bit` with context length `8192`
  and `parallel=1`.
- Ran staged direct checks:
  ordinary `/v1/chat/completions` without `response_format`;
  simple structured output with string-only schema `type` values;
  Roehub-like structured output without nullable union.
- Ran exactly 10 corrected direct structured-output `/v1/chat/completions`
  attempts.
- Recorded accepted evidence for LM Studio local API with schema constraint:
  do not use `"type": ["string", "null"]`.

## Gate Markers

- accepted: false
- blocking_reason: single-shot prompt/blob adapter contract retired; tool-agent
  runtime is pending implementation
- next_prompt_allowed: true

The original serving evidence remains historical. It is not rollout acceptance
for the next LM Studio tools implementation. No S1/S5/S10/S50/S100 benchmark is
accepted from this folder after the reset.

## Evidence Files

- `lmstudio_serving_gate.md`
- `lmstudio_serving_gate.json`
- `lmstudio_adapter_acceptance.md`
- `lmstudio_adapter_acceptance.json`
- `lmstudio_service_lifecycle.md`
- `lmstudio_service_lifecycle.json`
- `single_shot_contract_retirement.md`
- `single_shot_contract_retirement.json`

## Runtime Decision

`mlx_lm.server` is not accepted for this checkpoint. LM Studio local API is the
accepted serving boundary for the next Roehub adapter step when used through
`POST /v1/chat/completions` on loopback only.

The next runtime decision is not the old structured-output adapter. It must use
LM Studio OpenAI-compatible tools through a backend-owned allowlisted executor.
