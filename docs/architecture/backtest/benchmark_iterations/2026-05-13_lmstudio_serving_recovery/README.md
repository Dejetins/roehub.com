# Backtest AI Configurator Iteration 10 - LM Studio Serving Recovery

Recovery evidence for the `/backtests` AI configurator local serving gate on
the real Mac Studio host before any adapter changes or S1/S5/S10/S50/S100
benchmark rerun.

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

- accepted: true
- blocking_reason: null
- next_prompt_allowed: true

No S1/S5/S10/S50/S100 benchmark was run in this iteration.

## Evidence Files

- `lmstudio_serving_gate.md`
- `lmstudio_serving_gate.json`

## Runtime Decision

`mlx_lm.server` is not accepted for this checkpoint. LM Studio local API is the
accepted serving boundary for the next Roehub adapter step when used through
`POST /v1/chat/completions` on loopback only.

Important API rule: the request itself is JSON, while the model prompt is text
inside `messages[].content`. Structured output is requested through
`response_format.type=json_schema`. In the JSON Schema sent to LM Studio, keep
every `type` value as a string; do not use nullable unions such as
`"type": ["string", "null"]`.
