# LM Studio Adapter Acceptance - Mac Studio

Historical adapter-level structured-output gate for `/backtests` AI
Configurator.

Supersession note: on 2026-05-16 this single-shot adapter contract was retired.
Do not use this evidence as acceptance for the next LM Studio tools runtime.

This evidence proves the Roehub adapter contract only. It is not an
S1/S5/S10/S50/S100 benchmark and does not accept load performance.

## Gate Verdict

- accepted: false
- blocking_reason: superseded by single-shot contract retirement
- next_prompt_allowed: true
- host: `MacStudioDaniil`
- timestamp UTC: `2026-05-15T21:34:31.096789Z`
- runtime: `lm_studio`
- model identifier: `gemma-4-e2b-it-4bit`
- base_url: `http://127.0.0.1:8080`
- endpoint: `POST /v1/chat/completions`

## Command

The smoke ran from a temporary Mac Studio worktree containing the local adapter
diff before publishing:

```bash
ssh macstudio 'cd /tmp/roehub-lmstudio-adapter-smoke && /opt/homebrew/bin/uv run python scripts/backtest_ai/run_lmstudio_adapter_smoke.py --config configs/prod/backtest_ai_configurator.yaml --attempts 10'
```

## Adapter Contract Proved

- Worker wiring uses `LMStudioOpenAICompatibleAdapter`.
- Runtime config uses `runtime: lm_studio`.
- The adapter posts to `/v1/chat/completions`.
- Prompt policy and user content are sent as separate chat messages in
  `messages[].content`.
- The request includes `response_format.type=json_schema`.
- The schema sent to LM Studio uses string JSON Schema `type` values only.
- The adapter parses HTTP JSON, then parses `choices[0].message.content` as the
  model JSON string.
- The adapter validates the parsed object against the compact LM Studio schema
  and then against the full `backtest_ai_model_output_schema()`.
- For `needs_clarification`, the adapter normalizes LM Studio placeholder
  `config` objects to `null` before application-schema validation.
- HTTP errors carry status code and sanitized response body in internal
  diagnostics.

## Smoke Result

| Kind | Successes | Failures | Status |
| --- | ---: | ---: | --- |
| generate | 10 | 0 | all valid schema JSON |
| repair | 10 | 0 | all valid schema JSON |

All accepted calls returned `mode=create` and `status=needs_clarification`,
which is expected for this adapter smoke prompt because the period was
intentionally omitted. This smoke proves structured JSON generation and repair
through the adapter, not business readiness for a loadable config.

## Post-Smoke LM Studio State

```json
{"status":"running","pid":74849,"isDaemon":false}
{"running":true,"port":8080}
[{"identifier":"gemma-4-e2b-it-4bit","contextLength":8192,"parallel":1,"status":"idle","queued":0}]
```

## Machine-Readable Summary

See `lmstudio_adapter_acceptance.json` in this directory.
