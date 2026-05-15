# Backtest AI Configurator Iteration 10 - LM Studio Serving Recovery

Recovery evidence for the `/backtests` AI configurator local serving gate on
the real Mac Studio host before any adapter changes or S1/S5/S10/S50/S100
benchmark rerun.

## Scope

- Checked LM Studio daemon, server, loaded model state and loopback port on
  Mac Studio using `/Users/daniildegtyarev/.lmstudio/bin/lms`.
- Loaded `gemma-4-e2b-it` as `gemma-4-e2b-it-4bit` with context length `8192`
  and `parallel=1`.
- Ran exactly 10 direct structured-output `/v1/chat/completions` attempts.
- Recorded blocker evidence because the direct serving gate did not pass.

## Gate Markers

- accepted: false
- blocking_reason: `structured_output_success_count=0/10`
- next_prompt_allowed: false

No S1/S5/S10/S50/S100 benchmark was run in this iteration.

## Evidence Files

- `lmstudio_serving_gate.md`
- `lmstudio_serving_gate.json`

## Runtime Decision

`mlx_lm.server` is not accepted for this checkpoint. LM Studio local API is the
target serving boundary, but it is still blocked until direct structured-output
generation succeeds 10/10 without unloading or crashing the model.
