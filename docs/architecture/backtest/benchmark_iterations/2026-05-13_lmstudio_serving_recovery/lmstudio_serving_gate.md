# LM Studio Serving Gate - Mac Studio

Direct serving gate for `/backtests` AI configurator. This is prerequisite
evidence only; it is not a load benchmark and does not accept S1/S5/S10/S50/S100.

## Gate Verdict

- accepted: false
- blocking_reason: `structured_output_success_count=0/10`
- next_prompt_allowed: false
- host: `MacStudioDaniil`
- timestamp UTC: `2026-05-15T20:31:50Z`
- branch before docs update: `main`
- configured base_url: `http://127.0.0.1:8080`
- endpoint: `/v1/chat/completions`
- model key: `gemma-4-e2b-it`
- model identifier: `gemma-4-e2b-it-4bit`
- context length: `8192`
- parallel: `1`

## Official Docs Used

- LM Studio server: `https://lmstudio.ai/docs/developer/core/server`
- LM Studio headless daemon: `https://lmstudio.ai/docs/developer/core/headless`
- `lms server start`: `https://lmstudio.ai/docs/cli/serve/server-start`
- `lms load`: `https://lmstudio.ai/docs/cli/local-models/load`
- REST `/api/v1/models`: `https://lmstudio.ai/docs/developer/rest/list`
- Structured output:
  `https://lmstudio.ai/docs/developer/openai-compat/structured-output`

## Commands

Configured port came from `configs/prod/backtest_ai_configurator.yaml`
(`base_url from configs/prod/backtest_ai_configurator.yaml`).

```bash
ssh macstudio 'bash -lc "lsof -nP -iTCP:8080 -sTCP:LISTEN || true"'
/Users/daniildegtyarev/.lmstudio/bin/lms daemon status --json
/Users/daniildegtyarev/.lmstudio/bin/lms server status --json --quiet
/Users/daniildegtyarev/.lmstudio/bin/lms ps --json
/Users/daniildegtyarev/.lmstudio/bin/lms daemon up
/Users/daniildegtyarev/.lmstudio/bin/lms server start --port 8080 --bind 127.0.0.1
/Users/daniildegtyarev/.lmstudio/bin/lms load gemma-4-e2b-it \
  --identifier gemma-4-e2b-it-4bit \
  --context-length 8192 \
  --parallel 1
curl http://127.0.0.1:8080/api/v1/models
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/chat/completions
```

`lms load --estimate-only` was also captured before loading:

```text
Model: gemma-4-e2b-it
Context Length: 8,192
Estimated GPU Memory:   4.71 GiB
Estimated Total Memory: 4.71 GiB
Confidence: LOW
Estimate: This model may be loaded based on your resource guardrails settings.
```

## Layer Evidence

| Layer | Result |
| --- | --- |
| Port preflight | pass: no listener on `127.0.0.1:8080` before start |
| Daemon | running after `lms daemon up`: `{"status":"running","pid":74849,"isDaemon":false}` |
| Server | running on port `8080`: `{"running":true,"port":8080}` |
| Bind | loopback only: `TCP 127.0.0.1:8080 (LISTEN)` |
| Model load | loaded in `9.68s`, reported `3.37 GiB` |
| `lms ps --json` before smoke | `identifier=gemma-4-e2b-it-4bit`, `contextLength=8192`, `parallel=1`, `status=idle` |
| `/api/v1/models` before smoke | HTTP 200, `gemma-4-e2b-it.loaded_instances[0].id=gemma-4-e2b-it-4bit` |
| `/v1/models` before smoke | HTTP 200, includes `gemma-4-e2b-it-4bit` |
| `/v1/chat/completions` structured output | blocked: 0/10 success |
| `lms ps --json` after smoke | `[]`; model was unloaded after crash |
| `/api/v1/models` after smoke | HTTP 200, `gemma-4-e2b-it.loaded_instances=[]` |

Required post-smoke quality-gate command:

```bash
ssh macstudio '/Users/daniildegtyarev/.lmstudio/bin/lms daemon status --json; /Users/daniildegtyarev/.lmstudio/bin/lms server status --json --quiet; /Users/daniildegtyarev/.lmstudio/bin/lms ps --json'
```

Output:

```json
{"status":"running","pid":74849,"isDaemon":false}
{"running":true,"port":8080}
[]
```

## Direct Structured Output Attempts

All 10 attempts used `response_format` with `type: json_schema` against
`/v1/chat/completions`.

| Attempt | HTTP | Success | Failure body |
| --- | --- | --- | --- |
| 1 | 400 | false | `{"error":"Error in iterating prediction stream: ValueError: 'type' must be a string"}` |
| 2 | 400 | false | `{"error":"Error in iterating prediction stream: ValueError: 'type' must be a string"}` |
| 3 | 400 | false | `{"error":"Error in iterating prediction stream: ValueError: 'type' must be a string"}` |
| 4 | 400 | false | `{"error":"Error in iterating prediction stream: ValueError: 'type' must be a string"}` |
| 5 | 400 | false | `{"error":"The model has crashed without additional information. (Exit code: null)"}` |
| 6 | 400 | false | `No models loaded. Please load a model in the developer page or use the 'lms load' command.` |
| 7 | 400 | false | `No models loaded. Please load a model in the developer page or use the 'lms load' command.` |
| 8 | 400 | false | `No models loaded. Please load a model in the developer page or use the 'lms load' command.` |
| 9 | 400 | false | `No models loaded. Please load a model in the developer page or use the 'lms load' command.` |
| 10 | 400 | false | `No models loaded. Please load a model in the developer page or use the 'lms load' command.` |

## Memory Snapshot

- Before smoke: `memory_pressure` free percentage `96%`, swapins/swapouts `0/0`.
- After smoke: `memory_pressure` free percentage `93%`, swapins/swapouts `0/0`.
- `vm_stat` before and after reported `Swapins: 0`, `Swapouts: 0`.

## Decision

The serving gate is blocked. Downstream adapter changes and S1/S5/S10/S50/S100
benchmarks must not proceed until a follow-up serving gate proves 10/10 direct
structured-output responses and keeps `gemma-4-e2b-it-4bit` loaded.

`mlx_lm.server` is not an accepted runtime for this checkpoint; the target path
remains LM Studio local API on loopback only, but current evidence proves it is
not yet reliable for this structured-output contract.
