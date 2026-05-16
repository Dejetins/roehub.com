# LM Studio Serving Gate - Mac Studio

Historical direct serving gate for `/backtests` AI configurator. This evidence
is retained for LM Studio serving investigation only. It is superseded by the
2026-05-16 single-shot contract retirement and must not be used as current
AI Configurator acceptance.

## Current Gate Verdict

- accepted: false
- blocking_reason: superseded by single-shot contract retirement
- next_prompt_allowed: true
- historical_accepted_at_collection_time: true
- host: `MacStudioDaniil`
- timestamp UTC: `2026-05-15T20:52:39Z`
- branch before docs update: `main`
- configured base_url: `http://127.0.0.1:8080`
- retired endpoint under test: `/v1/chat/completions`
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
| Model load | loaded successfully; latest reload took `7.47s`, reported `3.37 GiB` |
| `lms ps --json` before smoke | `identifier=gemma-4-e2b-it-4bit`, `contextLength=8192`, `parallel=1`, `status=idle` |
| `/api/v1/models` before smoke | HTTP 200, `gemma-4-e2b-it.loaded_instances[0].id=gemma-4-e2b-it-4bit` |
| `/v1/models` before smoke | HTTP 200, includes `gemma-4-e2b-it-4bit` |
| Plain `/v1/chat/completions` without `response_format` | HTTP 200, response content `lmstudio-basic-ok` |
| Simple structured output with string-only schema types | HTTP 200, parsed content JSON `{"status":"ok","message":"lmstudio-simple-ok"}` |
| Roehub-like structured output without nullable union | HTTP 200, parsed content JSON with `accepted=true` |
| Corrected 10-attempt structured output gate | pass: 10/10 success |
| `lms ps --json` after corrected smoke | model still loaded with `identifier=gemma-4-e2b-it-4bit`, `contextLength=8192`, `parallel=1`, `status=idle` |

Required post-smoke quality-gate command:

```bash
ssh macstudio '/Users/daniildegtyarev/.lmstudio/bin/lms daemon status --json; /Users/daniildegtyarev/.lmstudio/bin/lms server status --json --quiet; /Users/daniildegtyarev/.lmstudio/bin/lms ps --json'
```

Output:

```json
{"status":"running","pid":74849,"isDaemon":false}
{"running":true,"port":8080}
[{"type":"llm","modelKey":"gemma-4-e2b-it","identifier":"gemma-4-e2b-it-4bit","contextLength":8192,"parallel":1,"status":"idle"}]
```

## LM Studio API Usage Contract

Use the OpenAI-compatible chat endpoint:

```text
POST http://127.0.0.1:8080/v1/chat/completions
Content-Type: application/json
```

The HTTP request body is JSON. The natural-language instruction for the model
is text inside `messages[].content`.

For structured output, include `response_format`:

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "roehub_like_smoke",
    "strict": "true",
    "schema": {
      "type": "object",
      "properties": {
        "accepted": {"type": "boolean"},
        "blocking_reason": {"type": "string"},
        "next_prompt_allowed": {"type": "boolean"},
        "model_identifier": {"type": "string"},
        "stage": {"type": "string"},
        "attempt": {"type": "integer"}
      },
      "required": [
        "accepted",
        "blocking_reason",
        "next_prompt_allowed",
        "model_identifier",
        "stage",
        "attempt"
      ]
    }
  }
}
```

Do not use JSON Schema nullable unions such as:

```json
{"type": ["string", "null"]}
```

That shape caused LM Studio/MLX structured-output generation to fail with
`ValueError: 'type' must be a string`. For now, encode absence as an empty
string or a separate boolean/status field.

Retired response handling:

1. Parse the HTTP response body as JSON.
2. Read `choices[0].message.content`.
3. Parse that content string as JSON.
4. Validate the parsed object against the expected fields.

## Historical Corrected Direct Structured Output Attempts

Historically, all 10 corrected attempts used `response_format` with
`type: json_schema`
against `/v1/chat/completions`, with all schema `type` values represented as
strings.

| Attempt | HTTP | Success | Parsed result |
| --- | --- | --- | --- |
| 1 | 200 | true | `accepted=true`, `blocking_reason=""`, `next_prompt_allowed=true`, `attempt=1` |
| 2 | 200 | true | `accepted=true`, `blocking_reason=""`, `next_prompt_allowed=true`, `attempt=2` |
| 3 | 200 | true | `accepted=true`, `blocking_reason=""`, `next_prompt_allowed=true`, `attempt=3` |
| 4 | 200 | true | `accepted=true`, `blocking_reason=""`, `next_prompt_allowed=true`, `attempt=4` |
| 5 | 200 | true | `accepted=true`, `blocking_reason=""`, `next_prompt_allowed=true`, `attempt=5` |
| 6 | 200 | true | `accepted=true`, `blocking_reason=""`, `next_prompt_allowed=true`, `attempt=6` |
| 7 | 200 | true | `accepted=true`, `blocking_reason=""`, `next_prompt_allowed=true`, `attempt=7` |
| 8 | 200 | true | `accepted=true`, `blocking_reason=""`, `next_prompt_allowed=true`, `attempt=8` |
| 9 | 200 | true | `accepted=true`, `blocking_reason=""`, `next_prompt_allowed=true`, `attempt=9` |
| 10 | 200 | true | `accepted=true`, `blocking_reason=""`, `next_prompt_allowed=true`, `attempt=10` |

The previous failed run is retained as a useful compatibility note: the schema
used `"blocking_reason": {"type": ["string", "null"]}`, which is valid JSON
Schema but not accepted by this LM Studio structured-output path.

## Memory Snapshot

- Before smoke: `memory_pressure` free percentage `96%`, swapins/swapouts `0/0`.
- After smoke: `memory_pressure` free percentage `93%`, swapins/swapouts `0/0`.
- `vm_stat` before and after reported `Swapins: 0`, `Swapouts: 0`.

## Decision

The LM Studio local serving gate is accepted for the next adapter step:
10/10 corrected structured-output attempts returned valid JSON and
`gemma-4-e2b-it-4bit` remained loaded.

`mlx_lm.server` is not an accepted runtime for this checkpoint. The accepted
path is LM Studio local API on loopback only, with the schema restriction above.
This does not accept S1/S5/S10/S50/S100 benchmark performance yet; it only
unblocks the next Roehub adapter/runtime-readiness iteration.
