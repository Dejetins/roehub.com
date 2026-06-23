---
doc: rl-trading-agent-platform-v1-stage-07a-training-runner-smoke
stage: "07A"
status: accepted
plan: docs/architecture/ml/rl-trading-agent-platform-v1.md
ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
collected_at: "2026-06-23"
---

# Stage 07A: D3QN/PER Training Runner Smoke

Статус: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat.

Stage `07A` started after checking the ledger: Stage `04` is `accepted`, Stage `06` is `accepted`, and `current_stage=07A`.

Stage `07A` is accepted as trainer-smoke capability after focused local gates, Mac Studio optional-ML tests, and a deterministic small Mac Studio training smoke from accepted Stage `06` sessionized `binance:futures` data. It does not claim a full candidate model, model quality, production/research candidate readiness, registry write, activation, paper/testnet/live behavior, or mainnet behavior.

## Scope

In scope:

- additive D3QN architecture surface, PER replay buffer, deterministic training-loop smoke, and sanitized run-record hashing;
- small offline environment fixture derived from accepted Stage `06` sessionized `binance:futures` sessions;
- Stage `02C` action/reward/state compatibility check before smoke acceptance;
- smoke artifacts under `/opt/roehub/state/rl_trading/`;
- CPU/MPS/RSS/resource evidence for the small Mac Studio smoke.

Out of scope:

- full candidate training run;
- candidate quality evaluation, model registry, promotion, activation, calibration, paper/testnet/live/mainnet;
- user-owned custom model training;
- exchange SDKs, exchange secrets, order intents, source events or execution paths;
- browser/API/UI changes.

## File Manifest

Planned concrete file list before implementation edits:

- `src/trading/contexts/rl_trading/domain/training_runner.py`
- `src/trading/contexts/rl_trading/domain/__init__.py`
- `scripts/rl_trading/stage07a_training_runner_smoke.py`
- `apps/worker/rl_trading_trainer/__init__.py`
- `apps/worker/rl_trading_trainer/main/__init__.py`
- `apps/worker/rl_trading_trainer/main/main.py`
- `tests/unit/contexts/rl_trading/domain/test_training_runner.py`
- `tests/unit/apps/worker/test_rl_trading_trainer.py`
- `tests/perf_smoke/contexts/rl_trading/test_stage07a_training_smoke.py`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/07a-training-runner-smoke.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `docs/architecture/README.md` only if docs index regeneration is required

Final file manifest:

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `src/trading/contexts/rl_trading/domain/training_runner.py` | - | - | Additive D3QN/PER smoke primitives, deterministic environment transition fixture, run-record hashing, and optional PyTorch smoke loop. | `compatible-change` additive offline domain helper |
| `scripts/rl_trading/stage07a_training_runner_smoke.py` | - | - | Opt-in CLI that loads accepted Stage `06` sessionized artifacts and writes sanitized smoke artifacts under `/opt/roehub/state/rl_trading/`. | `compatible-change` operator helper |
| `apps/worker/rl_trading_trainer/__init__.py`; `apps/worker/rl_trading_trainer/main/__init__.py`; `apps/worker/rl_trading_trainer/main/main.py` | - | - | Minimal worker entrypoint delegating to the opt-in Stage `07A` smoke runner; no scheduler/service enablement. | `compatible-change` disabled/offline worker surface |
| `tests/unit/contexts/rl_trading/domain/test_training_runner.py` | - | - | Focused tests for Stage `02C` compatibility, trainable-source rejection, PER sampling/update, transition determinism, run-record hashing, and optional D3QN/PER update shapes. | `compatible-change` test-only |
| `tests/unit/apps/worker/test_rl_trading_trainer.py` | - | - | Worker entrypoint fail-closed test for non-training sources. | `compatible-change` test-only |
| `tests/perf_smoke/contexts/rl_trading/test_stage07a_training_smoke.py` | - | - | Optional `rl-ml` smoke test for file-backed sessionized artifact loading and D3QN/PER update shape. | `compatible-change` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/07a-training-runner-smoke.md` | - | - | Stage `07A` report. | `compatible-change` docs/report only |
| - | `src/trading/contexts/rl_trading/domain/__init__.py` | - | Export additive Stage `07A` trainer-smoke helpers. | `compatible-change` additive exports |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Record Stage `07A` acceptance, evidence, delivery state and Stage `07B` handoff. | `compatible-change` docs/ledger only |
| - | `docs/architecture/README.md` | - | Docs index regeneration after adding this report. | `compatible-change` docs index only |

Outside expected paths: none.

## Prompt Evidence

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/07a-training-runner-smoke.md` |
| Prompt sha256 | `20e457873823102a9cce85e70d408de61f3d72510bc9af3b53206dd3779fab91` |
| Ledger state before implementation | Stage `04` accepted; Stage `06` accepted; `current_stage=07A`; Stage `07A` pending |
| Required prerequisites | Stage `04` accepted; Stage `06` accepted |
| Stage `06` input manifest | `/opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1/stage06_sessionized_manifest.json` |
| Stage `06` input manifest sha256 | `61995c61228705090a9cd5d868776c14435ae421bdf35677a7f5c654af71ac08` |
| Delivery state | Direct-main delivery requested after Stage `07A` acceptance; primary implementation delivery commit `a5415627fd9ac3fd413199202bd83070ffdbe468`; final `origin/main`/host sync evidence is recorded by the delivery run |

## Implementation Summary

| Area | Result |
|---|---|
| D3QN model surface | `D3qnArchitectureConfig` records dueling/double-DQN MLP architecture and deterministic `architecture_hash`. |
| PER replay | `PrioritizedReplayBuffer` supports deterministic seed, proportional sampling, normalized importance weights, and TD-error priority updates. |
| Environment/action-reward fixture | Transition builder consumes Stage `06` session tensors, derives state windows with action history and state extras, and applies Stage `02C` reward semantics. |
| Run records | Stage `07A` run records include seed/config hash, dataset manifest hash, model architecture hash, metrics, resource usage, transition/artifact hashes, and safety flags. |
| Torch isolation | `torch` is imported dynamically only inside the optional smoke loop; default API/runtime imports do not require PyTorch. |
| Artifact policy | Smoke model state and run record are runtime artifacts only under `/opt/roehub/state/rl_trading/`; no tensors/checkpoints are committed. |

## Mac Studio Smoke Evidence

Boundary label: `target_host_non_production_sample_pre_main`.

Mac Studio source checkout was first fast-forwarded to clean `main...origin/main` at `89e9a09202ab8403790e2a2c506238e21beb243c`. The scoped Stage `07A` diff was applied temporarily for the sample run, then reversed. Final Mac Studio source checkout status was clean at `89e9a09202ab8403790e2a2c506238e21beb243c`.

Command:

```bash
uv run --extra rl-ml python scripts/rl_trading/stage07a_training_runner_smoke.py \
  --output-root /opt/roehub/state/rl_trading/training_smokes/stage07a_training_runner_smoke_v1 \
  --generated-at-utc 2026-06-23T12:00:00Z \
  --max-sessions 4 \
  --batch-size 8 \
  --update-steps 8 \
  --torch-num-threads 2 \
  --torch-num-interop-threads 1 \
  --device-policy cpu_only_deterministic
```

Sanitized result:

| Field | Value |
|---|---|
| Status | `accepted` |
| Torch | `2.12.1` through optional `rl-ml` path |
| Device policy | `cpu_only_deterministic` |
| MPS | `mps_built=true`, `mps_available=true`, not selected for deterministic smoke |
| Selected device | `cpu` |
| Config hash | `6611a92984f36da8077f44e0c44060e340fe470458d3bdcbc379d69f101e9e69` |
| Architecture hash | `32293f7c31d5fbec6d239d0d52099abf37058487de687adc05935628ec12cbe8` |
| Run record hash | `df3f1917bd061fa6c1a0be1a9652cca86a260c6931ed8f0c8870bb422b1cc90e` |
| Run record path | `/opt/roehub/state/rl_trading/training_smokes/stage07a_training_runner_smoke_v1/stage07a_training_run_record.json` |
| Run record file sha256 | `75c2c56744852db8d052419db572cca229e6e875d76d73c4f01d4b16b56490ab` |
| Smoke model state path | `/opt/roehub/state/rl_trading/training_smokes/stage07a_training_runner_smoke_v1/stage07a_smoke_model_state.pt` |
| Smoke model state sha256 | `6de19935ff5e9e11b08d8180fe59f9942688e3a417a1bfe0699632ab11ab6598` |
| Transition set sha256 | `b55a9e44d85da887fb77c31148dbe6f2c847da396050cf3620f506f111cd7c35` |
| Transition count | `40` from `4` source sessions |
| State dimension | `334` |
| Batch shapes | observations `[8,334]`, q-values `[8,4]`, targets `[8]`, TD errors `[8]`, weights `[8]` |
| Loss smoke | first `53084580.0`, final `337502.375`, count `8`; no quality/profitability claim |
| Resource usage | wall `0.00639087s`, RSS `280.609375 MiB`, CPU user `0.005967s`, CPU system `0.00045s`, threads observed `5`, torch threads `2`, interop `1` |
| Safety flags | no candidate model claim, no registry write, no exchange side effect, no paper/testnet/live/mainnet, no raw provider payloads, no secrets |

Mac Studio focused optional-ML tests while the scoped patch was temporarily applied:

```bash
uv run --extra rl-ml pytest -q \
  tests/unit/contexts/rl_trading/domain/test_training_runner.py \
  tests/unit/apps/worker/test_rl_trading_trainer.py \
  tests/perf_smoke/contexts/rl_trading/test_stage07a_training_smoke.py
```

Result: `8 passed in 1.25s`.

## Quality Gates

| Gate | Result |
|---|---|
| `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/07a-training-runner-smoke.md` | passed; `20e457873823102a9cce85e70d408de61f3d72510bc9af3b53206dd3779fab91` |
| Focused local `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_training_runner.py tests/unit/apps/worker/test_rl_trading_trainer.py` | passed; `6 passed, 1 skipped`; skip is the optional Torch test in default non-`rl-ml` env |
| Focused local ruff on Stage `07A` files | passed |
| Focused local pyright on Stage `07A` files | passed; `0 errors` |
| Mac Studio optional-ML focused tests | passed; `8 passed` |
| Prompt gate `uv run ruff check src/trading/contexts/rl_trading apps tests` | passed |
| Prompt gate `uv run pyright src/trading/contexts/rl_trading apps tests` | passed; `0 errors` |
| Prompt gate `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` | passed; `391 passed, 1 skipped, 3 warnings`; skip is optional Torch in default non-`rl-ml` env |
| `python -m tools.docs.generate_docs_index --check` | passed after docs index regeneration |
| Direct-main publish prep `uv sync --locked --all-groups` | passed; `Resolved 205 packages`, `Audited 176 packages` |
| Direct-main publish prep `uv run ruff check .` | passed |
| Direct-main publish prep `uv run pyright` | passed; `0 errors` |
| Direct-main publish prep `uv run pytest -q -ra` | passed; `1343 passed, 2 skipped, 3 warnings` |
| Direct-main publish prep `uv run python -m tools.docs.generate_docs_index --check` | passed; docs index up to date |

## Contract Impact

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No route, request, response payload, auth or browser-visible behavior changed. |
| Port contract | `none` | No existing port/protocol signature changed. |
| DTO schema | `none` | No wire DTO changed. |
| Persisted schema | `none` | No migration, database table, registry table or persisted service schema changed. |
| Config schema/defaults | `none` | No env/YAML/default changed; CLI flags are opt-in and local to the smoke runner. |
| Request hash / cache key / persistence identity | `none` | No existing runtime identity, cache key or request hash changed. |
| Offline trainer artifact/run-record contract | `compatible-change` | Adds Stage `07A` smoke-only run record schema and hashes under `/opt/roehub/state/rl_trading/`. |
| Dependency/default runtime | `compatible-change` | Reuses optional `rl-ml`; default imports do not require `torch`. |
| Service-call auth/timeout/retry/error semantics | `none` | No service, provider, exchange, ClickHouse, Redis, Postgres or HTTP call added by the smoke path. |
| External side effects / unknown state | `none` | Runtime writes are local files under `/opt/roehub/state/rl_trading/`; no exchange/account/order/provider side effect. |
| Logs/metrics/traces/audit/ledger/report/redaction | `compatible-change` | Adds sanitized stage report/ledger/runtime hashes and counts; no secrets, raw provider payloads or raw tensors in docs. |
| Alerts/runbook semantics | `none` | No Monit, launchd, alert, scheduler or runbook behavior changed. |
| Browser-visible behavior | `none` | Browser verification is N/A; prompt disabled browser runtime verification and no UI changed. |
| Performance hot path | `none` | Offline smoke-only trainer path; no live inference/execution/API hot path changed. |

## Mac Studio Proof Boundary

| Boundary label | Status | Evidence / rule |
|---|---|---|
| `target_host_readiness_pre_main` | collected | SSH reached `MacStudioDaniil`; source checkout at `/Users/daniildegtyarev/Projects/roehub.com` was fast-forwarded to clean `main...origin/main` at `89e9a092`; accepted Stage `06` manifest hash matched. |
| `target_host_non_production_sample_pre_main` | collected and accepted for Stage `07A` | Scoped Stage `07A` diff was applied temporarily, optional `rl-ml` focused tests passed, deterministic small smoke wrote sanitized artifacts under `/opt/roehub/state/rl_trading/training_smokes/stage07a_training_runner_smoke_v1`, then the diff was reversed and checkout returned clean. |
| `read_only_existing_runtime_smoke` | N/A | No existing `/opt/roehub/app` service or browser/runtime behavior was changed. |
| `post_main_production_runtime_proof` | tracked by delivery run | Stage `07A` does not change a production service/browser surface; source/runtime host sync and smoke evidence are collected by the direct-main delivery run after commit `a5415627fd9ac3fd413199202bd83070ffdbe468`. |

## Cold Self-Review

Cold-head review: completed
Mode: cold self-review fallback
Review scope: Stage `07A` report, Stage ledger update, docs index entry, trainer-smoke entrypoints, focused tests, and Mac Studio smoke evidence for `rl-trading-agent-platform-v1`.
Review instructions: architecture-review/references/cold-head-plan-prompt-pack-review.md
Verdict: Release after fixes
Blockers fixed: added this cold-head receipt after the hook flagged the missing readiness-gate report.
Local follow-up check: completed
Residual risks: Stage `07A` is accepted only as smoke mechanics; clean-main Stage `07B` must start from the synced `main` checkout verified by the delivery run and must not reuse the Stage `07A` smoke artifact as a candidate model.

Checklist result:

- prerequisite continuity: Stage `04` and Stage `06` are accepted before Stage `07A`;
- stage ledger continuity: ledger now records `current_stage=07B`, Stage `07A` accepted, Stage `07B` pending, primary implementation delivery commit, evidence path and next-stage handoff;
- validation depth: tests-only acceptance is not used; Mac Studio `target_host_non_production_sample_pre_main` is recorded with resource evidence;
- file manifest discipline: report lists created, modified, deleted and outside-expected paths;
- Mac Studio path contract: git operations are under `/Users/daniildegtyarev/Projects/roehub.com`; runtime artifacts are under `/opt/roehub/state/rl_trading/`; `/opt/roehub/app` is not claimed;
- secret/redaction boundary: report includes hashes/counts/resource metrics only, with no secrets, raw provider payloads or raw tensors;
- prompt/stage traceability: prompt path/hash, Stage `06` manifest hash, action/reward/state contract hash and Stage `07B` handoff are recorded;
- non-goals preserved: no full candidate training, no registry write, no promotion/activation, no exchange side effect, no paper/testnet/live/mainnet behavior.

## Blockers And Residual Risks

| Item | Status | Next action |
|---|---|---|
| Stage `07A` prerequisites | No blocker | Stage `04` and Stage `06` were accepted before implementation. |
| Full candidate training | Not done by design | Stage `07B` must run the full candidate training job from accepted Stage `06` data. |
| Candidate/model quality | Not claimed | Stage `08` must evaluate the concrete Stage `07B` candidate artifact. |
| Delivery | Direct-main delivery recorded | Primary implementation delivery commit `a5415627fd9ac3fd413199202bd83070ffdbe468` is recorded in the ledger; a clean-main Stage `07B` executor must start from the synced `main` checkout verified by the delivery run. |
| Runtime artifacts | Smoke-only, host-local | Smoke model state is not a candidate and is not registered/promoted/activated. |

## Next-Stage Handoff

Stage `07B` may start only from the accepted Stage `07A` mechanics and must not reuse the smoke model state as a candidate.

Stage `07B` must:

- use accepted Stage `06` manifest `/opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1/stage06_sessionized_manifest.json` sha256 `61995c61228705090a9cd5d868776c14435ae421bdf35677a7f5c654af71ac08`;
- run the full candidate training job with bounded CPU/MPS/RSS policy and comparable resource evidence;
- write candidate artifacts under `/opt/roehub/state/rl_trading/`, not git;
- record seed, config hash, dataset manifest hash, architecture hash, metrics, resource usage and artifact hashes;
- keep Binance spot, Bybit spot and Bybit futures as `blocked_not_training_source_v1`;
- avoid registry write, promotion, activation, paper/testnet/live/mainnet until their own accepted stages.
