---
doc: rl-trading-agent-platform-v1-stage-07b-full-candidate-training-run
stage: "07B"
status: accepted
plan: docs/architecture/ml/rl-trading-agent-platform-v1.md
ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
collected_at: "2026-06-23"
---

# Stage 07B: Full Candidate Training Run

Статус: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat.

Stage `07B` started after checking the ledger: Stage `06` is `accepted`, Stage `07A` is `accepted`, and `current_stage=07B`. Stage `07B` is accepted after a full candidate training run completed on Mac Studio and wrote a candidate manifest under `/opt/roehub/state/rl_trading/`. Stage `08` may now start from the explicit candidate manifest path/hash recorded below.

## Scope

In scope:

- run full candidate training from the accepted Stage `06` `binance:futures` sessionized manifest;
- freeze candidate config, dataset manifest, architecture, seed, code version, device policy and resource limits before launch;
- write durable step-based progress under the run directory;
- write candidate checkpoint, report and manifest under `/opt/roehub/state/rl_trading/`;
- record hashes, resource evidence, train/validation curves and resume/failure behavior;
- update this report and the stage ledger with Stage `08` handoff state.

Out of scope:

- Stage `08` evaluation;
- model registry, promotion, activation, calibration, paper/testnet/live/mainnet;
- exchange SDKs, exchange secrets, order intents, source events or execution paths;
- browser/API/UI changes;
- user-owned custom model training or cloud/S3 hosting.

## File Manifest

Planned concrete file list before implementation edits:

- `src/trading/contexts/rl_trading/domain/training_runner.py`
- `src/trading/contexts/rl_trading/domain/__init__.py`
- `scripts/rl_trading/stage07b_full_candidate_training_run.py`
- `apps/worker/rl_trading_trainer/main/main.py`
- `tests/unit/contexts/rl_trading/domain/test_training_runner.py`
- `tests/unit/apps/worker/test_rl_trading_trainer.py`
- `tests/perf_smoke/contexts/rl_trading/test_stage07b_candidate_training.py`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/07b-full-candidate-training-run.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `docs/architecture/README.md` only if docs index regeneration is required

Final file manifest:

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `scripts/rl_trading/stage07b_full_candidate_training_run.py` | - | - | Opt-in `run`/`status` CLI for full Stage `07B` candidate training from accepted Stage `06` train/validation split artifacts. | `compatible-change` operator helper |
| `tests/perf_smoke/contexts/rl_trading/test_stage07b_candidate_training.py` | - | - | Optional `rl-ml` fixture smoke proving candidate progress, checkpoint/report/manifest writes on a tiny local Stage `06` fixture. | `compatible-change` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/07b-full-candidate-training-run.md` | - | - | Stage `07B` report. | `compatible-change` docs/report only |
| - | `src/trading/contexts/rl_trading/domain/training_runner.py` | - | Additive candidate config, full-run trainer loop, durable JSONL progress, deterministic checkpoint/resume state, candidate report/manifest helpers and resource summaries. | `compatible-change` additive offline trainer surface |
| - | `src/trading/contexts/rl_trading/domain/__init__.py` | - | Export additive Stage `07B` helper surface. | `compatible-change` additive exports |
| - | `apps/worker/rl_trading_trainer/main/main.py` | - | Route explicit `stage07b ...` invocations to the new candidate CLI while preserving the existing Stage `07A` default. | `compatible-change` disabled/offline worker surface |
| - | `tests/unit/contexts/rl_trading/domain/test_training_runner.py` | - | Focused deterministic tests for 07B config/transition hashing. | `compatible-change` test-only |
| - | `tests/unit/apps/worker/test_rl_trading_trainer.py` | - | Worker dispatch coverage for `stage07b status`. | `compatible-change` test-only |
| - | `docs/architecture/README.md` | - | Docs index regeneration after adding this report. | `compatible-change` docs index only |

Outside expected paths: none.

## Prompt Evidence

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/07b-full-candidate-training-run.md` |
| Prompt sha256 | `dd5bc8cbd1d9cdacb2666eb456292a7dcff5669ed1cee1806810e9372a7a94d5` |
| Ledger state before implementation | Stage `06` accepted; Stage `07A` accepted; `current_stage=07B`; Stage `07B` pending |
| Required prerequisites | Stage `06` accepted; Stage `07A` accepted |
| Stage `06` input manifest | `/opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1/stage06_sessionized_manifest.json` |
| Stage `06` input manifest sha256 | `61995c61228705090a9cd5d868776c14435ae421bdf35677a7f5c654af71ac08` |
| Delivery state | `local-accepted`; Mac Studio `target_host_non_production_training_pre_main` completed from a scoped dirty source sync; not delivered to `origin/main` in this chat |

## Implementation Summary

| Area | Result |
|---|---|
| Candidate config | `CandidateTrainingConfig` freezes seed, train split, validation split, planned steps, progress/checkpoint/validation cadence, replay config, architecture dimensions, device policy and Torch thread limits. |
| Dataset loading | `stage07b_full_candidate_training_run.py run` validates the accepted Stage `06` manifest hash and loads all selected train/validation split artifacts unless test-only max flags are explicitly passed. |
| Progress | The run writes compact one-event-per-line `progress.jsonl` plus `latest_status.json`. Progress is step-based: `completed_training_steps / planned_training_steps * 100`. |
| Checkpoint/resume | Checkpoints contain model, target model, optimizer, replay priorities, replay RNG, train curve and validation curve. Resume rebuilds deterministic transitions from Stage `06`, then restores latest checkpoint state. |
| Candidate artifacts | On completion the run writes final checkpoint, candidate training report and candidate manifest under `/opt/roehub/state/rl_trading/`. No artifact is committed to git. |
| Safety | No registry write, promotion, activation, exchange SDK, source event, paper/testnet/live/mainnet or browser/API/UI behavior is enabled. |

## Frozen Training Plan

| Field | Value |
|---|---|
| Run id | `stage07b_candidate_b43be9c1_61995c61_c5fbee2b` |
| Run root | `/opt/roehub/state/rl_trading/training_runs/stage07b_full_candidate_training_run_v1` |
| Run dir | `/opt/roehub/state/rl_trading/training_runs/stage07b_full_candidate_training_run_v1/stage07b_candidate_b43be9c1_61995c61_c5fbee2b` |
| Config hash | `b43be9c1ad42ac688d68a508ed908284661aca6b912774d00531362b280d538d` |
| Dataset manifest hash | `61995c61228705090a9cd5d868776c14435ae421bdf35677a7f5c654af71ac08` |
| Train source | `hf_period_rebuild_current_trading:train` from accepted Stage `06` manifest |
| Validation source | `hf_period_rebuild_current_trading:validation` from accepted Stage `06` manifest |
| Planned training steps | `100000` |
| Progress cadence | every `10000` steps or every `300` seconds, whichever comes first; first running event also emitted at step `1` |
| Checkpoint cadence | every `10000` steps and final step |
| Validation cadence | every `10000` steps, step `1`, and final step |
| Validation sample | first `4096` validation transitions per validation point |
| Batch size | `256` |
| Replay capacity | `200000` transitions; full train transition count observed as `133810` |
| Device policy | `cpu_only_deterministic`; MPS availability is recorded but not selected for this deterministic candidate run |
| Torch threads | `torch_num_threads=4`, `torch_num_interop_threads=1` |
| Resume command | `ssh macstudio 'zsh -lc "cd /Users/daniildegtyarev/Projects/roehub.com && uv run --extra rl-ml python scripts/rl_trading/stage07b_full_candidate_training_run.py run --resume --run-id stage07b_candidate_b43be9c1_61995c61_c5fbee2b --output-root /opt/roehub/state/rl_trading/training_runs/stage07b_full_candidate_training_run_v1 --sessionized-manifest /opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1/stage06_sessionized_manifest.json --expected-sessionized-manifest-sha256 61995c61228705090a9cd5d868776c14435ae421bdf35677a7f5c654af71ac08 --planned-training-steps 100000 --progress-emit-every-steps 10000 --progress-emit-every-sec 300 --checkpoint-every-steps 10000 --validation-every-steps 10000 --validation-max-transitions 4096 --batch-size 256 --replay-capacity 200000 --torch-num-threads 4 --torch-num-interop-threads 1 --device-policy cpu_only_deterministic"` |
| Status command | `ssh macstudio 'zsh -lc "cd /Users/daniildegtyarev/Projects/roehub.com && uv run --extra rl-ml python scripts/rl_trading/stage07b_full_candidate_training_run.py status --run-dir /opt/roehub/state/rl_trading/training_runs/stage07b_full_candidate_training_run_v1/stage07b_candidate_b43be9c1_61995c61_c5fbee2b"` |

## Runtime Evidence

Current state: `accepted`. The full candidate training run completed and wrote the candidate manifest/report/checkpoint under `/opt/roehub/state/rl_trading/`.

| Field | Value |
|---|---|
| Evidence label | `target_host_non_production_training_pre_main` |
| Host | `MacStudioDaniil` |
| Remote checkout path | `/Users/daniildegtyarev/Projects/roehub.com` |
| Remote checkout state | `main...origin/main` with scoped Stage `07B` source/test sync; dirty by design for pre-main non-production training |
| Script sha256 on Mac Studio | `c4184558e5ebc37c55133493828f087f0f4a25e5c558aa59868bc9cf51825f59` |
| Trainer module sha256 on Mac Studio | `c5fbee2b0ef17a315db22a66b80f1e48b5a6a5262d8b00ce49a3370aa7529bb0` |
| Stage `06` manifest proof on Mac Studio | path exists; sha256 `61995c61228705090a9cd5d868776c14435ae421bdf35677a7f5c654af71ac08` |
| Mac focused optional-ML tests | `uv run --extra rl-ml pytest -q tests/unit/contexts/rl_trading/domain/test_training_runner.py tests/unit/apps/worker/test_rl_trading_trainer.py tests/perf_smoke/contexts/rl_trading/test_stage07b_candidate_training.py` -> `10 passed` |
| Background PID | `23412` |
| PID file | `/opt/roehub/state/rl_trading/training_runs/stage07b_full_candidate_training_run_v1/stage07b_candidate_b43be9c1_61995c61_c5fbee2b/stage07b_candidate.pid` |
| Nohup log | `/opt/roehub/state/rl_trading/training_runs/stage07b_full_candidate_training_run_v1/stage07b_candidate_b43be9c1_61995c61_c5fbee2b/stage07b_candidate_nohup.log` |
| Progress path | `/opt/roehub/state/rl_trading/training_runs/stage07b_full_candidate_training_run_v1/stage07b_candidate_b43be9c1_61995c61_c5fbee2b/progress.jsonl` |
| Latest status path | `/opt/roehub/state/rl_trading/training_runs/stage07b_full_candidate_training_run_v1/stage07b_candidate_b43be9c1_61995c61_c5fbee2b/latest_status.json` |
| Latest status sha256 | `3c57c589eb62ee9a1ab9b90820d4704dd7c10e1b7252a1b71edc16083478781a` |
| Final status | `completed`, `completed_training_steps=100000`, `planned_training_steps=100000`, `progress_pct=100.0`, timestamp `2026-06-23T21:02:04Z` |
| Candidate manifest | `/opt/roehub/state/rl_trading/training_runs/stage07b_full_candidate_training_run_v1/stage07b_candidate_b43be9c1_61995c61_c5fbee2b/candidate_manifest.json` |
| Candidate manifest file sha256 | `709b4cc39d54ab1415e29c095aea6306d7ff9e0e25e0785e2605d42602a1a158` |
| Candidate manifest canonical hash | `228f9d529d72f1e473d36503488682cb6aea80884c7b9a5d7214044687d4394c` |
| Candidate report sha256 | `2961cf60c83997fc70f2b0c92b6b4168c17756b38a483c574858bd37db5d0c88` |
| Training config sha256 | `6df503a76f8465f7aefd9780ed051cea18bcc4f96e5e3c08ad20366daac54fcc` |
| Progress JSONL sha256 | `a1ea4365c7851f80b3cb05cdbde0c40d0c1989e1ac92730524d705889a6e3cb1` |
| Latest checkpoint JSON sha256 | `add6a990707309c3ca2f19609db14f9057ce32b0bc5bca9b1851b092087726da` |
| Final checkpoint sha256 | `5f11bf6901d4052a0e8f57a4eafbc773cc71506ce005c4ae1953dc7f93173d19` |
| Architecture hash | `25d70e22aaede600692209f9237e8e206ba4746f806b9a49ed254e2ec7f757dd` |
| Transition hashes | train `ad96be6ba575f9cbf2ebf1dd744979631cc1cd7048a6e49b85f7c9f34ef92c3e`, validation `8671286522f80f7ffbc1037f8bad49c68bb09d074d88f8094f51da95c411670f` |
| Transition counts | train `133810`, validation `102490` |
| Curves | train `11` points, validation `11` points |
| Throughput | `247.39589952` steps/sec |
| Wall-clock | `403.95649246s` |
| Resource summary | RSS `1506.296875 MiB`; CPU user delta `374.782482s`; CPU system delta `34.666541s`; MPS built/available `true/true`; selected device `cpu`; Torch threads `4`, interop `1` |
| Stage `08` allowed | yes |

Sanitized final progress event:

```json
{"completed_training_steps":100000,"device":"cpu","planned_training_steps":100000,"progress_pct":100.0,"run_id":"stage07b_candidate_b43be9c1_61995c61_c5fbee2b","stage":"07B","status":"completed","timestamp":"2026-06-23T21:02:04Z"}
```

## Quality Gates

| Gate | Result |
|---|---|
| `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/07b-full-candidate-training-run.md` | passed; `dd5bc8cbd1d9cdacb2666eb456292a7dcff5669ed1cee1806810e9372a7a94d5` |
| Focused local `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_training_runner.py tests/unit/apps/worker/test_rl_trading_trainer.py` | passed; `8 passed, 1 skipped`; skip is optional Torch in default non-`rl-ml` env |
| Focused local ruff on Stage `07B` files | passed |
| Focused local pyright on Stage `07B` files | passed; `0 errors` |
| Local optional-ML `uv run --extra rl-ml pytest -q tests/perf_smoke/contexts/rl_trading/test_stage07b_candidate_training.py` | passed; `1 passed` |
| Prompt gate `uv run ruff check src/trading/contexts/rl_trading apps tests` | passed |
| Prompt gate `uv run pyright src/trading/contexts/rl_trading apps tests` | passed; `0 errors` |
| Prompt gate `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` | passed; `394 passed, 3 warnings` |
| `python -m tools.docs.generate_docs_index --check` | passed |
| Mac Studio optional-ML focused tests on scoped source sync | passed; `10 passed` |

## Contract Impact

Final classification:

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No route, request, response payload, auth or browser-visible behavior changed. |
| Port contract | `none` | No existing port/protocol signature changed. |
| DTO schema | `none` | No wire DTO changed. |
| Persisted schema | `none` | No migration, database table, registry table or persisted service schema changed. |
| Config schema/defaults | `compatible-change` | Adds opt-in candidate-training CLI flags and local runtime artifact schema; no service default changes. |
| Request hash / cache key / persistence identity | `none` | No existing runtime identity, cache key or request hash changed. |
| Offline trainer artifact/run-record contract | `compatible-change` | Adds Stage `07B` candidate run manifest, checkpoint/report hashes and durable progress schema under `/opt/roehub/state/rl_trading/`. |
| External side effects / unknown state | `none` | Runtime writes are local files under `/opt/roehub/state/rl_trading/`; no exchange/account/order/provider side effect. |
| Browser-visible behavior | `none` | Browser verification is N/A; prompt disables browser runtime verification and no UI is planned. |
| Performance/resource evidence | `compatible-change` | Adds full-training wall-clock/RSS/CPU/MPS/throughput reporting for the offline candidate run. Full-run completion metrics are recorded above. |

## Mac Studio Proof Boundary

| Boundary label | Status | Evidence / rule |
|---|---|---|
| `target_host_readiness_pre_main` | collected | SSH reached `MacStudioDaniil`; remote git commands used `/Users/daniildegtyarev/Projects/roehub.com`; Stage `06` manifest path/hash exists under `/opt/roehub/state/rl_trading/`. |
| `target_host_non_production_training_pre_main` | collected and accepted | Scoped Stage `07B` source/test files were synced into the Mac Studio checkout for a non-production training run; focused optional-ML tests passed; background PID `23412` completed and wrote the candidate manifest/checkpoint/report under `/opt/roehub/state/rl_trading/`. |
| `read_only_existing_runtime_smoke` | N/A | No existing `/opt/roehub/app` service or browser/runtime behavior was checked or changed. |
| `post_main_production_runtime_proof` | N/A | Stage `07B` changes no production service/browser surface, and the current training run is pre-main non-production evidence for an offline artifact only. |

## Blockers And Residual Risks

| Item | Status | Next action |
|---|---|---|
| Stage `06` prerequisite | No blocker | Use the accepted manifest path/hash exactly. |
| Stage `07A` prerequisite | No blocker | Reuse trainer mechanics, but do not reuse the Stage `07A` smoke model as a candidate. |
| Stage `07B` candidate run | No blocker | Full candidate training completed with final progress `100%` and candidate manifest file sha256 `709b4cc39d54ab1415e29c095aea6306d7ff9e0e25e0785e2605d42602a1a158`. |
| Stage `08` handoff | No blocker | Use the candidate manifest path/hash recorded in Runtime Evidence. |

## Next-Stage Handoff

Stage `08` may start from candidate manifest `/opt/roehub/state/rl_trading/training_runs/stage07b_full_candidate_training_run_v1/stage07b_candidate_b43be9c1_61995c61_c5fbee2b/candidate_manifest.json`, file sha256 `709b4cc39d54ab1415e29c095aea6306d7ff9e0e25e0785e2605d42602a1a158`, canonical manifest hash `228f9d529d72f1e473d36503488682cb6aea80884c7b9a5d7214044687d4394c`.

## Cold Self-Review

Cold-head review: completed
Mode: cold self-review fallback
Review scope: Stage `07B` report, Stage ledger update, docs index entry, candidate-training CLI/progress/checkpoint/manifest artifact contract, and Mac Studio full-training evidence for `rl-trading-agent-platform-v1`.
Review instructions: architecture-review/references/cold-head-plan-prompt-pack-review.md
Verdict: Release
Blockers fixed: none; no Blocker or High findings were found in the cold self-review.
Local follow-up check: completed
Residual risks: Stage `07B` code and docs are local/scoped-Mac accepted in this chat and not delivered to `origin/main`; Stage `08` may consume the completed candidate manifest, but repository delivery remains a separate action if requested. Stage `08` still owns research evaluation only and must not register, promote, activate, paper trade, testnet trade, live trade or mainnet submit this candidate.
