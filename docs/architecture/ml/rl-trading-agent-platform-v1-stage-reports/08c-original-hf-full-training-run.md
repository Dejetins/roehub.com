---
doc: rl-trading-agent-platform-v1-stage-08c-original-hf-full-training-run
stage: "08C"
status: in_progress
plan: docs/architecture/ml/rl-trading-agent-platform-v1.md
ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
collected_at: "2026-06-24"
---

# Stage 08C: Original HF Full Training Run

Status: `in_progress`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat.

Stage `08C` started after checking the ledger: Stage `04` is `accepted`,
Stage `08B` is `accepted`, `current_stage=08C`, and Stage `08C` is allowed.
Browser/auth QA is `N/A` for this offline training stage; the Roehub smoke
Keycloak username and host-local password source were not used.

This report records implementation and a managed Mac Studio full training launch.
It is not accepted yet because the full `hf_original_candidate` manifest is not
complete. Stage `08D` remains blocked until the completed full-training manifest
exists and the ledger is advanced after fresh completion evidence.

## Source Pinning

| Source | Evidence |
|---|---|
| Prompt path | `/Users/daniildegtyarev/.codex/attachments/6b64868c-bd6d-44e1-8dd7-0764aec8e830/pasted-text.txt` |
| Prompt sha256 | `a745ed404df17a6bb6441a0cf0df4ffa358604ba232a882f48836fbeefb70c39` |
| Repo prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/08c-original-hf-full-training-run.md` |
| Previous stage gate | passed: Stage `04` and Stage `08B` are `accepted`; `current_stage=08C` |
| HF dataset source | Stage `04` original HF dataset under `/opt/roehub/state/rl_trading/hf_reproducibility/dataset/ResearchRL/open-rl-trading-binance-dataset` |
| External code vendored | none |
| Raw datasets/checkpoints/provider payloads in git | none |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `src/trading/contexts/rl_trading/domain/hf_original_training.py` | - | - | Stage `08C` HF-original environment-rollout trainer, durable progress writer, best/final checkpoint materialization, resume checkpoint, report/manifest builders and validation metric loop. | `compatible-change` additive internal Python domain surface |
| `scripts/rl_trading/stage08c_original_hf_full_training_run.py` | - | - | Operator CLI for real Stage `04` HF NPZ loading, strict hash checks, status/resume support and full Mac Studio training launch. | `compatible-change` additive opt-in CLI |
| `tests/unit/contexts/rl_trading/domain/test_hf_original_training.py` | - | - | Focused unit coverage for progress, checkpoint policy, manifest and no scripted-transition path. | `none` test-only |
| `tests/perf_smoke/contexts/rl_trading/test_stage08c_original_hf_training.py` | - | - | Tiny CLI fixture smoke for the Stage `08C` operator surface. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08c-original-hf-full-training-run.md` | - | - | This in-progress Stage `08C` report. | `compatible-change` docs/report |
| - | `src/trading/contexts/rl_trading/domain/upstream_methodology.py` | - | Add PER buffer state snapshot/restore for resumable full training and ensure `learn()` returns the policy network to train mode after validation/inference. | `compatible-change` additive Python behavior for offline training |
| - | `src/trading/contexts/rl_trading/domain/__init__.py` | - | Export additive Stage `08C` trainer identifiers and helpers. | `compatible-change` additive Python export |
| - | `apps/worker/rl_trading_trainer/main/main.py` | - | Add `stage08c` dispatch to the existing trainer worker entrypoint. | `compatible-change` additive worker subcommand |
| - | `tests/unit/contexts/rl_trading/domain/test_upstream_methodology.py` | - | Cover PER state snapshot/restore. | `none` test-only |
| - | `tests/unit/apps/worker/test_rl_trading_trainer.py` | - | Cover `stage08c status` worker dispatch. | `none` test-only |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Mark Stage `08C` `in_progress` and keep `08D` blocked. | `compatible-change` docs/ledger |
| - | `docs/architecture/README.md` | - | Docs index regeneration after adding this report. | `compatible-change` docs index |

Outside expected paths: none in git.

Runtime artifacts:

| Path | Host | Reason | sha256 / state |
|---|---|---|---|
| `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/code_snapshot/` | Mac Studio | Non-production code snapshot used because the Mac Studio git checkout was already dirty/stale and missing Stage `08B` files. This avoided mutating `/Users/daniildegtyarev/Projects/roehub.com` and `/opt/roehub/app`. | directory artifact |
| `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/macstudio_smoke/stage08c_macstudio_smoke_clean/hf_original_candidate_manifest.json` | Mac Studio | Tiny real-HF target-host smoke with Stage `04` hash checks enabled; not a full candidate. | manifest hash `b500b825f72d5c434d3b97b89476acaeb464af78c09c79522a1ec19158660e98` |
| `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full/stdout.log` | Mac Studio | Managed background run stdout. | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` at evidence snapshot |
| `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full/stderr.log` | Mac Studio | Managed background run stderr. | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` at evidence snapshot |
| `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full/training_config.json` | Mac Studio | Full-run sanitized config and source/dataset lineage. | exists; sanitized JSON |
| `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full/train_only_normalization_stats.json` | Mac Studio | Train-only normalization stats from original HF train split. | exists; sanitized JSON |
| `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full/progress.jsonl` | Mac Studio | Durable episode/env-step progress. | `d3f8119b53b35168b50061acf9b1af6b4309f5d8d9a268e6401cba081b2864c2` at evidence snapshot |
| `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full/latest_status.json` | Mac Studio | Latest full-run status. | `f1403d1b8cc4e8d8dfa44687639166d26d8c1785ce4725a3366eaa658d5f251e` at evidence snapshot |
| `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full/latest_checkpoint.json` | Mac Studio | Resume checkpoint pointer. | `36d6848d524ccc70e60155e2f65a9adf1087503c3a44917867279fb0a199df46` at evidence snapshot |
| `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full/checkpoints/best.pth` | Mac Studio | Current validation-selected best checkpoint at the first validation point; not final acceptance evidence. | `684ff16df4b989d5e4eae98eb8c637546bc39e4eae610b7f3e3affb48417db34` at evidence snapshot |
| `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full/checkpoints/latest_resume.pth` | Mac Studio | Managed resumable checkpoint with model/optimizer/PER state. | `294b619f6e141e32fa001d8ce7ca6c56de077cf59c03a80aa71c5f46167d00c4` at evidence snapshot |

No completed full-run `hf_original_candidate_manifest.json` existed at the evidence
snapshot. The stage therefore remains `in_progress`.

Delivery state: `local-only` implementation plus `target_host_non_production_training_pre_main` managed run. No branch, commit, PR, production deploy, `/opt/roehub/app` sync, browser/auth proof, registry write, promotion, activation, exchange side effect, paper/testnet/live run, or mainnet submit was performed.

## Implemented Training Path

| Area | Result |
|---|---|
| Architecture | Uses `roehub_d3qn_cnn_dueling_v1` from Stage `08B`; historical MLP/scripted-transition path is not used. |
| Data | CLI loads original Stage `04` HF `train_data.npz` and `val_data.npz` with strict expected sha256 checks by default. Stage `06` Roehub-native data is not used. |
| Full profile | Defaults to `episodes=55_000`, `batch_size=16`, `learning_rate=1e-4`, `train_start=10_000`, PER capacity `230_000`, validation every `1000` episodes and validation-selected `best.pth` plus `final.pth`. |
| Progress | Writes `progress.jsonl` and `latest_status.json` with completed episodes, planned episodes, completed env steps, elapsed, ETA, device and resource snapshot. |
| Resume | Writes `latest_resume.pth` plus `latest_checkpoint.json`; resume restores policy/target/optimizer and PER buffer state. |
| Candidate manifest | On completion only, writes `hf_original_candidate_manifest.json` with train dataset hashes, config hash, code state, train-only normalization stats hash, best/final checkpoint hashes, validation curves, resource metrics and progress hash. |
| Evaluation boundary | Stage `08D` owns test/backtest acceptance. Stage `08C` does not run evaluation/backtest acceptance. |

## Target-Host Evidence

Tiny real-HF smoke:

| Field | Value |
|---|---|
| Run id | `stage08c_macstudio_smoke_clean` |
| Manifest path | `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/macstudio_smoke/stage08c_macstudio_smoke_clean/hf_original_candidate_manifest.json` |
| Manifest hash | `b500b825f72d5c434d3b97b89476acaeb464af78c09c79522a1ec19158660e98` |
| Train split hash | matched Stage `04` expected hash |
| Validation split hash | matched Stage `04` expected hash |
| Completed episodes / env steps | `2` / `20` |
| Device | `cpu` |
| Wall seconds | `0.070736` |
| Best checkpoint sha256 | `c5d1d319c262d9411c510393001f5920a58bb84832a180cc491264582f9afb5a` |
| Final checkpoint sha256 | `67aa57a35d008c82b6616ab0227c69e36d0ecd70c8a60366d77148e5e9d29191` |
| Progress sha256 | `e6bcc3b175f13d79ff1c0e4c2385178eee594211e29d842fe1ff11d4d84bc3d5` |

Full managed background run:

| Field | Value |
|---|---|
| PID | `74035` |
| Run id | `stage08c_hf_original_full` |
| Run dir | `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full` |
| Command boundary | `PYTHONPATH=/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/code_snapshot/src uv run --extra rl-ml python ... stage08c_original_hf_full_training_run.py run --output-root .../full --run-id stage08c_hf_original_full --device-policy mps_preferred_cpu_fallback --torch-num-threads 1 --torch-num-interop-threads 1` |
| Evidence snapshot timestamp | `2026-06-24T07:49:27Z` |
| Status | `running` |
| Completed episodes / planned episodes | `4601 / 55000` |
| Completed env steps / planned env steps | `46010 / 550000` |
| Progress | `8.3654545455%` |
| Device | `mps` |
| Validation points | `4` |
| Current best episode / metric | `1000` / `0.4938830067` |
| Learn updates | `36011` |
| RSS | `1799.9375 MiB` |
| CPU user/system seconds | `199.43395` / `37.267538` |
| Latest status sha256 | `f1403d1b8cc4e8d8dfa44687639166d26d8c1785ce4725a3366eaa658d5f251e` |
| Progress sha256 | `d3f8119b53b35168b50061acf9b1af6b4309f5d8d9a268e6401cba081b2864c2` |
| Resume checkpoint sha256 | `294b619f6e141e32fa001d8ce7ca6c56de077cf59c03a80aa71c5f46167d00c4` |

Status command:

```bash
ssh macstudio 'zsh -lc '"'"'cd /Users/daniildegtyarev/Projects/roehub.com && PYTHONPATH=/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/code_snapshot/src uv run --extra rl-ml python /opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/code_snapshot/scripts/rl_trading/stage08c_original_hf_full_training_run.py status --run-dir /opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full'"'"''
```

Resume command if the process stops before completion:

```bash
ssh macstudio 'zsh -lc '"'"'cd /Users/daniildegtyarev/Projects/roehub.com && PYTHONPATH=/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/code_snapshot/src uv run --extra rl-ml python /opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/code_snapshot/scripts/rl_trading/stage08c_original_hf_full_training_run.py run --resume --run-id stage08c_hf_original_full --output-root /opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full --device-policy mps_preferred_cpu_fallback --torch-num-threads 1 --torch-num-interop-threads 1'"'"''
```

## Contract Impact

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No route, request, response, auth or browser behavior changed. |
| Port contract | `none` | No application port/protocol or service boundary changed. |
| DTO schema | `none` | No wire DTO changed. |
| Persisted schema | `none` | No migration or database schema changed. |
| Config schema/defaults | `compatible-change` | Additive Python training config and additive `stage08c` worker/CLI subcommand. Existing runtime defaults remain unchanged. |
| Request hash / cache key / persistence identity | `none` | No request/cache/persistence identity changed. |
| Service-call auth/timeout/retry/error semantics | `none` | No service calls or auth surfaces changed. |
| External side effects / unknown-state semantics | `none` | No exchange, DB, Redis, registry, paper/testnet/live or mainnet side effect. |
| Logs / metrics / traces / audit / reports | `compatible-change` | Adds sanitized progress, status, manifest/report and stage ledger evidence under the ML artifact root. |
| Benchmark / rollout gates | `compatible-change` | Stage `08C` is now `in_progress`; Stage `08D` stays blocked until full candidate completion. |
| Browser-visible behavior | `none` | Browser/auth QA is `N/A`. |
| Performance hot path | `none` | Offline training only; no API or live inference hot path changed. Runtime resource evidence is training-progress evidence, not a production-latency claim. |

## Quality Gates

| Gate | Result |
|---|---|
| Previous-stage ledger gate | passed; Stage `04` and Stage `08B` are `accepted`, `current_stage=08C`, and Stage `08C` may run |
| Prompt hash | passed; `a745ed404df17a6bb6441a0cf0df4ffa358604ba232a882f48836fbeefb70c39` |
| Focused `08C` tests | passed; `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_hf_original_training.py tests/perf_smoke/contexts/rl_trading/test_stage08c_original_hf_training.py tests/unit/contexts/rl_trading/domain/test_upstream_methodology.py tests/unit/apps/worker/test_rl_trading_trainer.py` -> `16 passed` |
| Focused ruff | passed after fixes |
| Focused pyright | passed; `0 errors` |
| `uv run ruff check src/trading/contexts/rl_trading apps tests` | passed |
| `uv run pyright src/trading/contexts/rl_trading apps tests` | passed; `0 errors` |
| `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` | passed; `411 passed, 3 warnings` |
| Mac Studio tiny real-HF smoke | passed; manifest hash `b500b825f72d5c434d3b97b89476acaeb464af78c09c79522a1ec19158660e98`; train/validation split hashes matched Stage `04` |
| Mac Studio full HF run | in progress; fresh `running` event observed with PID `74035`, `1701/55000` episodes and resumable checkpoint |
| `uv run python -m tools.docs.generate_docs_index --check` | passed after docs index regeneration |

## Cold-Head Review

Cold-head review: completed.

Mode: cold self-review fallback. Independent subagent review was not used because
the available multi-agent tool contract requires an explicit user request before
spawning subagents.

Review scope: Stage `08C` implementation/report, ledger handoff, file/runtime
manifest, proof-boundary/browser-auth wording, contract impact, quality gates and
`08D` handoff.

Review instructions: `architecture-review/references/cold-head-plan-prompt-pack-review.md`.

Verdict: Block for Stage acceptance until the full HF training run completes and
writes the `hf_original_candidate` manifest.

Blockers fixed: local implementation/gates, target-host smoke, managed background
launch and fresh running event are present; the report and ledger explicitly avoid
acceptance language.

Local follow-up check: completed. Focused tests, required ruff/pyright/unit gates
and docs index check passed.

Residual risks: the background run may fail or stop before completion; the current
`best.pth` is only an intermediate validation-selected checkpoint; Stage `08D`
cannot start until a completed `hf_original_candidate` manifest exists and the
ledger advances to `08D`.

## Residual Risks

- Stage `08C` is not accepted yet. The full run has no completed candidate
  manifest at the evidence snapshot.
- The run is executing from a non-production code snapshot because the Mac Studio
  git checkout was dirty/stale. This is valid `target_host_non_production_training_pre_main`
  evidence, not post-main production runtime proof.
- `best.pth` currently exists only as an intermediate checkpoint after the first
  validation point. Completion must record final `best.pth`, `final.pth`, validation
  curves, progress hash and manifest hash before acceptance.
- No `08D` evaluation/backtest, registry write, promotion, activation, paper/testnet/live
  execution or mainnet submit was performed.

## 08D Handoff

Stage `08D` remains blocked.

The next executor must first re-check:

1. the full-run process/status under `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full`;
2. the presence and sha256 of `hf_original_candidate_manifest.json`;
3. final progress is `55000/55000` episodes and `550000/550000` env steps;
4. final manifest records Stage `04` train/validation dataset hashes, train-only normalization stats, config hash, code state, validation-selected `best.pth`, `final.pth`, validation curves, resource metrics and progress hash.

Only after that evidence exists may the ledger mark `08C` `accepted` and advance
`current_stage` to `08D`.
