---
doc: rl-trading-agent-platform-v1-stage-04-hf-reproducibility
stage: "04"
status: accepted
plan: docs/architecture/ml/rl-trading-agent-platform-v1.md
ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
collected_at: "2026-06-19"
---

# Stage 04: External Repo / HF Reproducibility

Статус: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat.

Stage `04` proves that the external Binance Futures HF baseline can be reproduced
inside the Roehub ML environment as a research-only smoke: with attribution,
dataset/hash manifest, deterministic run config hash, small train/eval/backtest
execution, and no exchange side effects.

## Scope

Included:

- verify prerequisite Stage `03`;
- record prompt path/hash and concrete file list before implementation edits;
- add a bounded HF manifest/reproducibility domain helper and operator script;
- run a deterministic train/eval/backtest smoke on HF Binance Futures baseline data
  under `/opt/roehub/state/rl_trading/`;
- record hashes, counts, metrics, limitations, contract impact and next-stage handoff.

Not included:

- training user-owned custom models;
- saving checkpoint tensors, raw NPZ payloads, provider payloads or secrets in git;
- replacing Roehub-native dataset stages `04A`-`08`;
- paper/testnet/live/mainnet execution, exchange SDK calls, secret custody or
  `ml_agent_decision` runtime production.

## Методология анализа

| Поле | Значение |
|---|---|
| Уровень глубины | `стандартный анализ`, reproducibility/data-quality smoke без production model approval. |
| Тип задачи | External HF baseline manifest audit + deterministic ML smoke. |
| Выбранная методология | Source manifest hashing + fixed-seed sample selection + simple baseline comparison + train/eval/backtest split reporting. |
| Простое объяснение метода | Проверяем, что HF NPZ файлы совпадают с зафиксированными hash/counts, затем на маленьком deterministic sample обучаем простую модель и сравниваем ее с baseline на eval/backtest smoke. |
| Бизнес-язык | Это проверка, что внешний подход технически воспроизводим в Roehub ML окружении; она не доказывает прибыльность или готовность к production. |
| Единица анализа | HF session array `fetcher_N` with shape `(150, 7)` and `_keys_map_` metadata. |
| Основные метрики | NPZ sha256, session counts, unique symbols, sample keys, config hash, accuracy, action counts, reward/PnL proxy. |
| Прокси-метрики | Directional close-price movement label and realized reward smoke; not a production trading scorecard. |
| Статистический/ML подход | Deterministic Torch logistic classifier on tiny feature summaries; majority-label baseline comparison. |
| Период анализа | HF split periods from the inspected NPZ metadata. |
| Группы сравнения | Train smoke sample, eval/validation smoke sample, backtest smoke sample, majority-label baseline. |
| Риски интерпретации | Tiny smoke can prove reproducibility plumbing only; it cannot approve strategy quality, futures metadata, funding/slippage/liquidation assumptions, or live execution. |
| Проверки перед выводом | Hash/count manifest checks, deterministic config hash, focused unit tests, Mac Studio script runtime smoke, docs index check. |
| Вопросы, которые нужно подтвердить до расчетов | Нет; Stage `03` accepted and dataset/runtime source was available on Mac Studio under `/opt/roehub/state/rl_trading/`. |

## File Manifest

Planned concrete file list before implementation edits:

- `src/trading/contexts/rl_trading/domain/hf_reproducibility.py`
- `scripts/rl_trading/reproduce_hf_baseline_smoke.py`
- `tests/unit/contexts/rl_trading/domain/test_hf_reproducibility.py`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04-hf-reproducibility.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `docs/architecture/README.md` only if docs index regeneration is required

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `src/trading/contexts/rl_trading/domain/hf_reproducibility.py` | - | - | Deterministic HF manifest, split inspection, sampling, tiny train/eval/backtest smoke, sanitized JSON rendering. | `compatible-change` additive internal domain helper |
| `scripts/rl_trading/reproduce_hf_baseline_smoke.py` | - | - | Operator script for HF download/hash verification and sanitized smoke evidence under `/opt/roehub/state/rl_trading/`. | `compatible-change` opt-in operator script only |
| `tests/unit/contexts/rl_trading/domain/test_hf_reproducibility.py` | - | - | Deterministic unit coverage with tiny synthetic NPZ fixtures; no network or real HF data required. | `compatible-change` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04-hf-reproducibility.md` | - | - | Stage `04` report. | `compatible-change` docs/report only |
| - | `src/trading/contexts/rl_trading/domain/__init__.py` | - | Export additive HF reproducibility helpers next to existing RL domain contracts. | `compatible-change` additive Python exports only |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Record Stage `04` acceptance, evidence and Stage `04A` handoff. | `compatible-change` docs/ledger only |
| - | `docs/architecture/README.md` | - | Generated docs index after adding this report, if regenerated by the docs tool. | `compatible-change` docs index only |

Outside expected paths: none. `docs/architecture/README.md` is a prompt-listed possible secondary touch and is only generated docs index state.

## Prompt Evidence

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/04-hf-reproducibility.md` |
| Prompt sha256 | `5713de6f1f47f8a491b68ea82de287c1dab4151124db9ea9772b8308dc093083` |
| Ledger state before implementation | Stage `03` accepted; `current_stage=04`; Stage `04` pending |
| Required prerequisite | Stage `03` accepted |
| Delivery state | `local-only`; no branch, PR, main delivery, deploy, service enablement, schema migration, API, UI, exchange, paper/testnet/live/mainnet change |
| Large artifacts | HF NPZ files and smoke JSON live on Mac Studio under `/opt/roehub/state/rl_trading/hf_reproducibility/`; no raw arrays/checkpoints committed to git. |

## Attribution And Dataset Manifest

Sources recorded:

- HF dataset: `ResearchRL/open-rl-trading-binance-dataset`, MIT License, `https://huggingface.co/datasets/ResearchRL/open-rl-trading-binance-dataset`.
- External repo: `YuriyKolesnikov/rl-trading-binance`, MIT License, `https://github.com/YuriyKolesnikov/rl-trading-binance`.
- External code vendoring: none. Stage `04` adapts concepts only: NPZ session format, 90/60 split, action/reward semantics, and train/eval/backtest lifecycle.

Observed NPZ format:

- files contain `fetcher_N` arrays and `_keys_map_`;
- arrays are article-compatible sessions shaped `(150, 7)`;
- actual inspected channel order is `open`, `high`, `volume_weighted_average`, `low`, `close`, `volume`, `num_trades`;
- source market is pinned as `binance:futures`;
- train split has `24,086` observed sessions, `18` fewer than the public card count `24,104`.

Runtime dataset root on Mac Studio:

`/opt/roehub/state/rl_trading/hf_reproducibility/dataset/ResearchRL/open-rl-trading-binance-dataset`

Dataset directory size: `276M`.

| Split file | sha256 | File size | Observed sessions | Unique symbols | Observed period | Hash |
|---|---|---:|---:|---:|---|---|
| `train_data.npz` | `1c5cdf179777f0a68a81da915749f50d97826282e1419a5314a67b170e9cb14d` | `210,111,925` bytes | `24,086` | `309` | `2020-01-14 14:28:00+00:00` to `2024-08-30 18:33:00+00:00` | matched |
| `val_data.npz` | `1e1e347bd4f842680f8a1781bc1e51f790f5e5865796e9ef3bd69548e20c51f4` | `12,009,749` bytes | `1,377` | `280` | `2024-09-01 06:02:00+00:00` to `2024-11-30 22:46:00+00:00` | matched |
| `test_data.npz` | `ff72d998fbf7d507b3db46e543aae324bece368a50ad043c057217ec2c744b1b` | `29,654,999` bytes | `3,400` | `362` | `2024-12-01 00:16:00+00:00` to `2025-02-28 22:53:00+00:00` | matched |
| `backtest_data.npz` | `dce732fda8fe1d33e92617d12f0defa3e202013617b91bb34df4d0b65aa023ee` | `27,787,724` bytes | `3,186` | `321` | `2025-03-01 00:15:00+00:00` to `2025-05-31 22:47:00+00:00` | matched |

Expected dataset manifest hash: `b111334e96c8fe4783725f6f5986d03a5c2e6e909ce4d4cc3cbe4e774c91d919`.

## Run Config And Smoke Evidence

Mac Studio command:

```bash
uv run --extra rl-ml python scripts/rl_trading/reproduce_hf_baseline_smoke.py \
  --download \
  --train-sample-size 32 \
  --evaluation-sample-size 16 \
  --backtest-sample-size 16 \
  --torch-epochs 16 \
  --output-json /opt/roehub/state/rl_trading/hf_reproducibility/stage04_hf_reproducibility_smoke.json
```

Repeat determinism command used the same arguments without `--download` and wrote
`stage04_hf_reproducibility_smoke_repeat.json`.

Evidence artifacts:

| Artifact | sha256 | Result |
|---|---|---|
| `/opt/roehub/state/rl_trading/hf_reproducibility/stage04_hf_reproducibility_smoke.json` | `2239791616edd3c5453aeec872379d8e48da3464533a74e975318802eb9db1e5` | first run |
| `/opt/roehub/state/rl_trading/hf_reproducibility/stage04_hf_reproducibility_smoke_repeat.json` | `2239791616edd3c5453aeec872379d8e48da3464533a74e975318802eb9db1e5` | byte-identical repeat |

Run config:

| Field | Value |
|---|---|
| Run config hash | `a6847569904620aab7012c44dc257fd6e24a751c14087e467c9630fe4efe410c` |
| Seed | `240604` |
| Trainer | `torch_logistic` |
| Device | `cpu` |
| Samples | train `32`, validation/eval `16`, backtest `16` |
| Window | pre-signal `90`, post-signal `60` |
| Fee/slippage smoke | transaction fee `0.001`, slippage `0.0`, initial balance `100.0` |
| Runtime | macOS `15.7.5` arm64, Python `3.12.13`, NumPy `2.0.2`, Torch `2.12.1`, MPS available `true` |

Smoke metrics:

| Surface | Sample | Accuracy | Baseline accuracy | PnL proxy | PnL ratio | Win rate | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Training smoke | `32` | `0.84375` | `0.65625` | `+61.9945267931` | `+0.0193732896` | `0.8125` | final loss `0.5416691303` vs initial `0.6311320066` |
| Evaluation smoke (`val_data.npz`) | `16` | `0.625` | `0.5` | `+4.7083339675` | `+0.0029427087` | `0.625` | proves eval path, not model quality |
| Backtest smoke (`backtest_data.npz`) | `16` | `0.4375` | `0.6875` | `-14.406225617` | `-0.009003891` | `0.375` | negative result recorded, no production approval |

Interpretation: the smoke proves that the HF dataset can be downloaded, hash-verified,
sampled deterministically, and passed through a small train/eval/backtest loop in the
accepted Stage `03` optional ML environment. The backtest smoke is intentionally not
treated as profitability evidence.

## Contract Impact

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No routes or response payloads changed. |
| Port contract | `none` | No existing Python port/protocol signature changed. |
| DTO schema | `none` | No wire DTO changed. |
| Persisted schema | `none` | No migration, table or storage schema changed. |
| Config schema/defaults | `none` | No env/YAML/default config changed. |
| Dependency/default runtime | `none` | No dependency file changed; `torch` remains optional through Stage `03` `rl-ml`. |
| Python internal module exports | `compatible-change` | Additive RL domain exports only. |
| Request hash / cache key / persistence identity | `none` | No request/cache/persistence identity changed. |
| Service-call auth/timeout/retry/error semantics | `none` | No service-call behavior changed. |
| External side effects / idempotency / unknown-state semantics | `compatible-change` | Adds opt-in HF HTTP download in an operator script; no exchange, paper, testnet, mainnet, provider credential or money-moving side effect. |
| Logs/metrics/traces/audit/ledger/report/redaction | `compatible-change` | Adds sanitized stage report and smoke JSON; no secrets/provider payloads/raw arrays/checkpoints in git. |
| Alert/runbook semantics | `none` | No monitoring, Monit, launchd or alert config changed. |
| Performance hot path | `none` | No API, inference, trainer or execution service is enabled. |
| Browser-visible behavior | `none` | Browser runtime verification is disabled by the prompt and no UI changed. |
| Docs/runbooks | `compatible-change` | Adds Stage `04` report and ledger entry. |

## Quality Gates

| Gate | Result |
|---|---|
| `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/04-hf-reproducibility.md` | passed; `5713de6f1f47f8a491b68ea82de287c1dab4151124db9ea9772b8308dc093083` |
| Focused `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_hf_reproducibility.py` | passed; `4 passed` |
| Focused `uv run ruff check src/trading/contexts/rl_trading/domain/hf_reproducibility.py scripts/rl_trading/reproduce_hf_baseline_smoke.py tests/unit/contexts/rl_trading/domain/test_hf_reproducibility.py` | passed |
| Focused `uv run pyright src/trading/contexts/rl_trading/domain/hf_reproducibility.py scripts/rl_trading/reproduce_hf_baseline_smoke.py tests/unit/contexts/rl_trading/domain/test_hf_reproducibility.py` | passed; `0 errors` |
| Mac Studio HF dataset download/hash/run smoke | passed; all four split hashes matched; evidence JSON written under `/opt/roehub/state/rl_trading/hf_reproducibility/` |
| Mac Studio deterministic repeat | passed; first and repeat evidence JSON are byte-identical with sha256 `2239791616edd3c5453aeec872379d8e48da3464533a74e975318802eb9db1e5` |
| `uv run ruff check src/trading/contexts/rl_trading apps tests` | passed |
| `uv run pyright src/trading/contexts/rl_trading apps tests` | passed; `0 errors` |
| `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` | passed; `345 passed, 3 warnings` |
| `python -m tools.docs.generate_docs_index --check` | passed after docs index regeneration |

## Cold Self-Review

Mode: `cold self-review fallback`.

Reason: subagent tooling is installed, but the subagent tool contract explicitly
forbids spawning unless the user asks for subagents, delegation, or parallel agent
work. No such user instruction was present, so the repository cold-head gate was
performed locally.

Final result: `Release`.

Reviewed lenses:

- prerequisite continuity: Stage `03` accepted before Stage `04`;
- prompt path/hash recorded;
- concrete file manifest recorded before implementation edits and finalized after validation;
- attribution and license captured for HF dataset and external repo;
- HF source remains pinned to `binance:futures`; no Binance spot, Bybit spot, or
  Bybit futures substitution;
- dataset format, channel order, split hashes/counts, train count mismatch and
  runtime artifact paths are recorded;
- runtime evidence is real-boundary Mac Studio evidence, not tests-only acceptance;
- deterministic repeat evidence is byte-identical;
- external code was not vendored, and no raw arrays/checkpoints/provider payloads
  or secrets are committed;
- no API, DTO, persistence, browser, live_execution, exchange SDK, paper/testnet,
  live or mainnet surface was opened;
- delivery state is explicitly `local-only`;
- Stage `04A` handoff is narrow and keeps HF reproducibility separate from
  Roehub-native dataset refresh.

No blocker/High finding remains inside the Stage `04` artifact, evidence or
contract scope.

## Blockers And Residual Risks

| Item | Status | Next action |
|---|---|---|
| Stage `03` prerequisite | No blocker | Stage `03` was accepted before implementation. |
| HF data access | No blocker | Four NPZ files downloaded and hash-verified on Mac Studio. |
| Determinism | No blocker | Repeat smoke JSON is byte-identical. |
| Local `/opt/roehub` write access | Non-blocking environment note | Local MacBook cannot create `/opt/roehub`; real boundary evidence was collected on Mac Studio where the artifact root exists. |
| Model quality | Residual risk | Tiny Torch logistic smoke is plumbing evidence only; Stage `07` and `08` own real training/evaluation quality gates. |
| Futures realism | Residual risk | Funding, mark/index, point-in-time filters, leverage tiers, fees/slippage/liquidation remain later-stage gates before production-grade evaluation/activation. |

## Next-Stage Handoff

Stage `04A` may start after verifying the ledger still records Stage `04` as
accepted.

The Stage `04A` executor should know:

- HF remains a Binance Futures external reproducibility baseline, not a Roehub-native dataset;
- use observed HF counts/hashes from this report, not public-card train count `24,104`;
- actual feature order is `open`, `high`, `volume_weighted_average`, `low`, `close`, `volume`, `num_trades`;
- the full HF split files are available on Mac Studio under `/opt/roehub/state/rl_trading/hf_reproducibility/dataset/ResearchRL/open-rl-trading-binance-dataset`;
- large data remains outside git; future stage reports must continue recording sanitized summaries and hashes only;
- Stage `04A` owns current Binance Futures universe/whitelist resolution and must keep the HF Binance Futures baseline separate from Roehub-native refresh work;
- no paper/testnet/live/mainnet or exchange SDK path was opened by Stage `04`.
