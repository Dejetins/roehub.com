# Stage 01 - Local Benchmark Harness

Stage `01` реализует stdlib-first локальный Python harness для проверки manifests, агрегации saved evaluator JSON, строгого pairwise keep/discard решения и генерации локальных `results.tsv`, `events.jsonl`, `summary.md`.

## Status

- Stage status: `accepted`
- Execution mode: `goal_driven`
- Previous gate: Stage `00` was `accepted`; ledger allowed `current_stage: 01`
- Source skill/plugin edits: none
- External LLM/API calls from Python: none
- Raw local fixture path: `.codex/tmp/skill-plugin-auto-improve-benchmark-v1/stage01-fixture/`

## Business And Operations Context

Business impact: Stage `01` делает benchmark воспроизводимым на машине пользователя. Улучшения skill/plugin текстов теперь можно оценивать через одинаковую схему `manifest -> saved evaluator JSON -> deterministic aggregation -> pairwise decision -> reports`, а не через одноразовую субъективную оценку.

Operational signal: Stage `01` acceptance unlocks Stage `02`, where the harness can be used for exactly ten iteration-attempt rows per selected target. Generated raw run state remains local under `.codex/tmp/` and is not a repo-delivery artifact.

| Surface | Stage `01` classification | Reason |
|---|---|---|
| Roehub runtime/service calls | `N/A` | No product API, worker, UI, DB, deploy, Mac Studio runtime, or browser route is touched. |
| Auth/secrets | `compatible-change` | Harness fails closed on secret redaction violations and does not require credentials. |
| Alerts/monitoring | `N/A` | No production alerting surface changes; stage status in the ledger remains the operational signal. |
| Runbook | `compatible-change` | CLI commands in this report are the Stage `02` local runbook. |
| External provider calls | `N/A` | Harness accepts saved evaluator JSON only; it has no live LLM provider adapter. |

## Harness API And CLI

Package created under `tools/codex_quality_benchmark/`:

| File | Responsibility |
|---|---|
| `__init__.py` | Package marker and public `BenchmarkError`. |
| `models.py` | Dataclasses for manifests, eval cases, versions, evaluator records, pairwise records, and TSV rows. |
| `manifest.py` | `manifest.json` parsing and validation: target/version hashes, rubric total, skill types, eval cases. |
| `scoring.py` | Saved evaluator JSON parsing, dimension score validation, case coverage validation, deterministic aggregation. |
| `pairwise.py` | Pairwise record parsing and strict `2-0` candidate keep rule. |
| `reports.py` | Writers for `results.tsv`, `events.jsonl`, and `summary.md`. |
| `cli.py` | `validate-manifest`, `aggregate`, and `summarize` commands. |

Minimum CLI shape implemented:

```bash
uv run python -m tools.codex_quality_benchmark.cli validate-manifest --manifest <path>
uv run python -m tools.codex_quality_benchmark.cli aggregate --run-dir <path>
uv run python -m tools.codex_quality_benchmark.cli summarize --run-dir <path>
```

Fail-closed behavior implemented:

- missing or invalid target/version `sha256`;
- rubric dimensions not summing to `100`;
- evaluator JSON missing required fields or dimension scores;
- dimension scores not matching rubric dimensions;
- version/case coverage mismatch against manifest eval cases;
- candidate keep without strict `2-0`;
- declared pairwise verdict inconsistent with computed result;
- any secret redaction violation, plus severe/critical locality or contract violations.

## Local Fixture Evidence

Fixture run path:

```text
.codex/tmp/skill-plugin-auto-improve-benchmark-v1/stage01-fixture/
```

Observed commands:

```bash
uv run python -m tools.codex_quality_benchmark.cli validate-manifest --manifest .codex/tmp/skill-plugin-auto-improve-benchmark-v1/stage01-fixture/manifest.json
uv run python -m tools.codex_quality_benchmark.cli aggregate --run-dir .codex/tmp/skill-plugin-auto-improve-benchmark-v1/stage01-fixture
uv run python -m tools.codex_quality_benchmark.cli summarize --run-dir .codex/tmp/skill-plugin-auto-improve-benchmark-v1/stage01-fixture
```

Observed outputs:

| Artifact | Evidence |
|---|---|
| `manifest.json` validation | `OK: manifest valid run_id=stage01-fixture targets=1 rubric_total=100` |
| `results.tsv` | Generated 2 rows: `v00` baseline score `82`, `v01` candidate score `92`. |
| `events.jsonl` | Generated 2 local aggregation events. |
| `summary.md` | Generated table with `workflow.staged_plan_runner` `v01` pairwise verdict `candidate`, `candidate_vs_champion=2-0`, eval cases `2/2`. |

Raw fixture files under `.codex/tmp/` are local evidence only and must not be committed.

## Thought Experiments

| Competing explanation or edge case | Stage `01` decision |
|---|---|
| "Harness should call an LLM directly to score candidates." | Rejected; Stage `01` accepts saved evaluator JSON only, so provider credentials and raw payload storage are out of scope. |
| "Pairwise record can trust declared `pairwise_verdict`." | Rejected; the harness recomputes verdict from both orderings and fails closed on declared/computed mismatch. |
| "Candidate can be kept on `1-1` if numeric score improves." | Rejected; keep requires strict pairwise `2-0`, matching the transferred `auto-improve` methodology. |
| "Fixture outputs can be durable repo docs." | Rejected; raw run state is local under `.codex/tmp/`; durable docs summarize only evidence and commands. |

## Contract Impact

| Dimension | Classification | Notes |
|---|---|---|
| Roehub public API | `none` | No API route or payload changes. |
| Port/DTO contracts | `none` | No application port or DTO changes. |
| Persisted schema | `none` | No database or persisted product state changes. |
| Config schema | `none` | No runtime config changes. |
| Runtime/deploy/browser behavior | `none` | No deployed runtime or browser-visible behavior changes. |
| Local Codex workflow | `compatible-change` | Adds optional local benchmark tooling and deterministic artifacts. |
| Benchmark/rollout gate | `compatible-change` | Defines Stage `02` command surface and fail-closed evidence checks. |

## Quality Gates

| Gate | Result |
|---|---|
| Focused tests | `passed`: `uv run pytest -q tests/unit/tools/test_codex_quality_benchmark.py` -> `5 passed in 0.03s`. |
| Focused lint | `passed`: `uv run ruff check tools/codex_quality_benchmark tests/unit/tools/test_codex_quality_benchmark.py` -> `All checks passed!`. |
| CLI fixture validate | `passed`: manifest validation command returned `OK`. |
| CLI fixture aggregate | `passed`: wrote `results.tsv` and `events.jsonl`. |
| CLI fixture summarize | `passed`: wrote `summary.md`; readback confirmed `v01` pairwise `candidate` / `2-0`. |
| Docs index | `passed`: `uv run python -m tools.docs.generate_docs_index` updated `docs/architecture/README.md`; `uv run python -m tools.docs.generate_docs_index --check` returned `OK`. |
| Markdown diff whitespace | `passed`: `git diff --check -- tools/codex_quality_benchmark tests/unit/tools/test_codex_quality_benchmark.py docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/01-local-benchmark-harness.md docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/skill-plugin-auto-improve-benchmark-v1-stage-ledger.md docs/architecture/README.md` returned no findings. |

## File Manifest

Created:

- `tools/codex_quality_benchmark/__init__.py`
- `tools/codex_quality_benchmark/models.py`
- `tools/codex_quality_benchmark/manifest.py`
- `tools/codex_quality_benchmark/scoring.py`
- `tools/codex_quality_benchmark/pairwise.py`
- `tools/codex_quality_benchmark/reports.py`
- `tools/codex_quality_benchmark/cli.py`
- `tests/unit/tools/test_codex_quality_benchmark.py`
- `docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/01-local-benchmark-harness.md`

Modified:

- `docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/skill-plugin-auto-improve-benchmark-v1-stage-ledger.md`
- `docs/architecture/README.md` if regenerated by docs index after adding this report

Deleted:

- none

Outside expected paths:

- none

Foreign changes excluded:

- Existing uncommitted prompt pack and Stage `00` artifacts were preserved.
- Source skill/plugin files under `/Users/daniildegtyarev/.codex/skills/**` and `/Users/daniildegtyarev/.codex/plugins/cache/**` were not edited.

Mixed files:

- `docs/architecture/README.md` may include index entries from Stage `00` and plan creation; Stage `01` owns only any new index entry for this report.

Local generated artifacts excluded from durable repo manifest:

- `.codex/tmp/skill-plugin-auto-improve-benchmark-v1/stage01-fixture/manifest.json`
- `.codex/tmp/skill-plugin-auto-improve-benchmark-v1/stage01-fixture/evaluations/**`
- `.codex/tmp/skill-plugin-auto-improve-benchmark-v1/stage01-fixture/pairwise/**`
- `.codex/tmp/skill-plugin-auto-improve-benchmark-v1/stage01-fixture/results.tsv`
- `.codex/tmp/skill-plugin-auto-improve-benchmark-v1/stage01-fixture/events.jsonl`
- `.codex/tmp/skill-plugin-auto-improve-benchmark-v1/stage01-fixture/summary.md`

## Stage 02 Handoff

Stage `02` may use the harness with the Stage `00` manifest data and saved evaluator JSON. The expected local run shape is:

```text
.codex/tmp/skill-plugin-auto-improve-benchmark-v1/<run_id>/
  manifest.json
  evaluations/<target_id>/<version_id>/<case_id>.json
  pairwise/<target_id>/<version_id>.json
  results.tsv
  events.jsonl
  summary.md
```

Stage `02` command sequence:

```bash
uv run python -m tools.codex_quality_benchmark.cli validate-manifest --manifest .codex/tmp/skill-plugin-auto-improve-benchmark-v1/<run_id>/manifest.json
uv run python -m tools.codex_quality_benchmark.cli aggregate --run-dir .codex/tmp/skill-plugin-auto-improve-benchmark-v1/<run_id>
uv run python -m tools.codex_quality_benchmark.cli summarize --run-dir .codex/tmp/skill-plugin-auto-improve-benchmark-v1/<run_id>
```

Stage `02` must still provide clean-context evaluator evidence separately. This harness validates and aggregates saved evidence; it does not itself create evaluator judgments or call an external provider.
