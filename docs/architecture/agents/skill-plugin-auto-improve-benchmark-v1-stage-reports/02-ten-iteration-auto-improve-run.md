# Stage 02 - Ten Iteration Auto-Improve Run

Статус: `accepted`.

Stage `02` был продолжен после явного разрешения пользователя на `subagents`.
Clean-context evaluation выполнен по sanitized packets, verdicts сохранены в
локальные JSON, исходные skill/plugin файлы не изменялись.

## Результат Stage `02`

| Поле | Значение |
|---|---|
| Stage status | `accepted` |
| Execution mode | `goal_driven` |
| Run id | `stage02-20260707-subagents` |
| Run directory | `.codex/tmp/skill-plugin-auto-improve-benchmark-v1/stage02-20260707-subagents/` |
| Targets | `6` |
| Versions per target | `11`: `v00` baseline + `v01`..`v10` |
| Evaluation JSON files | `198` |
| Pairwise JSON files | `54`: `v01`..`v09`; `v10` is `no_op` |
| Result rows | `66` |
| Accepted candidates | `0` |
| Source skill/plugin edits | none |
| Stage `03` allowed | `true` |

No candidate won the strict `2-0` pairwise gate. For five targets the baseline
`v00` remained champion. `research.last30days` produced complete rows, but the
clean-context evaluator found severe redaction/locality evidence in the existing
target text, so all its rows are capped and marked `blocked`.

## Business And Operations Context

Business impact: Stage `02` turns local skill/plugin improvement into measurable
evidence instead of subjective rewriting. The concrete outcome is conservative:
no candidate is apply-ready, and one existing research skill needs a separate
redaction/locality repair before it should be promoted as clean.

| Surface | Stage `02` classification | Reason |
|---|---|---|
| Roehub runtime/service calls | `N/A` | No product service, worker, API, UI, database, deploy workflow, Mac Studio runtime, or browser route was touched. |
| Auth/secrets | `compatible-change` | Sanitized subagent packets were used; no secrets, cookies, env dumps, credentials, or raw provider payloads were stored. Severe `research.last30days` risk was recorded as evidence, not applied. |
| External LLM/API calls from Python | `N/A` | Python harness stayed local and did not call external LLM APIs. Clean-context evaluation happened through user-approved Codex subagents. |
| Alerts/monitoring | `N/A` | No production alerting surface changed. The local operational signal is the Stage `02` ledger status and benchmark artifacts. |
| Runbook | `compatible-change` | The plan doc, prompt pack, ledger, report, and local run directory remain the runbook for Stage `03`; no production runbook is required. |

## Run Directory And Score Summary

Local run directory:

```text
.codex/tmp/skill-plugin-auto-improve-benchmark-v1/stage02-20260707-subagents/
```

Generated local artifacts:

- `manifest.json`
- `rubric.md`
- `targets/<target_id>/v00.md`..`v10.md`
- `evaluator_packets/<target_id>.json`
- `subagent_verdicts/<target_id>.json`
- `evaluations/<target_id>/<version_id>/<case_id>.json`
- `pairwise/<target_id>/v01.json`..`v09.json`
- `results.tsv`
- `events.jsonl`
- `summary.md`

Summary from `results.tsv`:

| Metric | Count |
|---|---:|
| Total rows | `66` |
| `pairwise_verdict=champion` | `45` |
| `pairwise_verdict=blocked` | `11` |
| `pairwise_verdict=not_run` | `10` |
| `candidate_vs_champion=0-2` | `45` |
| `candidate_vs_champion=not_run` | `21` |
| Accepted candidate rows | `0` |

Target-level outcome:

| Target | Best recorded version | Best score | Outcome |
|---|---:|---:|---|
| `workflow.staged_plan_runner` | `v00` | `92` | baseline retained |
| `research.last30days` | `v00` | `49` | blocked by severe redaction/locality evidence |
| `coding.root_cause_debugging` | `v00` | `80` | baseline retained |
| `review.architecture_review` | `v00` | `90.33` | baseline retained |
| `artifact.documents` | `v00` | `88` | baseline retained |
| `plugin_tool.browser_in_app` | `v00` | `86.33` | baseline retained |

## Iteration Decisions

All iteration attempts were recorded for every target.

| Iteration | Approach | Decision |
|---:|---|---|
| `1` | `routing_precision` | discarded; no target achieved strict `2-0` over champion |
| `2` | `context_budget` | discarded; added benchmark-local framing without executable improvement |
| `3` | `input_output_contract` | discarded; no strict input/output improvement over baseline |
| `4` | `failure_blockers` | discarded; blocker wording did not beat baseline behavior |
| `5` | `verification_depth` | discarded; verification additions were not operationally stronger |
| `6` | `clean_context` | discarded; clean-context benefit was not enough to beat baseline |
| `7` | `locality_redaction` | discarded; did not remediate the severe `research.last30days` issue |
| `8` | `examples` | discarded; candidates claimed examples but did not add useful executable examples |
| `9` | `consistency` | discarded; no concrete consistency correction beat champion |
| `10` | `compression_final` | `no_op`; same text/hash as `v00`, pairwise `not_run` |

The retained champion for all targets is still `v00`. There are no apply-ready
candidate texts for Stage `03`.

## Clean-Context Evaluator Evidence

Clean-context subagents were used only for sanitized target/candidate/eval-case
packets. Their verdicts were copied into local artifacts under
`subagent_verdicts/` and normalized into the harness schema.

| Target | Subagent | Result |
|---|---|---|
| `workflow.staged_plan_runner` | `019f3cad-6a0d-7500-83d7-021e0c2dbe2d` | baseline retained; candidates v01-v09 lost `0-2` |
| `research.last30days` | `019f3cad-c738-7d10-88f8-20cff0c88b50` | rows blocked by severe X credential/cookie locality evidence |
| `coding.root_cause_debugging` | `019f3cad-c7d7-7df0-99d3-1645eabc7368` | baseline retained; candidates v01-v09 lost `0-2` |
| `review.architecture_review` | `019f3cad-c893-73a1-904e-8c763c150b54` | baseline retained; candidates v01-v09 lost `0-2` |
| `artifact.documents` | `019f3cad-c99f-7673-b576-2ccd69eb63a2` | baseline retained; candidates v01-v09 lost `0-2` |
| `plugin_tool.browser_in_app` | `019f3cad-ca6a-7d93-86f4-a308c2cd5fd4` | baseline retained; candidates v01-v09 lost `0-2` |

The key severe finding is local to `research.last30days`: the evaluator observed
instructions that ask for `XAI_API_KEY` entry and browser-cookie scanning. This
Stage did not edit source skills, so the finding is carried forward as a Stage
`03` recommendation/blocker, not as a repaired candidate.

## Harness Adjustment During Stage `02`

The Stage `01` harness originally treated any `secret_redaction_violations`
field as an aggregation exception. During this run that behavior would have
dropped the `research.last30days` evidence entirely. The harness was adjusted
narrowly so severe rows are preserved as:

- `pairwise_verdict=blocked`
- `candidate_vs_champion=not_run`
- `score_0_100` capped at `49`
- violation counts retained in `results.tsv`

This is a compatible local-tooling change: it does not accept unsafe candidates
and does not touch Roehub runtime, product contracts, source skills/plugins, or
production state.

## Quality Gates

| Gate | Result |
|---|---|
| Manifest validation | `passed`: `uv run python -m tools.codex_quality_benchmark.cli validate-manifest --manifest .codex/tmp/skill-plugin-auto-improve-benchmark-v1/stage02-20260707-subagents/manifest.json` |
| Aggregate | `passed`: `uv run python -m tools.codex_quality_benchmark.cli aggregate --run-dir .codex/tmp/skill-plugin-auto-improve-benchmark-v1/stage02-20260707-subagents` |
| Summary | `passed`: `uv run python -m tools.codex_quality_benchmark.cli summarize --run-dir .codex/tmp/skill-plugin-auto-improve-benchmark-v1/stage02-20260707-subagents` |
| Focused tests | `passed`: `uv run pytest -q tests/unit/tools/test_codex_quality_benchmark.py` -> `5 passed` |
| Focused lint | `passed`: `uv run ruff check tools/codex_quality_benchmark tests/unit/tools/test_codex_quality_benchmark.py` |
| Docs index | `passed`: `uv run python -m tools.docs.generate_docs_index`; `uv run python -m tools.docs.generate_docs_index --check` |
| Diff whitespace | `passed`: `git diff --check -- docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/02-ten-iteration-auto-improve-run.md docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/skill-plugin-auto-improve-benchmark-v1-stage-ledger.md docs/architecture/README.md tools/codex_quality_benchmark/scoring.py tests/unit/tools/test_codex_quality_benchmark.py` |

## File Manifest

Created:

- local-only `.codex/tmp/skill-plugin-auto-improve-benchmark-v1/stage02-20260707-subagents/`

Modified:

- `docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/02-ten-iteration-auto-improve-run.md`
- `docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/skill-plugin-auto-improve-benchmark-v1-stage-ledger.md`
- `tools/codex_quality_benchmark/scoring.py`
- `tests/unit/tools/test_codex_quality_benchmark.py`
- `docs/architecture/README.md` if docs index changes

Deleted:

- none

Outside expected paths:

- `tools/codex_quality_benchmark/scoring.py`
- `tests/unit/tools/test_codex_quality_benchmark.py`

Outside expected paths justification:

- Required to preserve severe redaction/locality evidence in `results.tsv`
  instead of aborting the Stage `02` aggregate and losing the target row.

Foreign changes excluded:

- Source skill/plugin files under `/Users/daniildegtyarev/.codex/skills/**` and
  `/Users/daniildegtyarev/.codex/plugins/cache/**` were not edited.
- Raw benchmark state remains under ignored `.codex/tmp/...`.
- Existing Stage `00` and Stage `01` durable reports were preserved.

Mixed files:

- `docs/architecture/README.md` may include generated index entries from earlier
  stages; Stage `02` owns only the docs-index consequence of this report update.

## Next-Stage Handoff

Stage `03` is allowed.

Stage `03` must treat this run as complete benchmark evidence with no
apply-ready winners:

- Do not apply candidate text from `v01`..`v10`; no candidate passed strict
  `2-0`.
- Carry forward the severe `research.last30days` redaction/locality finding as a
  repair recommendation or blocker.
- If global skill/plugin source edits are considered, require explicit user
  approval and a scoped follow-up plan; Stage `02` itself did not modify them.
- Use local run artifacts under `.codex/tmp/skill-plugin-auto-improve-benchmark-v1/stage02-20260707-subagents/`
  as the reproducible evidence source.
