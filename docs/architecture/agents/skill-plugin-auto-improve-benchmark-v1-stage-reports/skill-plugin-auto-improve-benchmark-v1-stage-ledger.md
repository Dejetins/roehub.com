# Skill/Plugin Auto-Improve Benchmark v1 - журнал выполнения stages

Единый handoff-документ для выполнения плана через prompt pack или Codex Goal
mode.

## Execution Artifacts

- plan_doc: `docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1.md`
- prompt_pack_dir: `.codex/agents/generated/skill-plugin-auto-improve-benchmark-v1/`
- stage_ledger: `docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/skill-plugin-auto-improve-benchmark-v1-stage-ledger.md`
- execution_mode: `goal_driven`
- intended_agent_model: `gpt-5.5`
- reasoning_effort: `xhigh`
- ledger_status: `completed`
- current_stage: `completed`
- goal_mode_optional: `true`
- goal_artifact_required: `false`
- default_branch: `main`
- separate_branch_allowed: `false`
- worktree_allowed: `false`
- stash_allowed: `false`

## Update Rules

| Rule | Contract |
|---|---|
| Source of truth | Executor derives next work from this ledger plus `plan_doc` and `prompt_pack_dir`, not from chat memory. |
| Stage update | Every stage updates this file after validation and before final report. |
| Stage states | `pending`, `running`, `accepted`, `accepted_for_learning`, `blocked`, `rejected`, `superseded`, `skipped`, `completed`. |
| Goal mode | Continue only while the next stage is explicitly allowed. Stop on `blocked`, `completed`, missing artifact, missing evidence or required user approval. |
| Local-only | Python scripts and raw benchmark outputs are local to this machine. Durable summaries may be copied into this report directory. |
| Subagents | Clean-context subagents may evaluate sanitized prompts/answers. They are not persistence; verdicts must be copied into local JSON/TSV artifacts. |
| Secrets | Do not write secrets, tokens, cookies, credentials, raw provider payloads, env dumps or private keys into prompts, docs, ledgers, traces, logs or reports. |
| File manifest | Every stage report must list created, modified, deleted, outside expected paths, foreign changes excluded and mixed files. |

## Business And Operations Scope

Этот ledger нужен, чтобы локальный benchmark skills/plugins был воспроизводимым:
следующий executor видит текущий stage, blocker, score schema и handoff без
старого chat context.

| Surface | Status |
|---|---|
| Business impact | Улучшает качество локальных Codex skills/plugins через измеримый score вместо вкусового переписывания. |
| Roehub runtime/service calls | `N/A`: plan does not change product services, runtime workers, API, UI, persistence or deploy. |
| Auth/secrets | Secret handling is applicable only as a redaction constraint for local artifacts and subagent packets. |
| Alerts/monitoring | `N/A` for production alerts; stage status in this ledger is the operational signal. |
| Runbook | `plan_doc`, prompt pack and this ledger are the local runbook; no separate production runbook is required. |

## Stage Table

| Stage | Status | Prompt | Report/evidence target | Previous gate | Next allowed | Contract impact | Last evidence | Notes |
|---|---|---|---|---|---|---|---|---|
| `00` Baseline Inventory And Rubric | `accepted` | `.codex/agents/generated/skill-plugin-auto-improve-benchmark-v1/00-baseline-inventory-and-rubric.md` | `docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/00-baseline-inventory-and-rubric.md` | none | `true`: Stage `01` may run | `none` for Roehub runtime; `compatible-change` for local benchmark planning | Report `00-baseline-inventory-and-rubric.md`; inventory 23 global skills + 38 plugin skills; selected 6 targets; `auto-improve` SHA `6bcc4ef40d31736320c5650e3bd58bedba5a4edf`; README/criteria hashes recorded. | Source snapshot, target manifest, rubric, eval cases and clean-context packet frozen. No source skills/plugins edited. |
| `01` Local Benchmark Harness | `accepted` | `.codex/agents/generated/skill-plugin-auto-improve-benchmark-v1/01-local-benchmark-harness.md` | `docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/01-local-benchmark-harness.md` | Stage `00` accepted | `true`: Stage `02` may run | `compatible-change` for local tooling | Report `01-local-benchmark-harness.md`; `uv run pytest -q tests/unit/tools/test_codex_quality_benchmark.py` -> 5 passed; `uv run ruff check tools/codex_quality_benchmark tests/unit/tools/test_codex_quality_benchmark.py` -> passed; local fixture generated `results.tsv`, `events.jsonl`, `summary.md`. | Created stdlib-first harness under `tools/codex_quality_benchmark/`. Raw fixture state remains under `.codex/tmp/` and is excluded from durable repo manifest. |
| `02` Ten Iteration Auto-Improve Run | `accepted` | `.codex/agents/generated/skill-plugin-auto-improve-benchmark-v1/02-ten-iteration-auto-improve-run.md` | `docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/02-ten-iteration-auto-improve-run.md` | Stage `01` accepted | `true`: Stage `03` may run | `compatible-change` for local harness evidence handling; `none` for Roehub runtime/source skills | Report `02-ten-iteration-auto-improve-run.md`; local run `.codex/tmp/skill-plugin-auto-improve-benchmark-v1/stage02-20260707-subagents/`; `198` evaluation JSON; `54` pairwise JSON; `66` result rows; `0` accepted candidates. | Clean-context subagents evaluated sanitized packets. Baseline `v00` retained for five targets; `research.last30days` rows are blocked by severe redaction/locality evidence. No source skill/plugin files edited. |
| `03` Final Analysis And Handoff | `completed` | `.codex/agents/generated/skill-plugin-auto-improve-benchmark-v1/03-final-analysis-and-handoff.md` | `docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/03-final-analysis-and-handoff.md` | Stage `02` accepted | `false`: prompt pack closed | `compatible-change` for local docs/harness; `none` for Roehub runtime/source skills | Report `03-final-analysis-and-handoff.md`; summary regenerated from saved local run; `0` apply-ready candidates; source skills/plugins unchanged. | Final disposition rejects all Stage `02` candidates, records `/Users/daniildegtyarev/.codex/skills/last30days/SKILL.md` repair as `needs_user_approval`, and completes the plan. |

## Iteration Score Schema

Every target/version row must preserve this schema:

| Field | Required | Meaning |
|---|---|---|
| `run_id` | yes | Local benchmark run id. |
| `target_id` | yes | Stable id for one skill/plugin file. |
| `target_path` | yes | Local absolute or repo-relative path, redacted if sensitive. |
| `skill_type` | yes | One of the target types in `plan_doc`. |
| `iteration` | yes | `0` for baseline, `1..10` for candidate iterations. |
| `version_id` | yes | `v00`, `v01`, ... |
| `sha256` | yes | Hash of the exact candidate file text. |
| `approach_label` | yes | One of the ten approach labels in `plan_doc`. |
| `score_0_100` | yes | Weighted total score. |
| `dimension_scores_json` | yes | Dimension-level scores. |
| `pairwise_verdict` | yes | `candidate`, `champion`, `tie`, `blocked`, `not_run`; `not_run` is valid only for baseline, no-op or blocked rows. |
| `candidate_vs_champion` | yes | `2-0`, `1-1`, `0-2`, `not_run`. |
| `eval_cases_total` | yes | Fixed case count. |
| `eval_cases_passed` | yes | Passed fixed cases. |
| `contract_violations` | yes | Count or list reference. |
| `locality_violations` | yes | Count or list reference. |
| `secret_redaction_violations` | yes | Count or list reference. |
| `decision_reason` | yes | Short keep/discard/block explanation. |

## Current Handoff

- Current stage: `completed`.
- Next prompt: none; prompt pack is closed.
- Required before start: `N/A`.
- Branch/worktree/stash: stay on `main`; do not create branch, worktree or stash.
- Raw benchmark state must remain local and excluded from durable docs unless summarized.
- Future source skill/plugin repair requires a new explicit user request. The most concrete follow-up is `/Users/daniildegtyarev/.codex/skills/last30days/SKILL.md` redaction/locality repair plus focused clean-context rerun.

## Change Log

| Date | Stage | Change |
|---|---|---|
| 2026-07-07 | plan creation | Created draft plan, prompt pack links and initial stage ledger. |
| 2026-07-07 | Stage `00` | Accepted baseline inventory, methodology snapshot, target manifest, rubric and eval cases; Stage `01` is now allowed. |
| 2026-07-07 | Stage `01` | Accepted local deterministic benchmark harness, tests, lint and sample fixture run; Stage `02` is now allowed. |
| 2026-07-07 | Stage `02` | Blocked before candidate iterations because clean-context subagent/equivalent evaluator approval is required. |
| 2026-07-07 | Stage `02` | Repaired after explicit subagent approval; accepted clean-context run `stage02-20260707-subagents`, retained all baselines, recorded `research.last30days` severe redaction/locality blocker, and allowed Stage `03`. |
| 2026-07-07 | Stage `03` | Completed final analysis and handoff; rejected all Stage `02` candidates, applied no source skill/plugin edits, and closed the prompt pack. |
