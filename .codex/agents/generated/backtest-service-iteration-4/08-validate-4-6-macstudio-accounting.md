---
prompt_name: backtest_service_iteration_4_6_macstudio_accounting_validation
repo: roehub.com
branch: main
scope: "Validate Iteration 4.6 benchmark runner accounting on Mac Studio and update evidence records."

language:
  implementation: python
  agent_report: ru

skill_routing:
  - skill: backend-performance-evidence
    use_when: "validating benchmark stage accounting and reporting acceptance evidence"
    timing: before and during verification
    reason: "the task is benchmark evidence, not feature implementation"
  - skill: backend-quality-gates
    use_when: "running the local validation command and docs index check"
    timing: during verification
    reason: "keep repository checks focused and reproducible"

hard_requirements:
  macstudio_validation_required: true
  no_runtime_algorithm_changes: true
  update_iteration_4_6_evidence: true
  preserve_canonical_stage_order: true

non_goals:
  - "Do not change scoring, heap, proxy-fill, or memory-cleanup implementation."
  - "Do not add new benchmark stages."
  - "Do not compare service-only telemetry with canonical notebook stages."

expected_primary_touches:
  - "docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_4_6_benchmark_runner_accounting/README.md"
  - "docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_4_6_benchmark_runner_accounting/macstudio_accounting_validation.json"

quality_gates:
  - cmd: "uv run python scripts/backtest/validate_benchmark_accounting.py --out docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_4_6_benchmark_runner_accounting/local_accounting_validation.json"
    expect: "passes locally before Mac Studio validation"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes after evidence docs are updated"
---

# Task

Validate Iteration 4.6 benchmark runner accounting on Mac Studio and update the evidence record so it is no longer `local pass, Mac Studio validation pending`.

Done means:

- `macstudio_accounting_validation.json` exists in `docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_4_6_benchmark_runner_accounting/`;
- `README.md` records Mac Studio status as pass, with command, host, commit, and key accounting fields;
- local and Mac Studio validation agree that `request.top_n = 100`, `benchmark_top_k = 5`, `sample_warmup_top_k = 1`, `top_results_count = 5`, and `heap_capacity = 5`;
- validation confirms `prepare_pools -> prepare_pools_core`, `total -> total_without_warmup`, and `service_total_without_warmup` is not compared as a canonical stage;
- docs index check passes.

## Required context

Read only these first:

1. `.codex/AGENTS.md`
2. `docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_4_6_benchmark_runner_accounting/README.md`
3. `docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_4_6_benchmark_runner_accounting/local_accounting_validation.json`
4. `scripts/backtest/validate_benchmark_accounting.py`
5. `docs/architecture/backtest/benchmark_iterations/README.md`

Consult `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md` only if the accounting contract is ambiguous.

## Work plan

1. Run the local validation command and keep/update `local_accounting_validation.json` only if output changes.
2. Ensure the current code is available on Mac Studio:
   - if working from a committed/pushed branch, SSH to `macstudio` and update `/opt/roehub/app` to the target commit using the repository's normal deploy/sync path;
   - if `/opt/roehub/app` is a runtime copy, record that fact and the commit SHA used for validation.
3. On Mac Studio, run:

```bash
cd /opt/roehub/app
export PATH="/opt/homebrew/bin:$PATH"
uv run python scripts/backtest/validate_benchmark_accounting.py \
  --out docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_4_6_benchmark_runner_accounting/macstudio_accounting_validation.json
```

4. Copy or commit the updated Mac Studio evidence back into the repository.
5. Update `README.md`: replace pending wording with Mac Studio pass evidence, including host, command, commit/runtime-copy note, and the key JSON fields.
6. Run `python -m tools.docs.generate_docs_index --check`; regenerate the index first if it is stale because the README changed.

## Acceptance criteria

- Mac Studio validation result is `pass`.
- `service_total_compared_to_canonical` is `false`.
- `prepare_pools_alias_normalized` is `true`.
- canonical stage order is unchanged.
- service-only telemetry remains outside canonical stage comparison.
- final report in Russian states exact files changed, commands run, and whether `/opt/roehub/app` was a git checkout or runtime copy.
