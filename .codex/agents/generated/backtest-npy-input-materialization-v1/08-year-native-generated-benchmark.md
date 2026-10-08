---
{
  "schema_version": "stage-prompt/v1",
  "prompt_pack_execution": {
    "plan_doc": "implementation-plan.md",
    "prompt_pack_dir": ".",
    "stage_ledger": "iteration-ledger.md"
  },
  "stage_contract": {
    "id": "S08",
    "title": "BTCUSDT year benchmark of native, generated and mixed NPY inputs",
    "prompt_path": "08-year-native-generated-benchmark.md",
    "report_path": "reports/S08.md",
    "receipt_dir": "receipts/S08",
    "depends_on": [
      "S07"
    ],
    "expected_touches": [
      "../../../../scripts/backtest/full_result_parity.py",
      "../../../../scripts/backtest/run_api_runner_benchmark_parity.py",
      "../../../../src/trading/contexts/backtest/application/services/v2/benchmark_accounting.py",
      "zone: Existing parity comparators only if the new semantic identity requires a tested update"
    ],
    "acceptance_criteria": [
      "W1–W4 use validated real year data and actual production N/G/M file paths with recorded code/config/dataset identities.",
      "Complete selected arrays, top-N and required lazy details agree; independent reference/self-check and resource/cleanup gates pass.",
      "Comparable measured costs and uncertainty are reported, including inclusive generated preparation and disk cost; no invented performance threshold or unsupported speedup claim.",
      "No one-off benchmark script, archive, npy fixture or JIT output remains in the repository; aggregate report is reproducible without pretending external scratch is permanent."
    ],
    "proof_boundary": "Measured local performance/correctness for the specified year workloads and machine; not other instruments, installations, 200-symbol throughput or a latency-SLO pass.",
    "validation": {
      "profile": "roehub-focused-gates-and-prompt-pack-artifacts/v1",
      "checks": [
        "Run the external harness with the frozen production environment and record the literal command/harness hash in reports/S08.md. The command does not exist until the external harness is authored; do not invent an existing repository benchmark entrypoint.",
        "Validate complete top-N using the existing full_result_parity contract with deliberately versioned semantic recipe matching; legacy context validation remains enforced. Run independent self-check for at least two combinations and C03 cases.",
        "Before/after repository status and scratch/database inventory prove no temporary benchmark code/data leaked into Git or the user database. Any comparator changes require focused tests, ruff and pyright as applicable."
      ],
      "requires_user_acceptance": false
    },
    "entry_inputs": [
      {
        "path": "implementation-plan.md",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/architecture/backtest/README.md",
        "producer_stage": null
      },
      {
        "path": "../../../../scripts/backtest/full_result_parity.py",
        "producer_stage": null
      },
      {
        "path": "../../../../scripts/backtest/run_api_runner_benchmark_parity.py",
        "producer_stage": null
      },
      {
        "path": "reports/S07.md",
        "producer_stage": "S07"
      }
    ]
  }
}
---

# S08 — BTCUSDT year benchmark of native, generated and mixed NPY inputs

Relevant implementation-plan sections: 3 evidence limits, 7 complete protocol, 9 report requirements.

## Execution authority and state

Read the live ledger before doing anything. It alone owns statuses, claims and decision packets. Execute exactly S08 when the owner submits this prompt, then stop with the next prompt link. The owner sends prompts sequentially; no separate stage acceptance is required. Follow `staged-plan-runner` for lifecycle rules and use this pack's `ledger_update.py` for every execution-state mutation. The former updater gap was resolved by explicit owner instruction on 2026-10-07. Never edit active ledger JSON by hand, steal another executor's claim, or expire a claim automatically.

Once a stage is validly claimed, complete the authorized implementation, verification and in-scope fixes. Routine reversible engineering choices stay delegated. Do not ask the owner to approve an unchanged design or each next stage. A real missing external prerequisite blocks only its dependent proof. A new material owner decision must be distinguished from a failed test, implementation detail or earlier-stage output. No publication, deployment, credential change, destructive existing-artifact cleanup, Goal, branch or worktree is implied.

## Context acquisition

Always read repository `AGENTS.md`, this stage's live ledger record and the implementation plan's authority, invariant, compatibility and relevant stage sections. Refresh current source and ownership before editing. The plan's context table establishes observed baseline, not permission to overwrite concurrent changes. Read the task entrypoints below and only conditional bundles needed for the named change. Earlier reports are evidence; implementation decisions remain constrained by this plan and current user corrections.

## Conditional skills

- `backend-quality-gates`: when Python implementation/gates begin; select checks from current CI and affected boundaries.
- `contract-impact-analysis`: before changing API/DTO, file schema, persistence, config, port, cache or identity guarantees; retain classifications and migration directions in the report.
- `backend-performance-evidence`: for measurements, timing contracts or performance claims; mandatory in S08.
- `root-cause-debugging`: when a concrete failure's cause is unknown; fix only after causal evidence.
- `browser-qa-evidence`: if the implementation changes browser-visible behavior; use `playwright-cli` for terminal browser mechanics only when no user-selected browser overrides it.
- `architecture-design`: only if a newly evidenced invariant requires refining an unsettled design detail; do not reopen accepted npy/materialization/storage choices merely by preference.
- `prompt-manager`: only for an authorized artifact revision; preserve claimed/accepted contracts and use runner reconciliation instead of rewriting live stage IDs.

## Implementation constraints

Use production code and real contract boundaries. Reuse the existing builder, indicator computation, hit-time calculation, validators and mmap loader. No compatibility wrappers, fake slot/hash metadata, npy-to-CanonicalCandleReader adapter, generic helper/provider framework, duplicate numerical pipeline or direct ClickHouse fallback. No benchmark scaffolding or large data in the repository. Do not change financial precision, timeline/sentinel semantics, funding, canonical variant identity, ranking or supported direction policy for speed.

Inspect existing tests before adding a named planned module; extend the current owner when it already proves that boundary. Do not leave TODO-driven production behavior, catch-all silent fallbacks or success-shaped placeholder evidence. Preserve foreign paths/hunks and record necessary outside-expected paths with their reason. Existing diagrams/docs/history do not authorize unrelated work.

## Reporting and transition

Write the stage report at the exact contract path in English. Include actual created/modified/deleted paths, outside-expected changes with reasons, foreign changes excluded, exact commands/results/counts, actual evidence locations, compatibility classifications, remaining risk and next-stage inputs. Never log credentials, tokens, raw provider payloads or environment dumps. Keep large benchmark scripts/data/results outside Git; only bounded aggregate proof belongs in reports.

After required proof passes, use `ledger_update.py accept` with the actual report/evidence and `--checks-passed`. The command generates a new immutable receipt, binds live plan/prompt/report/evidence bytes, validates the successor, and commits the ledger transition under its directory lock. No manual receipt/hash construction is needed. A nonzero command result is not a transition. The next eligible stage is enabled but not started. End this stage's turn with its result and the next prompt link; the owner will submit that prompt separately. No additional acceptance question is required. Never claim a pass for missing proof.

## Task entrypoints and conditional source bundle

- `docs/architecture/backtest/README.md`
- `scripts/backtest/full_result_parity.py`
- `scripts/backtest/run_api_runner_benchmark_parity.py`
- `docs/runbooks/indicators-numba-cache-and-threads.md`
- `src/trading/contexts/backtest/application/services/v2/benchmark_accounting.py`
- `apps/worker/backtest_job_runner/wiring/modules/full_job_compute.py`

Read `reports/S07.md` produced by S07 before claiming. Its absence now is a deferred dependency, not an owner-input question.

## Required work

1. Use backend-performance-evidence before measuring. Build the one-off harness in a unique external scratch directory. Follow plan section 7 exactly for BTCUSDT spot, December 2024 seed prefix, full 2025 year, 1m/15m data validation, checksum provenance and parameter grids. No benchmark harness, downloaded data or generated arrays may be left in the repository.

2. Use the trusted S01 native baseline and same-candidate N/G/M comparison separately. Preserve baseline and candidate code/config/fixture identities. No mocked indicator calculation, monkeypatch bypass of contracts, or direct array-to-engine substitute for the production generated-file path.

3. Run required W1 (three 81-row indicators, no risk, production pass-through), W2 (year risk preparation), W3 (bounded row-grid full risk job) and W4 (full API/worker/persistence/lazy cycle). Use complete top-10 and trade evidence plus existing independent self-check and hand-prescribed financial cases.

4. Use at least nine paired/interleaved steady-state samples per N/G/M W1–W3 workload and three comparable fresh-process samples. Document JIT/page-cache/thread/filesystem behavior. Reset generated directories each time, keep cold/startup separate, and never call unique files proof of cold OS cache.

5. Report measured preparation, compute, write, hash/validation, mmap, scoring, service, parent cleanup, persistence/API intervals with their actual boundaries. Capture peak RSS, disk IO/bytes, peak and remaining scratch, counts and full parity. Do not add overlapping stages.

6. Report median/min/max/IQR and paired deltas/ratios with raw output externally. Latency acceptance is not assessed because no slowdown threshold was specified; correctness/resource/leak gates remain required. Do not require generated mode to outperform native. Remove owned bulky temporary data and any isolated DB after recording aggregate evidence; verify cleanup.

## Verification and acceptance

- Run the external harness with the frozen production environment and record the literal command/harness hash in reports/S08.md. The command does not exist until the external harness is authored; do not invent an existing repository benchmark entrypoint.

- Validate complete top-N using the existing full_result_parity contract with deliberately versioned semantic recipe matching; legacy context validation remains enforced. Run independent self-check for at least two combinations and C03 cases.

- Before/after repository status and scratch/database inventory prove no temporary benchmark code/data leaked into Git or the user database. Any comparator changes require focused tests, ruff and pyright as applicable.

All acceptance criteria in the front-matter contract must be evidenced. Measured local performance/correctness for the specified year workloads and machine; not other instruments, installations, 200-symbol throughput or a latency-SLO pass.

## Adjacent-stage handoff

S09 receives aggregate results, exact proof boundaries, residual uncertainty and a verified cleanup inventory.

## Prefix attestation measurement boundary

The first-run canonical payload-prefix scan and parent persistence acknowledgement are real production costs. Include them in G, N and M according to the actual equivalent starting state; do not pre-attest only one mode or hide this cost in fixture setup. Separate retained published metadata validation from per-attempt proof work. Any legitimate reuse of an already attested identical recipe must be applied symmetrically and labeled as a separate warm scenario.

## Journal commands for this stage

Run from the repository root. Choose a unique executor ID for this execution and retain it for resuming the same claim; do not reuse another executor's ID. The commands below use placeholders that the executor substitutes, not text the owner must provide.

```sh
python3 .codex/agents/generated/backtest-npy-input-materialization-v1/ledger_update.py claim --stage S08 --executor UNIQUE_EXECUTOR_ID
```

Only after the claim succeeds, perform the stage. Write the English stage report to `reports/S08.md`, including exact checks, results, owned changes and proof limits. Then, only when all required checks actually passed:

```sh
python3 .codex/agents/generated/backtest-npy-input-materialization-v1/ledger_update.py accept --stage S08 --executor SAME_EXECUTOR_ID --checks-passed --evidence reports/S08.md
```

Evidence paths resolve relative to this pack. Supply additional `--evidence` paths when needed. The report itself may serve as bounded check evidence when it records actual commands/results; a bare pass assertion is insufficient. This command does not run tests or establish their meaning.

Use `resume --stage S08 --executor SAME_EXECUTOR_ID` only for your own existing claim. For an actual new owner decision, `needs-input` requires `--reason`, `--resume-condition` and readable `--evidence`; resume then also requires `--resolution` pointing to the recorded answer. For a concrete hard blocker, `block` requires `--reason` and readable `--evidence`. A hard-blocked instance is terminal and needs a separately authored revision under the canonical lifecycle; do not reset it. Routine implementation problems should be diagnosed and fixed within the current claim, not converted into owner approvals. `status` is read-only. Every failed command leaves the prior ledger intact; an unreferenced immutable receipt after interruption is evidence only and must not be treated as acceptance.
