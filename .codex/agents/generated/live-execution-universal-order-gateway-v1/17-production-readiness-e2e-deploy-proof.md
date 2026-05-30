---
prompt_name: 'live-execution-v1-stage-17-production-readiness-e2e-deploy-proof'
repo: 'roehub.com'
branch: 'main'
scope: 'Prove the full safe live-trading path end to end with latency/slippage metrics, canary protocol, alerts, runbooks, rollback, deploy, and post-deploy evidence.'
language: 'en'
context_sources:
  always_read:
  - '.codex/AGENTS.md'
  - 'docs/architecture/live_execution/live-execution-universal-order-gateway-v1.md'
  - 'docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/live-execution-universal-order-gateway-v1-iteration-ledger.md'
  task_entrypoints:
  - 'docs/architecture/live_execution'
  - 'docs/runbooks'
  - 'configs'
  - 'deploy'
  - 'apps/exchange_execution'
  - 'apps/web/templates/pages/strategies.html'
  - 'apps/web/dist/js/pages/strategies.js'
  - 'apps/web/dist/css/pages/strategies.css'
  - 'apps/web/locales/{ru,en}.json'
  conditional_bundles:
  - name: 'runtime_evidence'
    read_when: "The stage touches this surface or validation needs it."
    paths:
      - 'configs'
      - 'deploy'
      - 'docs/runbooks'
      - 'apps/*'
  - name: 'ui_evidence'
    read_when: "The stage touches this surface or validation needs it."
    paths:
      - 'apps/web/templates'
      - 'apps/web/dist'
      - 'tests/e2e'
  - name: 'schema_evidence'
    read_when: "The stage touches this surface or validation needs it."
    paths:
      - 'migrations'
      - 'src/trading/contexts/*/adapters/outbound/persistence'
      - 'tests/unit/apps/migrations'
  consult_if_needed:
  - 'apps/exchange_execution/**'
  - 'src/trading/contexts/live_execution/**'
  - 'apps/web/templates/pages/strategies.html'
  - 'apps/web/dist/js/pages/strategies.js'
  - 'apps/web/dist/css/pages/strategies.css'
  - 'apps/web/locales/{ru,en}.json'
  - 'tests/e2e/**'
  reading_budget: "Start with always_read + task_entrypoints; add conditional bundles only when the touched surface requires them."
  stop_conditions:
  - "Stop if the previous required stage is not accepted in the ledger."
  - "Stop if acceptance requires credentials, runtime, or a service that is not available and cannot be provisioned safely inside this stage. Record the stage as blocked; do not downgrade required runtime evidence to tests."
  - "Stop if implementation would require mainnet order submit or direct secret exposure."
documentation_continuity:
  canonical_plan: 'docs/architecture/live_execution/live-execution-universal-order-gateway-v1.md'
  stage_report: 'docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/17-production-readiness-e2e-deploy-proof.md'
  ledger: 'docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/live-execution-universal-order-gateway-v1-iteration-ledger.md'
  docs_index_check: 'python -m tools.docs.generate_docs_index --check'
  rule: 'Update affected current docs plus the stage report and ledger before final output.'
hard_requirements:
  - 'Work on main only. Do not create a stage branch or PR unless the user explicitly overrides this prompt.'
  - 'Before implementation, confirm the previous required stage is accepted in the iteration ledger. Stop and report if it is not accepted, unless this prompt is explicitly being used to repair, unblock, or supersede that previous stage.'
  - 'Run local quality gates first, then prove the stage through real boundary calls. Tests are supporting evidence, not stage acceptance; missing required runtime evidence means blocked, not accepted.'
  - 'Update the stage report and the iteration ledger after validation and before the final report.'
  - 'After successful validation, use publish-ci-deploy for direct-main commit, push, CI/deploy watch, and post-deploy evidence. If local GitHub/deploy permissions are unavailable, record the exact blocker and do not mark the stage accepted as delivered.'
  - 'Never print, log, persist in reports, or commit secrets, cookies, raw Authorization headers, API keys, private keys, exchange signed payloads, passphrases, or sensitive provider responses.'
  - 'Any unknown exchange/order side effect must be reconciled from a durable source of truth before retry. Blind retries are forbidden.'
  - 'If a planned path does not exist, find the nearest bounded-context equivalent and record the deviation in the stage report and ledger.'
  - 'Run the full safe user journey from backtest variant to strategy profile, run, signal, source event, risk decision, Redis dispatch, exchange-execution testnet outcome, fill/funding/reconciliation, notification, and UI outcome link.'
  - 'Record latency/slippage metrics from source signal time to accepted/risk/dispatch/submit/ack/fill stages; do not make latency claims without runtime evidence.'
  - 'Prometheus/Monit/alert rules must include severity, owner, escalation, and runbook actions for DLQ, clock drift, private stream, rate limits, reconciliation lag, PITR/backup, and unknown state.'
  - 'Mainnet submit remains blocked until a separate explicit canary approval; this stage may define canary protocol but must not silently enable mainnet.'
task_toggles:
  direct_main_delivery: true
  create_stage_branch: false
  create_pull_request: false
  tests_are_acceptance: false
  mainnet_order_submit_allowed: false
  secrets_in_reports_allowed: false
skill_routing:
  - 'contract-impact-analysis for final public/runtime contract review'
  - 'backend-quality-gates during verification'
  - 'backend-performance-evidence for latency/slippage evidence'
  - 'browser-qa-evidence plus playwright for the full browser journey'
  - 'publish-ci-deploy for direct-main CI/deploy/post-deploy proof'
target_envs:
  - 'local-dev'
  - 'browser-runtime'
  - 'redis-runtime'
  - 'exchange-testnet'
  - 'macstudio-prod'
  - 'prometheus-runtime'
  - 'monit-runtime'
required_literals:
  - '17'
  - 'docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/17-production-readiness-e2e-deploy-proof.md'
  - 'latency'
  - 'slippage'
  - 'canary'
  - 'kill_switch'
  - 'DLQ'
  - 'PITR'
  - 'severity'
  - 'owner'
  - 'escalation'
non_goals:
  - 'No mainnet order submission in this plan stage unless the plan explicitly says a later separate canary approval has happened.'
  - 'No direct exchange SDK/API/secrets access from Strategy, ML, browser UI, or apps/api producers.'
  - 'No broad repository refactor, dependency churn, or unrelated UI redesign.'
  - 'No tests-only acceptance for public/API, persistence, Redis, UI, runtime, ops, performance, or provider integration changes.'
final_report_format:
  - 'Status: accepted | blocked | failed'
  - 'Files changed: grouped by code/config/docs/tests/ops'
  - 'Real-boundary evidence: exact commands/calls and concise results'
  - 'Quality gates: commands run and outcomes'
  - 'Contract impact: public/API, persistence, Redis, config, runtime/ops, UI/browser'
  - 'Secrets/redaction: grep/log evidence and any residual risk'
  - 'Ledger/report updates: paths and summary'
  - 'Publish/deploy: commit SHA, push status, CI/deploy status, post-deploy smoke or blocker'
  - 'Next-stage handoff: facts the next stage must know'
quality_gates:
  - 'uv run ruff check src apps tests'
  - 'uv run pyright src apps tests'
  - 'uv run pytest -q tests/unit tests/integration tests/e2e'
  - 'python -m tools.docs.generate_docs_index --check'
  - 'git diff --check'
validation_strategy:
  depth: 'Real boundary evidence is mandatory; local tests are supporting gates only.'
  acceptance_surfaces:
      - 'Use Playwright for the whole browser journey and capture enough screenshots/traces to prove every browser-visible acceptance step.'
      - 'Use API calls, SQL queries, redis-cli, metrics scrape, Monit/launchd checks, and testnet exchange evidence for non-browser boundaries.'
      - 'Run rollback/kill-switch commands in a safe environment and prove the system stops submitting while retaining durable ledger state.'
      - 'Run post-deploy smoke after direct-main delivery, or record the exact deployment blocker and owner.'
  tests_only_exception: 'Not allowed for this stage unless the stage is explicitly reduced to docs-only and the reason is recorded.'
stage_execution_ledger:
  path: 'docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/live-execution-universal-order-gateway-v1-iteration-ledger.md'
  update_rule: 'Update this ledger after validation and before final report with status, evidence, blockers, contract impact, rollback, and next-stage handoff.'
  previous_stage_requirement: 'Stage 16 must be accepted in docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/live-execution-universal-order-gateway-v1-iteration-ledger.md before starting.'
  secret_policy: 'Do not include secrets, tokens, cookies, raw signed payloads, ciphertext, passphrases, or sensitive exchange/provider responses.'
expected_primary_touches:
  - 'docs/runbooks/exchange-execution.md'
  - 'docs/runbooks/**live**.md'
  - 'configs/**prometheus**'
  - 'deploy/**'
  - '.github/workflows/**'
  - 'docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/live-execution-universal-order-gateway-v1-iteration-ledger.md'
  - 'docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/17-production-readiness-e2e-deploy-proof.md'
possible_secondary_touches:
  - 'apps/exchange_execution/**'
  - 'src/trading/contexts/live_execution/**'
  - 'apps/web/templates/pages/strategies.html'
  - 'apps/web/dist/js/pages/strategies.js'
  - 'apps/web/dist/css/pages/strategies.css'
  - 'apps/web/locales/{ru,en}.json'
  - 'tests/e2e/**'
safety_notes:
  - 'This is a money-adjacent plan. Prefer fail-closed behavior and durable audit over silent best-effort success.'
  - 'Redis is transport, not source of truth. Postgres ledger plus exchange reconciliation owns truth where money is involved.'
  - 'Do not bypass exchange key custody boundaries introduced by the exchange-connections work.'
---
# Task
Implement Stage 17: Production Readiness E2E Deploy Proof.

Prove the full safe live-trading path end to end with latency/slippage metrics, canary protocol, alerts, runbooks, rollback, deploy, and post-deploy evidence.

## Context / Current State
This prompt implements one stage of `docs/architecture/live_execution/live-execution-universal-order-gateway-v1.md`. The stage ledger is `docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/live-execution-universal-order-gateway-v1-iteration-ledger.md` and the stage report must be `docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/17-production-readiness-e2e-deploy-proof.md`.

Previous-stage rule: Stage 16 must be accepted in docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/live-execution-universal-order-gateway-v1-iteration-ledger.md before starting.

Known plan constraints to preserve:
- Live trading is producer-neutral: Strategy, Manual, ML, and Ops sources must enter through source-event/risk/ledger boundaries when the stage reaches execution surfaces.
- Redis is a fast transport only; durable truth lives in Postgres plus exchange reconciliation.
- Mainnet order submit is not allowed in this stage unless a later separate approval explicitly enables it.
- Secrets stay inside approved custody/exchange-control/exchange-execution boundaries; browser, Strategy, ML, and apps/api must not receive raw credentials.

## Requirements (Must)
- Run the full safe user journey from backtest variant to strategy profile, run, signal, source event, risk decision, Redis dispatch, exchange-execution testnet outcome, fill/funding/reconciliation, notification, and UI outcome link.
- Record latency/slippage metrics from source signal time to accepted/risk/dispatch/submit/ack/fill stages; do not make latency claims without runtime evidence.
- Prometheus/Monit/alert rules must include severity, owner, escalation, and runbook actions for DLQ, clock drift, private stream, rate limits, reconciliation lag, PITR/backup, and unknown state.
- Mainnet submit remains blocked until a separate explicit canary approval; this stage may define canary protocol but must not silently enable mainnet.
- Preserve repository conventions from `.codex/AGENTS.md` and existing DDD / ports-and-adapters boundaries.
- Keep the diff scoped to this stage and record any deviation from the planned file map.
- Update the stage report and iteration ledger after validation and before the final report.
- Use real boundary calls for acceptance. Unit tests, lint, type checks, and docs checks are required gates, not acceptance by themselves.
- Keep all reports, logs, ledgers, screenshots, and output secret-safe.

## Requirements (Should)
- Prefer existing local helper APIs, DTO patterns, repository adapters, migration style, metrics helpers, and UI conventions.
- Keep error/status/reason codes stable and explicit enough for later stages.
- Add focused regression tests around the changed domain/use-case/persistence/API behavior.
- Update related old/current docs so they no longer describe stale behavior.

## Requirements (Nice-to-have)
- Add compact diagnostic commands or scripts only when they reduce repeated manual validation work for later stages.
- Include screenshots or traces for browser-visible changes when they help handoff, without leaking secrets.

# Context acquisition protocol
1. Read `.codex/AGENTS.md`, the main plan, and the iteration ledger first.
2. Confirm the previous required stage status in the ledger. Stop if it is not accepted, unless this prompt is being used for an explicit repair/unblock/supersede.
3. Read only the task entrypoints first. Add conditional bundles only if the touched surface requires them.
4. Identify affected contracts before implementation: public/API, persistence, Redis, config, runtime/ops, UI/browser, metrics, logs, docs.
5. If a planned service, table, stream, credential, runtime, or browser target is missing, either provision it inside this stage if the plan assigns it here, or record the stage as blocked. Only explicitly optional/non-applicable surfaces may be skipped, and the report must state why.

# Reading manifest
Always read:
- `.codex/AGENTS.md`
- `{PLAN}`
- `{LEDGER}`

Task entrypoints:
- `docs/architecture/live_execution`
- `docs/runbooks`
- `configs`
- `deploy`
- `apps/exchange_execution`
- `apps/web/templates/pages/strategies.html`
- `apps/web/dist/js/pages/strategies.js`
- `apps/web/dist/css/pages/strategies.css`
- `apps/web/locales/{ru,en}.json`

Conditional bundles:
- Read docs/runbooks/config/deploy files only when this stage touches runtime, ops, or deployment surfaces.
- Read UI/templates/static/tests only when this stage changes or validates browser-visible behavior.
- Read migration/persistence files when this stage changes schema, DB read models, or durable state.

Stop conditions:
- Previous required stage is not accepted.
- Acceptance needs an unavailable service/credential and the plan does not assign provisioning to this stage.
- The work would require mainnet order submit, raw secret exposure, broad refactor, or unrelated product changes.

# Work plan (agent should follow)
1. Record current ledger status and previous-stage acceptance.
2. Inspect the bounded code/docs entrypoints and identify exact existing patterns to follow.
3. Classify contract impact before editing.
4. Implement the smallest stage-scoped code/config/UI/docs changes.
5. Add or update focused tests for the changed domain/use-case/API/persistence behavior.
6. Run local quality gates.
7. Run real-boundary validation for the acceptance surfaces below.
8. Update `{report_path}` with files changed, docs sync, service calls, error behavior, runtime config, latency/slippage if applicable, logging/redaction, monitoring, contract impact, rollback, and evidence.
9. Update `{LEDGER}` after validation and before the final report.
10. After successful implementation and validation, use `publish-ci-deploy` for direct-main commit/push/CI/deploy/post-deploy evidence. If GitHub or deploy access is unavailable, record the exact blocker and do not report the stage as delivered.

# Acceptance criteria (Definition of Done)
- Full Playwright + API + DB + Redis + metrics + testnet exchange proof is recorded.
- No stuck pending/DLQ messages or unexplained unknown states remain after the E2E run.
- PITR/backup, retention/partition, private stream reconnect, clock drift, limiter/backpressure, and notification evidence are recorded.
- Direct-main publish, CI/deploy watch, and post-deploy smoke are completed through publish-ci-deploy after successful implementation and validation; if access is unavailable, the stage is blocked for delivery.
- Stage report and iteration ledger are updated with evidence, blockers if any, contract impact, rollback path, and next-stage handoff.
- Quality gates pass or failures are classified as introduced, required-path pre-existing, unrelated pre-existing, environmental, or flaky.
- Final report includes exact real-boundary evidence, not only tests.

# Implementation constraints
- Preserve DDD boundaries and dependency direction. Domain/application code must not depend on web/UI/infrastructure details.
- Use structured DTOs/parsers/repositories instead of ad hoc string manipulation when local patterns exist.
- Do not add dependencies unless the stage cannot be implemented safely without them and the stage report justifies the addition.
- Use stable reason/status codes for user-facing and API-facing failures.
- Metrics must avoid unbounded labels and secrets. Logs must redact sensitive identifiers.
- Unknown side-effect result handling must use durable ledger/provider lookup/reconciliation before retry.
- Update documentation using the sibling stage-report/architecture/runbook shape already present in this repository.

# Files to indicate (expected touched areas)
Expected primary touches:
- `docs/runbooks/exchange-execution.md`
- `docs/runbooks/**live**.md`
- `configs/**prometheus**`
- `deploy/**`
- `.github/workflows/**`
- `docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/live-execution-universal-order-gateway-v1-iteration-ledger.md`
- `docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/17-production-readiness-e2e-deploy-proof.md`

Possible secondary touches:
- `apps/exchange_execution/**`
- `src/trading/contexts/live_execution/**`
- `apps/web/templates/pages/strategies.html`
- `apps/web/dist/js/pages/strategies.js`
- `apps/web/dist/css/pages/strategies.css`
- `apps/web/locales/{ru,en}.json`
- `tests/e2e/**`

If you touch a different path, explain why in the stage report and ledger.

# Non-goals
- No mainnet order submission in this plan stage unless the plan explicitly says a later separate canary approval has happened.
- No direct exchange SDK/API/secrets access from Strategy, ML, browser UI, or apps/api producers.
- No broad repository refactor, dependency churn, or unrelated UI redesign.
- No tests-only acceptance for public/API, persistence, Redis, UI, runtime, ops, performance, or provider integration changes.

# Quality gates (must run and pass)
- `uv run ruff check src apps tests`
- `uv run pyright src apps tests`
- `uv run pytest -q tests/unit tests/integration tests/e2e`
- `python -m tools.docs.generate_docs_index --check`
- `git diff --check`

Additional real-boundary validation required:
- Use Playwright for the whole browser journey and capture enough screenshots/traces to prove every browser-visible acceptance step.
- Use API calls, SQL queries, redis-cli, metrics scrape, Monit/launchd checks, and testnet exchange evidence for non-browser boundaries.
- Run rollback/kill-switch commands in a safe environment and prove the system stops submitting while retaining durable ledger state.
- Run post-deploy smoke after direct-main delivery, or record the exact deployment blocker and owner.

# Final output: report format (strict)
Return a concise report with these headings:
- `Status`
- `Files changed`
- `Real-boundary evidence`
- `Quality gates`
- `Contract impact`
- `Secrets/redaction`
- `Stage report and ledger`
- `Publish/deploy`
- `Next-stage handoff`

Do not include secrets, cookies, tokens, raw signed payloads, raw exchange private responses, or raw browser session data in the final report.
