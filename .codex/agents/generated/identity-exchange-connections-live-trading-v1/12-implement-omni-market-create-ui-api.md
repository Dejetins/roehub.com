---
prompt_name: identity_exchange_connections_v1_12_omni_market_create_ui_api
repo: roehub.com
branch: main
scope: "Stage 12: add compatible omni-market create in /settings and account API while preserving market-scoped exchange connections for execution."

language:
  implementation: python_fastapi_js_css
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, direct-main delivery, runtime evidence, secret handling"
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "Stage 12 source of truth and compatibility contract"
    - path: docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md
      why: "confirm Stage 11 acceptance and update Stage 12 handoff/evidence"
    - path: docs/architecture/identity/exchange-connections-stage-reports/11-strategy-binding-guard.md
      why: "accepted execution/binding guard baseline"
  task_entrypoints:
    - path: apps/api/dto/ui_account.py
      why: "public account DTO shape for create response/request"
      inspect_symbols: ["CreateExchangeConnectionRequest", "ExchangeConnectionResponse"]
    - path: apps/api/routes/ui_account.py
      why: "account facade create route and CSRF/recent-auth/audit behavior"
      inspect_symbols: ["post_exchange_connection"]
    - path: apps/api/exchange_control_client.py
      why: "single-market internal client boundary; do not change unless necessary"
      inspect_symbols: ["ExchangeControlClient", "create_connection"]
    - path: apps/web/templates/fragments/account/exchange_keys.html
      why: "/settings add-key form"
    - path: apps/web/dist/js/pages/settings.js
      why: "/settings payload construction, table refresh, modal status"
    - path: apps/web/dist/css/pages/settings.css
      why: "checkbox/segmented UI and responsive layout"
    - path: tests/unit/apps/api/test_ui_account_routes.py
      why: "API compatibility and secret-safe regression coverage"

documentation_continuity:
  plan_doc: "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  stage_report: "docs/architecture/identity/exchange-connections-stage-reports/12-omni-market-create-ui-api.md"
  ledger: "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  docs_gate: "python -m tools.docs.generate_docs_index --check"

hard_requirements:
  previous_stage_11_must_be_accepted: true
  market_type_legacy_required: true
  market_types_additive_required: true
  market_scoped_connections_required: true
  execution_guard_unchanged_required: true
  per_market_validation_result_required: true
  no_secret_response_required: true
  no_exchange_execution: true
  no_order_placement: true
  no_persistence_collapse_in_stage_12: true
  bybit_market_specific_validation_required: true
  browser_evidence_required: true
  runtime_evidence_required: true
  direct_main_push_after_validation_required: true

implementation_rules:
  - "Keep `CreateExchangeConnectionRequest.market_type` for legacy clients; add optional `market_types[]`."
  - "If `market_types[]` is present, call the existing single-market exchange-control create boundary once per selected market."
  - "Return the first created market as the top-level compatibility response and include optional `items[]` and `market_results[]` for multi-market clients."
  - "A partial multi-market failure may return a partial success response with per-market failed result, but a single-market failure must preserve existing error behavior."
  - "Do not create an omni execution handle; execution/readiness/strategy binding must keep using concrete `exchange_connection_id` rows."
  - "Do not migrate `exchange_credential_versions.connection_id` in this stage."
  - "The `/settings` form must use checkboxes for Spot/Futures and a visible Mainnet/Testnet segmented control."
  - "The `/settings` create payload should include `permissions=\"trade\"` as product intent while preserving the backend default for old clients."
  - "Bybit omni keys must still validate per market: Spot requires `SpotTrade`; Futures requires `ContractTrade` with `Order` + `Position`, `DerivativesTrade`, or `OptionsTrade` from `/v5/user/query-api`."
  - "Do not mark Futures Ready from a Spot-only active row; UI availability is derived from active market-scoped rows."
  - "Production repair of a missing Bybit futures binding must use UI/API create and must not use manual SQL insert/update."

skill_routing:
  - skill: architecture-design
    use_when: "confirming the target state and future credential-object boundary"
    timing: "before implementation"
  - skill: contract-impact-analysis
    use_when: "classifying API/DTO/persistence/execution/browser impacts"
    timing: "before finalizing implementation"
  - skill: backend-quality-gates
    use_when: "running pytest/ruff/pyright/docs checks and triaging failures"
    timing: "during verification"
  - skill: browser-qa-evidence
    use_when: "proving /settings controls and create payload/result in a real browser"
    timing: "during verification"
  - skill: publish-ci-deploy
    use_when: "local validation is complete"
    timing: "after verification"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api/test_ui_account_routes.py"
    expect: "passes"
  - cmd: "uv run ruff check apps/api/dto/ui_account.py apps/api/routes/ui_account.py apps/web/dist/js/pages/settings.js tests/unit/apps/api/test_ui_account_routes.py"
    expect: "passes"
  - cmd: "uv run pyright apps/api tests/unit/apps/api"
    expect: "passes or any unrelated baseline failure is documented"
  - cmd: "node --check apps/web/dist/js/pages/settings.js"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes after docs index generation"

runtime_acceptance:
  required: true
  acceptance_rule: "Stage 12 cannot be accepted from unit tests alone."
  required_evidence:
    - "authenticated /settings browser screenshot or structured Playwright evidence showing Spot/Futures checkboxes and visible Mainnet/Testnet segmented control"
    - "network payload proving `market_type` plus `market_types[]` for multi-market create"
    - "API/body proof that response has `items[]` and `market_results[]` without plaintext secret/ciphertext/HMAC"
    - "active-list proof that created rows remain market-scoped"
    - "Bybit repair proof, when applicable, showing active Spot and active Futures rows with `valid_trade_enabled` and `ready_for_trading` from sanitized API/DB evidence"
    - "Mac Studio smoke, exchange-control health, Prometheus/Monit/OpenBao post-deploy checks"

non_goals:
  - "Do not add exchange execution or submit orders."
  - "Do not make one `exchange_connection_id` valid for both spot and futures."
  - "Do not remove legacy `market_type`."
  - "Do not remove or migrate `permissions` compatibility fields in this stage."
  - "Do not implement the separate credential object migration in this stage."

final_report_format:
  language: ru
  sections: ["Вердикт", "Что изменено", "Runtime evidence", "Проверки", "Contract impact", "Direct-main delivery", "Residual risk"]
---

Implement Stage 12 exactly within the compatibility boundary above. Preserve
secret redaction, old single-market clients, and market-scoped execution guards.
