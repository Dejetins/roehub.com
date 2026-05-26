# Stage 10C: Settings Trading CJM UI

Дата проверки: 2026-05-27.

Статус: local implementation, required gates and browser evidence complete;
direct-main CI/deploy and Mac Studio post-deploy evidence pending.

Scope: `/settings` browser CJM for trading-only exchange connections. Stage 10C
does not change backend auto-validation, does not reclassify existing production
rows, does not place or simulate orders, and does not add exchange-execution
behavior.

## Verdict

| Area | Expected result | Observed evidence | Verdict | Residual risk |
|---|---|---|---|---|
| Stage prerequisite | Stage 10B must be accepted before 10C starts. | Iteration ledger marks 10B accepted with direct-main CI/deploy and Mac Studio runtime evidence complete. | Accepted. | None. |
| Add form | No read/trade selector; create action is `Connect and validate`; default environment is mainnet. | `/settings` SSR and Playwright proof show no `name="permissions"` or `data-permissions*`, visible `Connect and validate`, `mainnet` default. | Accepted locally. | Post-deploy proof pending until direct-main delivery. |
| Testnet | Testnet remains available only as advanced/dev control. | Browser proof opens `Advanced testnet` details and then the environment listbox; default stays `mainnet`. | Accepted locally. | None. |
| List model | User sees `Active` and `History`, not a separate `Disabled` tab. | SSR/browser show only `data-exchange-status-filter="active"` and `"history"`; history loads `status=all` and filters non trading-ready rows client-side. | Accepted locally. | Existing old active non-ready production rows still require 10D reclassification. |
| Actions | Active rows show `Ready for trading`, `Re-check`, `Rotate`, `Disconnect`. | Browser proof uses a trading-ready active row and verifies these labels/actions. `Disconnect` still calls the existing backend disable route. | Accepted locally. | None. |
| Readonly/not-ready clarity | Readonly and invalid outcomes are not shown as partially successful active connections. | History row shows `Read-only not supported`; create POST returning invalid/not-ready moves the UI to History and displays `Invalid API key or secret`. | Accepted locally. | Production readonly browser proof remains 10E. |
| Secret safety | Password-manager suppression and secret input clearing remain intact. | Browser proof verifies autocomplete/manager-suppression attributes, text inputs with masked CSS, and empty secret fields after submit. New artifact grep has no matches for secret patterns. | Accepted locally. | Historical artifacts still contain field-name-only password/API labels from earlier stages; see Runtime Evidence. |

## CJM / UI Contract

| Surface | Stage 10C behavior |
|---|---|
| Add form intent | `/settings` product intent is trading-only. Browser create payload omits `permissions`; API compatibility default remains server-side only. |
| Primary CTA | `Connect and validate`. Manual validation is no longer the happy-path create step. |
| Environment | `mainnet` is default. `testnet` is reachable only through `Advanced testnet`. |
| Active | Contains only rows with `status=active`, `effective_capability=trading`, and `connection_readiness=ready_for_trading`. |
| History | Contains disabled/archived/rejected rows and any non trading-ready active rows until 10D repairs existing data. |
| Active actions | `Re-check`, `Rotate`, `Disconnect`. `Disconnect` posts to the existing `/disable` facade and confirmation now asks for `DISCONNECT`. |
| Compatibility fields | Legacy `permissions`, `requested_permissions`, and `effective_permissions` remain API fields but are not rendered as authoritative product readiness in `/settings`. The table now renders capability/readiness. |

## Browser Evidence

| Surface | Command / artifact | Sanitized result | Verdict |
|---|---|---|---|
| Local SSR labels | `curl -fsS http://127.0.0.1:8031/settings \| rg 'Connect and validate\|Active\|History\|Disconnect\|Re-check'` against a local authenticated SSR runtime. | Required labels were present in SSR HTML/locale payload. | Pass. |
| Local SSR negative check | `! curl -fsS http://127.0.0.1:8031/settings \| rg 'name="permissions"\|data-permissions\|>Disabled<'`. | No user-facing permissions selector and no separate Disabled tab marker. | Pass. |
| Authenticated browser | `uv run --with playwright==1.60.0 python ...` against local authenticated SSR with controlled API transport. | `output/playwright/settings-trading-cjm-proof.json` has `accepted=true`; screenshot `output/playwright/settings-trading-cjm-proof.png`. | Pass. |
| Network payload | Same Playwright run captured the create POST summary. | `contains_permissions=false`, `environment=mainnet`, `exchange_name=binance`, `market_type=futures`, label `browser_payload_probe`; secret values were not stored in the artifact. | Pass. |
| Secret artifact grep | `rg -n 'stage10\|api_secret\|apiKey\|password' output/playwright/settings-trading-cjm-proof.json output/playwright/settings-trading-cjm-proof.png 2>/dev/null \|\| true`. | No matches in the new Stage 10C artifacts. | Pass. |
| Historical artifact grep | `rg -n 'stage10\|api_secret\|apiKey\|password' output/playwright .playwright-cli \|\| true`. | Matches are pre-existing Stage 08/09/local browser artifacts and field-name-only UI text such as account/password labels; no new Stage 10C artifact match and no real secret values were added. | Explained. |

## Contract Impact

| Dimension | Classification | Reason |
|---|---|---|
| Public API / DTO | `compatible-change` | DTO shape and endpoints remain unchanged. Browser create payload no longer sends legacy `permissions`, relying on server-side compatibility default while readiness is determined by Stage 10A/10B capability fields. |
| Browser-visible behavior | Intentional product workflow change | Removes user-facing read/trade choice, replaces Disabled/Archived tabs with History, renames Validate to Re-check and Disable to Disconnect. |
| Persistence | `none` | No migration, table, column, or data repair in 10C. |
| Config / env | `none` | No new env keys. |
| Ops / runtime | `none` | Browser/HTML/runtime calls are evidence only. |
| Trading execution | `none` | No order placement, order simulation, execution process, or order ledger code added. |

## Validation

| Gate | Result | Evidence |
|---|---|---|
| Focused web/API tests | Passed: `53 passed`. | `uv run pytest -q tests/unit/apps/web/test_app_routes.py tests/unit/apps/api/test_ui_account_routes.py`. |
| Focused ruff | Passed. | `uv run ruff check apps/web apps/api tests/unit/apps/web tests/unit/apps/api`. |
| Browser evidence | Passed. | `output/playwright/settings-trading-cjm-proof.json`; screenshot `output/playwright/settings-trading-cjm-proof.png`. |
| Required pytest | Passed: `95 passed`. | `uv run pytest -q tests/unit/apps/web/test_app_routes.py tests/unit/apps/api/test_ui_account_routes.py tests/unit/contexts/exchange_control`. |
| Required ruff | Passed. | `uv run ruff check apps/web apps/api src/trading/contexts/exchange_control tests/unit/apps/web tests/unit/apps/api tests/unit/contexts/exchange_control`. |
| Required pyright | Passed: `0 errors`. | `uv run pyright apps/api src/trading/contexts/exchange_control tests/unit/apps/api tests/unit/contexts/exchange_control`. |
| Docs index | Passed. | `python -m tools.docs.generate_docs_index --check`; index regenerated with `python -m tools.docs.generate_docs_index`. |

## Direct-Main Delivery

| Item | Evidence | Result |
|---|---|---|
| Branch | `git branch --show-current`. | `main`. |
| Fast-forward | `git pull --ff-only origin main`. | Already up to date before implementation. |
| Commit / push | Pending. | Stage 10C implementation/report must be committed and pushed directly to `main` after required gates. |
| CI / deploy | Pending. | Watch GitHub Actions and Mac Studio deploy after push. |
| Post-deploy runtime | Pending. | Run authenticated `/settings` HTML/browser checks after deploy and update this report/ledger with final evidence. |

## Residual Risk And Stage 10D Handoff

| Risk / handoff | Owner stage | Required next action |
|---|---|---|
| Production may still contain pre-10B active rows that are not trading-ready. | 10D | Run controlled dry-run/execution reclassification through supported lifecycle paths; no physical delete. |
| Public API still accepts legacy `permissions` for compatibility. | 10D/10E | Keep treating these fields as deprecated/non-authoritative; do not reintroduce UI selector. |
| Stage 10C browser evidence is local until direct-main deploy completes. | 10C | After CI/deploy, run Mac Studio/public authenticated HTML/browser proof and record final evidence. |
