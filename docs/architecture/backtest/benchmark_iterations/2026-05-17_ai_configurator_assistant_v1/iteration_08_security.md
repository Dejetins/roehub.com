# Iteration 08 Security Eval

Дата: 2026-05-18.

Статус: accepted before delivery; direct-main delivery verification pending.

## Предварительный gate

Iteration 07 проверен перед началом:

- `implementation_progress.json`: `07-ops.accepted=true`;
- `next_iteration_allowed=true`;
- `iteration_07_ops.json`: `pushed_to_main=true`;
- `macstudio_verified=true`;
- recorded delivery commit: `1890236f0a05499e91ad3197c7dcd844971bbd97`;
- current `origin/main` contains the Iteration 07 delivery commit.

## Что изменено

- Security eval harness переведен на current conversation API:
  `/backtests/ai-config/conversations`, `/messages`, `/load-action`.
- Добавлен fixture pack `tests/fixtures/ai_configurator/security_eval_cases.json`.
- Eval cases покрывают prompt injection, system prompt extraction,
  `secrets_env_vars`, `output_script_injection`,
  `auto_run_backtest_attempt`, resource abuse, unsupported values and 10 safe prompts.
- Backend now hard-gates informational-only prompts so they cannot produce a
  load action.
- `BacktestAiLoadAction.as_mapping()` and prompt package serialization now
  recursively convert frozen mappings before JSON serialization.

## Security eval

Mac Studio run:

```text
uv run python scripts/backtest_ai/run_configurator_security_eval.py \
  --direct-lmstudio \
  --out-dir /tmp/roehub-security-eval-mac \
  --http-timeout-seconds 300 \
  --strict-acceptance-exit-code
```

Run id: `20260518T184755139675Z`.

Target: in-process conversation API harness with the real
`LMStudioOpenAICompatibleAdapter` and `POST /v1/chat/completions`.

## Metrics

| Metric | Required | Actual |
| --- | ---: | ---: |
| unauthorized actions | 0 | 0 |
| secret/path leakage | 0 | 0 |
| invalid load_action | 0 | 0 |
| safe prompts blocked | 0/10 | 0/10 |
| accepted cases | 18/18 | 18/18 |

## Evidence

- `security_eval.md`
- `security_eval.json`

## Local Checks

Completed before direct-main publish:

```text
uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/api
```

Result: `209 passed`.

```text
uv run ruff check src/trading/contexts/backtest/application/ai_configurator apps/api tests scripts
```

Result: passed.

```text
uv run pyright
```

Result: passed, `0 errors`.

```text
uv run python -m tools.docs.generate_docs_index --check
```

Result: passed.

## Contract Impact

| Surface | Classification | Notes |
| --- | --- | --- |
| Public API contract | compatible-change | Conversation API behavior is narrowed: info-only prompts cannot expose `load_action.enabled=true`. |
| Browser-visible behavior | compatible-change | Safe informational prompts now remain non-loadable instead of possibly filling a default config. |
| DTO schema | none | Response fields unchanged. |
| Persisted schema | none | No migrations. |
| Config schema | none | No config shape change. |
| Request hash/cache identity | none | Prompt/user hash inputs unchanged. |

## Delivery

Direct-main delivery is pending in this pre-publish evidence revision.

Acceptance marker before delivery:

- accepted: true;
- next iteration allowed: true;
- pushed to `origin/main`: false;
- Mac Studio security eval passed: true;
- final Mac Studio post-delivery verification: pending.
