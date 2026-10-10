# S01 verification record

Date: 2026-10-08. Executor `s01-20261008-01a11bf9-7f61-contracts`.
All commands ran from repository root. This is observation evidence, not a product test suite.

| Check | Observed result | Boundary |
|---|---|---|
| `.venv/bin/python /Users/daniildegtyarev/.codex/skills/prompt-manager/scripts/validate_pack.py --root /Users/daniildegtyarev/Projects/roehub.com --ledger /Users/daniildegtyarev/Projects/roehub.com/.codex/agents/generated/data-artifact-management-v1/iteration-ledger.md --check entry --stage S01` | exit 0, status=pass, before claim | structure/file binding only |
| `.venv/bin/python .codex/agents/generated/data-artifact-management-v1/ledger_update.py claim --stage S01 --executor s01-20261008-01a11bf9-7f61-contracts` | exit 0, status=ok, stage_status=in_progress, ledger_status=active | exclusive claim |
| `.venv/bin/python .codex/agents/generated/data-artifact-management-v1/test_ledger_update.py -v` | exit 0, 9 tests passed | disposable ledger tests, no product/database writes |
| Read-only Python defaults/config resolution | 40 identical runtime-default and signal-rule IDs, 400 resolved signal targets; 11 prices/10 mappings; finite YAML defaults saved | configuration facts, not numerical execution |
| `cua.createBrowserTab("iab", "http://localhost:20120/data", {visible:false})`, AX read and screenshot | page loaded; baseline screenshot saved | read-only baseline; no user-flow QA |
| Entry validator with `--executor s01-20261008-01a11bf9-7f61-contracts` after report preparation | exit 0, status=pass | owned in-progress entry remains valid |
| Python source/requirement/foreign-file check | R01–R13 all have table rows; 29 explicit source paths exist and their file hashes match S01-source-bindings.json; 269 foreign baseline hashes unchanged; tracked changes are baseline-only; all S02 entry files exist | source-backed handoff, no future execution |

Source/catalog arithmetic used `YamlBacktestGridDefaultsProvider.from_yaml(config_path='configs/dev/indicators.yaml')`, `supported_indicator_ids_for_signals_v1()`, `load_backtest_artifacts_runtime_config(Path('configs/dev/backtest_artifacts.yaml'))` and canonical timeframe constants. Assertions checked exact catalog equality and 400 signal targets. `S01-catalog.json` retains plain non-secret configuration evidence and explicit sizing arithmetic. No NPY signal/hit computation was invoked.

Source bindings and baseline were checked with `hashlib.sha256(Path(path).read_bytes()).hexdigest()`. S02 inputs were resolved relative to the pack using its actual JSON front matter. These are existence/binding checks; they do not assert S02 test infrastructure or implementation readiness. The updater rechecks dependencies and next entry atomically during acceptance.

Not run: backend application pytest/Ruff/typecheck, docs index/map generators (no product/canonical-doc changes), real PostgreSQL/ClickHouse tests, browser command flows, benchmarks, provider access, migration, deployment. Existing test inventory was read to assign future proof. Numerical fixture descriptions are source inspection, not numerical parity results. Immutable receipt/next-stage validation is performed separately after review disposition is recorded.
