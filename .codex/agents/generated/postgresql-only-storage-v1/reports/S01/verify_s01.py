"""Validate documentation ownership, inventory closure and introduced local links."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[6]
OUT = Path(__file__).resolve().parent
rows = json.loads((OUT / 'doc-amendments.json').read_text())
pattern = r'\n## PostgreSQL-only target — S01, 2026-10-13\n\n.*?runtime, SQL schema или данные\.\n\n[^\n]+\n\n'
preserved = []
for row in rows:
    path = ROOT / row['path']
    text = path.read_text()
    match = re.search(pattern, text, re.S)
    assert match, path
    original = text[:match.start()] + text[match.end():]
    assert hashlib.sha256(original.encode()).hexdigest() == row['before_sha256'], path
    assert match.start() >= re.search(r'^# [^\n]+', text, re.M).end(), path
    preserved.append(row['path'])
allowed = set(preserved) | {'docs/architecture/README.md', 'docs/architecture/project-map/PROJECT_MAP.md', 'docs/architecture/project-map/project-map.json'}
changed = []
for row in json.loads((OUT / 'baseline-files.json').read_text()):
    path = ROOT / row['path']
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if digest != row['sha256']:
        assert row['path'] in allowed, f'Unexpected change: {path}'
        changed.append({'path': row['path'], 'before_sha256': row['sha256'], 'after_sha256': digest, 'kind': 'modified'})
inventory = json.loads((OUT / 'inventory.json').read_text())['files']
by_path = {r['path']: r for r in inventory}
assert set(json.loads((OUT / 'initial-appendix.json').read_text())) <= set(by_path)
assert all(r['owner'] and r['disposition'] for r in inventory)
for name in ['0024_market_data_full_history_v1.sql', '0029_backtest_artifact_management_v1.sql', '0034_market_data_provider_admission_v1.sql']:
    assert by_path['migrations/postgres/' + name]['schema_dependency']
assert not json.loads((OUT / 'audit-summary.json').read_text())['errors']
links = []
paths = [ROOT / 'docs/architecture/market_data/market-data-real-adapters-postgresql-parquet-v1.md']
for path in paths:
    for raw in re.findall(r'\]\(([^)]+)\)', path.read_text()):
        if '://' not in raw:
            destination = (path.parent / raw.split('#')[0]).resolve()
            assert destination.exists(), (str(path), raw)
            links.append(raw)
for row in rows:
    path = ROOT / row['path']
    raw = re.search(r'\[PG storage contract D1–D7\]\(([^)]+)\)', path.read_text())[1]
    assert (path.parent / raw).resolve().is_file(), path
    links.append(raw)
result = {'original_amended_documents_byte_preserved': len(preserved), 'all_other_baseline_files_unchanged': True, 'initial_appendix_coverage': 318, 'inventory_dispositions_complete': True, 'introduced_local_links_checked': len(links), 'runtime_schema_changes': False, 'changed': changed}
(OUT / 'ownership-and-contract-check.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k:v for k,v in result.items() if k != 'changed'}, indent=2))
