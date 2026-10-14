"""Verify stage ownership and immutable predecessor evidence, without Git mutation."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
HERE = Path(__file__).resolve().parent
baseline = json.loads((HERE / 'baseline-files.json').read_text())
manifest = json.loads((HERE / 'owned-files.json').read_text())
owned = {row['path'] for row in manifest}
pack = '.codex/agents/generated/postgresql-only-storage-v1/'
protected = [p for p in baseline if (p.startswith('migrations/postgres/') and p.endswith('.sql'))
             or any('/' + prefix + '/S0' + stage in p for prefix in ('reports','receipts')
                    for stage in ('1','2'))]
def digest(path):
    file = ROOT / path
    return hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else None
bad_protected = [p for p in protected if digest(p) != baseline[p]]
unowned = [p for p in baseline if not p.startswith(pack) and p not in owned
           and digest(p) != baseline[p]]
stale = [row['path'] for row in manifest if digest(row['path']) != row['sha256']]
result = {'protected_files':len(protected), 'changed_protected':bad_protected,
          'changed_unowned_baseline_files':unowned, 'stale_manifest':stale,
          'foreign_dirty_changes':'excluded; baseline-status.txt',
          'result':'pass' if not (bad_protected or unowned or stale) else 'fail'}
print(json.dumps(result,indent=2))
assert result['result'] == 'pass'
