"""Read-only repository audit; writes sanitized evidence in this report directory."""
from pathlib import Path
import ast
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[6]
OUT = Path(__file__).resolve().parent
PACK = OUT.parent.parent
PATTERN = r'(?i)clickhouse|clickhouse_connect|MergeTree|groupBitmap|bitmapSubset|argMax\(|maxOrNull\(|toUnixTimestamp64Milli|\bFINAL\b'
paths = sorted(set(subprocess.check_output(['git', 'ls-files', '-co', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0')) - {''})
texts = {}
errors = []
for path in paths:
    file = ROOT / path
    if not file.is_file() or str(file).startswith(str(PACK)):
        continue
    try:
        texts[path] = file.read_text()
    except UnicodeDecodeError:
        continue
    except OSError as exc:
        errors.append({'path': path, 'error': type(exc).__name__})
assert not errors, errors
initial = json.loads((OUT / 'initial-appendix.json').read_text())
# Follow actual imports of every market-store port and artifact source contract,
# then their import consumers to a fixed point. Preserve edges, not just matches.
seeds = {p for p in texts if p.startswith('src/trading/contexts/market_data/application/ports/stores/') and p.endswith('.py')}
seeds |= {p for p,t in texts.items() if p.endswith('.py') and (p.endswith(('base_source_reconciliation.py', 'funding_rate_reader.py', 'backup_contracts.py', 'live_tail_repair.py')) or ('backtest_artifacts' in p and re.search(r'StoredSourceObservation|SourceTarget|content_unverified|source_state', t)))}
modules = {(p[4:] if p.startswith('src/') else p)[:-3].replace('/', '.'): p for p in texts if p.endswith('.py')}
imports = {}
for p, t in texts.items():
    if not p.endswith('.py') or not p.startswith(('src/','apps/','scripts/','tools/','tests/')):
        continue
    try:
        tree = ast.parse(t)
    except SyntaxError:
        errors.append({'path': p, 'error': 'syntax'}); continue
    imports[p] = set()
    package = (p[4:] if p.startswith('src/') else p)[:-3].split('/')[:-1]
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom): continue
        if node.level:
            base = package[:len(package) - node.level + 1]
            module = '.'.join(base + ([node.module] if node.module else []))
        else:
            module = node.module or ''
        if module: imports[p].add(module)
        imports[p].update(module + '.' + alias.name for alias in node.names if module and alias.name != '*')
    imports[p] |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
closure = set(seeds)
edges = set()
while True:
    added = set()
    for p, names in imports.items():
        for name in names:
            target = modules.get(name) or modules.get(name + '.__init__')
            if target in closure and p != target:
                edges.add((p, target)); added.add(p)
    added -= closure
    if not added: break
    closure |= added
hits = {p: [i for i,l in enumerate(t.splitlines(),1) if re.search(PATTERN,l)] for p,t in texts.items()}
hits = {p:lines for p,lines in hits.items() if lines or 'clickhouse' in p.lower()}

def classify(p):
    if p not in texts: return 'absent-at-baseline','S07','verify existing deletion; do not resurrect'
    if p.startswith('.codex/') or '-stage-reports/' in p or '/legacy/' in p or '/evidence/' in p or 'iteration-log' in p or '/benchmark_iterations/' in p or '/production-integration-' in p or '-stage-ledger' in p or '/deep-research-report' in p or p.endswith('market-data-efficiency-local-activation-20261011.md'):
        return 'immutable-history','S07','preserve bytes; supersede only at navigation boundary'
    if p not in closure and not p.startswith(('migrations/postgres/','alembic/versions/')) and not re.search(PATTERN.replace(r'|\bFINAL\b', ''), texts[p]) and 'clickhouse' not in p.lower():
        return 'unrelated-final-token','S07','preserve: FINAL is ordinary prose/type/test name, no storage dependency found'
    if p == 'docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1.md':
        return 'illustrative-nonruntime-reference','S07','preserve benchmark scope example; not a storage architecture requirement'
    if p.startswith(('migrations/postgres/','alembic/versions/')):
        return 'applied-schema-history','S02/S05','preserve checksum; append migration for affected constraints/state'
    if p.startswith('migrations/clickhouse/'):
        return 'legacy-executable-schema','S02/S07','map all fields to additive PG DDL; retire runtime CH migration'
    if p.startswith('schemas/'):
        return 'generated-schema','S06','version owner source then regenerate; reject incompatible inputs'
    if p.startswith(('configs/installation/generated/','docs/runbooks/generated/','docs/architecture/project-map/')) or p == 'docs/architecture/README.md':
        return 'generated-or-generation-source','S06/S07','update source and regenerate; no handwritten generated edits'
    if p.startswith('docs/') and p.endswith('.md'):
        return 'normative-doc','S01/S07','label current versus PG target now; reconcile after implementation'
    if p.startswith('docs/runbooks/'):
        return 'runbook-source','S06/S07','replace CH operation with meaningful PG operation; regenerate'
    if p.startswith('tests/'):
        return 'semantic-test-or-fixture','S02-S08','port semantic assertions; retire only engine-specific fixture with coverage mapping; not a rejection allowlist'
    if p.startswith('scripts/ops/') or 'clickhouse_exporter' in p:
        return 'retire-executable','S06/S07','retire CH-specific operation; no fake PG OPTIMIZE equivalent'
    if p.startswith('scripts/'):
        return 'supported-optional-script','S05/S07','port direct SQL, CLI/config and resume manifests; retain numerical parity'
    if p.startswith(('infra/','configs/','tools/installation/','tools/release/','tools/backup/','apps/migrations/','apps/control_agent/','apps/monitoring/','src/trading/platform/','src/trading/contexts/operations/')):
        return 'installation-operations','S06','PG-only generation/bootstrap/backup/health; preserve safety boundaries'
    if '/persistence/clickhouse/' in p:
        return 'active-storage-adapter','S03/S04/S07','replace by PG port implementation then delete legacy module'
    if p in ('pyproject.toml','uv.lock'):
        return 'dependency-manifest','S07','remove driver and regenerate lock after all imports removed'
    if p.startswith('tools/qa/') or p.startswith('tests/perf_smoke/'):
        return 'verification-tool','S07/S09','port runnable tool; preserve historical results outside executable paths'
    return 'active-consumer-or-contract','S05/S07','migrate source/wiring and declared contracts; prove supported consumer parity'

# Follow persisted SQL tables used by transitive consumers, not only CH literals.
consumer_text = '\n'.join(texts[p] for p in closure if p in texts)
schema_dependencies = set()
for p,t in texts.items():
    if p.startswith(('migrations/postgres/', 'alembic/versions/')):
        names = re.findall(r'(?i)(?:CREATE TABLE(?: IF NOT EXISTS)?|ALTER TABLE|REFERENCES)\s+([a-z_][a-z_0-9.]*)', t)
        if any(re.search(r'\b' + re.escape(n.split('.')[-1]) + r'\b', consumer_text) for n in names):
            schema_dependencies.add(p)
schema_dependencies |= {'migrations/postgres/manifest.json', 'migrations/clickhouse/manifest.json'}
inventory=[]
for p in sorted(set(initial)|set(hits)|closure|schema_dependencies):
    category,owner,disposition=classify(p)
    inventory.append(dict(path=p,category=category,owner=owner,disposition=disposition,initial_appendix=p in initial,match_lines=hits.get(p,[]),caller_closure=p in closure,schema_dependency=p in schema_dependencies,sha256=hashlib.sha256(texts[p].encode()).hexdigest() if p in texts else None))
(OUT/'inventory.json').write_text(json.dumps({'search_pattern':PATTERN,'universe':'tracked plus nonignored untracked; current pack excluded; binary contents excluded; path matches retained','errors':errors,'files':inventory},indent=2)+'\n')
(OUT/'caller-edges.json').write_text(json.dumps({'seeds':sorted(seeds),'edges':[{'caller':a,'dependency':b} for a,b in sorted(edges)]},indent=2)+'\n')
# Exact field inventory from source DDL, preserving nullable/default expressions.
tables=[]
for p in sorted(texts):
    if p.startswith('migrations/clickhouse/') and p.endswith('.sql'):
        for m in re.finditer(r'CREATE TABLE IF NOT EXISTS ([\w.]+)\s*\((.*?)\n\)',texts[p],re.S):
            fields=[line.split('--')[0].strip().rstrip(',') for line in m[2].splitlines() if line.split('--')[0].strip()]
            tables.append({'source':p,'table':m[1],'fields':fields,'owner':'S02/S03/S04'})
(OUT/'persisted-fields.json').write_text(json.dumps(tables,indent=2)+'\n')
# Baseline signatures and SQL expressions for every adapter method and port.
methods=[]
for p,t in texts.items():
    if not p.endswith('.py') or not ('/persistence/clickhouse/' in p or p in seeds): continue
    tree=ast.parse(t)
    for cls in [n for n in ast.walk(tree) if isinstance(n,ast.ClassDef)]:
        for n in cls.body:
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
                sql=[]
                for v in ast.walk(n):
                    if isinstance(v,ast.Constant) and isinstance(v.value,str) and re.search(r'\b(SELECT|INSERT INTO|UPDATE|DELETE FROM)\b',v.value):
                        sql.append(v.value)
                methods.append({'path':p,'class':cls.name,'method':n.name,'line':n.lineno,'signature':ast.unparse(n.args),'returns':ast.unparse(n.returns) if n.returns else None,'body_sha256':hashlib.sha256(ast.get_source_segment(t,n).encode()).hexdigest(),'sql_fragments':sql})
(OUT/'baseline-signatures.json').write_text(json.dumps(methods,indent=2)+'\n')
summary={'inventory_paths':len(inventory),'initial_paths':len(initial),'literal_or_path_hits':len(hits),'closure_paths':len(closure),'caller_edges':len(edges),'schema_dependencies':len(schema_dependencies),'tables':len(tables),'method_signatures':len(methods),'categories':{c:sum(r['category']==c for r in inventory) for c in sorted({r['category'] for r in inventory})},'errors':errors}
(OUT/'audit-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
