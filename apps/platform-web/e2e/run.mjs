import { spawnSync } from 'node:child_process';
// Separate fixture lifetimes preserve the library's cold-empty assertion and
// isolate create/admission tests. Explicit file arguments select a focused run.
const selected = process.argv.slice(2);
const runs = selected.length ? [selected] : [['foundation.spec.ts', 'library.spec.ts'], ['builder.spec.ts'], ['compact-builder.spec.ts'], ['execution.spec.ts'], ['results.spec.ts'], ['journey.spec.ts']];
for (const args of runs) {
  const result = spawnSync('playwright', ['test', ...args.map(file => `(?:^|/)${file.replaceAll('.', '\\.')}$`)], { stdio: 'inherit' });
  if (result.status !== 0) process.exit(result.status ?? 1);
}
