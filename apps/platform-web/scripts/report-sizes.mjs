import { readFile, readdir, stat } from "node:fs/promises";
import path from "node:path";
import { gzipSync } from "node:zlib";

const repositoryRoot = process.cwd();
const packageManifest = JSON.parse(
  await readFile(path.join(repositoryRoot, "package.json"), "utf8"),
);
const lockfile = JSON.parse(
  await readFile(path.join(repositoryRoot, "package-lock.json"), "utf8"),
);

async function directoryBytes(target) {
  const targetStat = await stat(target);
  if (targetStat.isFile()) return targetStat.size;
  const entries = await readdir(target, { withFileTypes: true });
  let total = 0;
  for (const entry of entries) {
    total += await directoryBytes(path.join(target, entry.name));
  }
  return total;
}

const runtimeDependencies = {};
for (const [name, version] of Object.entries(packageManifest.dependencies)) {
  runtimeDependencies[name] = {
    version,
    installedBytes: await directoryBytes(path.join(repositoryRoot, "node_modules", name)),
  };
}

const distRoot = path.join(repositoryRoot, "apps/platform-web/dist");
const builtAssets = {};
for (const entry of await readdir(path.join(distRoot, "assets"))) {
  if (!entry.endsWith(".js") && !entry.endsWith(".css")) continue;
  const content = await readFile(path.join(distRoot, "assets", entry));
  builtAssets[entry] = {
    rawBytes: content.byteLength,
    gzipBytes: gzipSync(content, { level: 9 }).byteLength,
  };
}

const report = {
  measurement: "installed package bytes plus Vite production asset raw/gzip bytes",
  runtimeDependencies,
  runtimeDependencyInstalledBytes: Object.values(runtimeDependencies).reduce(
    (total, dependency) => total + dependency.installedBytes,
    0,
  ),
  lockfilePackageCount: Object.keys(lockfile.packages ?? {}).length,
  builtAssets,
  builtAssetRawBytes: Object.values(builtAssets).reduce(
    (total, asset) => total + asset.rawBytes,
    0,
  ),
  builtAssetGzipBytes: Object.values(builtAssets).reduce(
    (total, asset) => total + asset.gzipBytes,
    0,
  ),
};

process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
