"""One-shot mechanical repair using the pack updater's existing transaction primitives.

Only pending S08/S09 entry locators change. Scope, prompts, acceptance, claims and
accepted receipts stay unchanged. User authorized full sequential execution and
routine technical corrections; this does not grant stage execution authority.
"""
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("updater", BASE / "ledger_update.py")
updater = importlib.util.module_from_spec(spec)
spec.loader.exec_module(updater)
module = updater.validator_module(updater.VALIDATOR)
root = BASE.parents[3]
mapping = {
    "S08": {
        "../../../../tests/real_infra/market_data": "../../../../tests/real_infra/market_data/test_transactional_ingestion_postgres.py",
        "../../../../tests/real_infra/backtest": "../../../../tests/real_infra/backtest/test_artifact_dispatch.py",
    },
    "S09": {
        "../../../../tools/qa": "../../../../tools/qa/postgres_storage_prototype.py",
        "../../../../tests/perf_smoke": "../../../../tests/perf_smoke/contexts/indicators/test_compute_numba_perf_smoke.py",
        "../../../../src/trading/contexts/market_data/adapters/outbound/persistence/postgres": "../../../../src/trading/contexts/market_data/adapters/outbound/persistence/postgres/canonical_candle_reader.py",
    },
}

fd = os.open(BASE, os.O_RDONLY)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    before_text = updater.LEDGER.read_text()
    pack = module.Pack(root, updater.LEDGER)
    pack.binding(pack.data["claim_capability"]["evidence"], BASE / "ledger_update.py")
    updater.require(pack.data["ledger_status"] == "active", "Expected active pack")
    updater.require(pack.data["current_stage"] == "S07", "Expected S07 ownership")
    updater.require(pack.rows["S07"]["status"] == "in_progress", "S07 changed")
    original = copy.deepcopy(pack.data)
    preserved = {}
    for path in [BASE / "ledger_update.py", BASE / "implementation-plan.md", *BASE.glob("[0-9][0-9]-*.md"), *BASE.glob("receipts/**/*.md")]:
        preserved[str(path.relative_to(BASE))] = module.digest(path)
    for sid, replacements in mapping.items():
        row = pack.rows[sid]
        updater.require(row["status"] == "pending" and row["executor_claim"] is None and not row["execution_allowed"], "Target is no longer unclaimed pending")
        found = set()
        for item in row["contract"]["entry_inputs"]:
            old = item["path"]
            if old in replacements:
                updater.require(item["producer_stage"] is None and pack.path(old).is_dir(), "Unexpected old locator")
                updater.require(pack.path(replacements[old]).is_file(), "Replacement file missing")
                item["path"] = replacements[old]
                found.add(old)
        updater.require(found == set(replacements), "Mapping does not match baseline")
    # Receipt verification includes successor bindings. No accepted receipt may
    # bind a changed contract, and no receipt/evidence is regenerated.
    for row in pack.rows.values():
        if row["status"] == "accepted":
            pack.receipt(pack.path(row["transition_receipt"]), successor=False)
    # Producer reports remain legitimate deferred inputs. All other entry inputs
    # must be actual files now; this is not a claim of S08/S09 entry readiness.
    for sid in mapping:
        for item in pack.rows[sid]["contract"]["entry_inputs"]:
            if item["producer_stage"] is None:
                updater.require(pack.path(item["path"]).is_file(), "Unresolved non-produced entry input")
    evidence = BASE / "reports/pending-entry-input-repair.json"
    now = updater.timestamp()
    record = {"created_at": now, "authority": "User-authorized goal-driven execution and routine technical correction; staged-plan-runner migration procedure", "reason": "Bundled entry validator requires files; five pending entry locators incorrectly named directories", "mapping": mapping, "before_contracts": {sid: next(r["contract"] for r in original["stages"] if r["contract"]["id"] == sid) for sid in mapping}, "after_contracts": {sid: pack.rows[sid]["contract"] for sid in mapping}, "preserved_sha256": preserved, "proof_boundary": "Mechanical authoring repair only; full stage scope and acceptance unchanged; no stage claimed or accepted"}
    with evidence.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    pack.data.setdefault("execution_history", []).append({"at": now, "action": "migrate-pending-entry-locators", "stage": "S08,S09", "executor": "root-orchestrator", "receipt": None, "reason": "File-only locators corrected without changing scope or state", "evidence": "reports/pending-entry-input-repair.json"})
    updater.require(updater.LEDGER.read_text() == before_text, "Concurrent ledger change")
    for path, digest in preserved.items():
        updater.require(module.digest(BASE / path) == digest, "Concurrent bound artifact change")
    updater.write_ledger(before_text, pack.data, fd)
    checked = module.Pack(root, updater.LEDGER)
    for row in checked.rows.values():
        if row["status"] == "accepted":
            checked.receipt(checked.path(row["transition_receipt"]), successor=False)
    print(json.dumps({"status": "ok", "repaired_stages": list(mapping), "ledger_status": checked.data["ledger_status"], "current_stage": checked.data["current_stage"], "accepted_receipts_verified": 6, "evidence": str(evidence)}))
finally:
    os.close(fd)
