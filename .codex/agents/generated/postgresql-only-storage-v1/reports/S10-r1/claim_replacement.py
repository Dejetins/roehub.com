"""One-shot authorized replacement claim using existing directory flock/CAS/fsync.

Preserves terminal S10 and every old prompt/receipt. Does not change updater bytes.
The ordinary updater can resume/accept this active replacement after this transaction.
"""

import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path

base = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("updater", base / "ledger_update.py")
updater = importlib.util.module_from_spec(spec)
spec.loader.exec_module(updater)
module = updater.validator_module(updater.VALIDATOR)
root = base.parents[3]
fd = os.open(base, os.O_RDONLY)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    before = updater.LEDGER.read_text()
    pack = module.Pack(root, updater.LEDGER)
    pack.binding(pack.data["claim_capability"]["evidence"], base / "ledger_update.py")
    updater.require(pack.data["ledger_status"] == "blocked", "Expected blocked ledger")
    updater.require(pack.data["current_stage"] == "S10", "Unexpected current stage")
    updater.require(pack.rows["S10"]["status"] == "blocked", "Preserve terminal S10")
    updater.require("S10-r1" not in pack.rows, "Replacement already exists")
    repair = base / "reports/S10-r1/provider-repair.json"
    rows = json.loads(repair.read_text())
    updater.require(
        len(rows) == 4
        and all(
            r["http_status"] == 200 and r["api_ret_code"] in (None, 0) and r["instrument_count"] > 0
            for r in rows
        ),
        "Provider repair not established",
    )
    old_rows = copy.deepcopy(pack.data["stages"])
    paths = [
        base / "ledger_update.py",
        base / "implementation-plan.md",
        *base.glob("[0-9][0-9]-*.md"),
        *base.glob("receipts/**/*.md"),
        base / "reports/S10.md",
        *(base / "reports/S10").rglob("*"),
    ]
    preserved = {str(p.relative_to(base)): module.digest(p) for p in paths if p.is_file()}
    contract = copy.deepcopy(pack.rows["S10"]["contract"])
    contract.update(
        id="S10-r1",
        title="Selected-provider live completion after access repair",
        prompt_path="10-r1-live-completion.md",
        report_path="reports/S10-r1.md",
        receipt_dir="receipts/S10-r1",
    )
    contract["entry_inputs"] += [
        {"path": "reports/S10.md", "producer_stage": None},
        {"path": "reports/S10-r1/provider-repair.json", "producer_stage": None},
    ]
    prompt = base / contract["prompt_path"]
    prose = """
# S10-r1 — Complete remaining live obligations

This authorized revision replaces terminal blocked S10; inspect the ledger for claim.
Read AGENTS.md, implementation-plan.md sections 5–6, original S10 prompt/report,
provider repair evidence and current installation/runbooks. Original acceptance
criteria and S09 dependency remain unchanged. Reuse immutable S10 evidence where
still current; do not repeat its reset or rewrite its report, prompt or receipts.

Continue real selected-source reference/funding/history sync, observe at least ten
consecutive closed minutes for available selected spot/futures and a bounded
controlled reconnect. Independently compare installed SQL, API and event evidence.
Build the selected futures artifact without funding_unavailable; run an ordinary
backtest and real browser proof including selected-variant details previously loading.
Identify whether normal worker reactivation is needed. Preserve selections, provider
configuration, user settings, fresh generations, encrypted backup, archives and S09.
Do not change the tunnel, add sources, delete installation volumes, publish Git,
create goals/branches or redesign architecture. Correct concrete in-scope failures
with root-cause evidence and relevant tests. Do not fabricate live proof from replay.

Use staged-plan-runner for state; prompt-manager only for this revision contract;
root-cause-debugging for concrete failures; backend-quality-gates for Python changes;
contract-impact-analysis for crossed contracts; browser-qa-evidence and the established
playwright-cli mechanic for real UI. Reuse the same independent reviewer if needed.
Routine implementation choices are delegated; record true external blockers and
continue independent authorized work. No new owner acceptance requirement applies.

Write reports/S10-r1.md and sanitized evidence below reports/S10-r1. Report owned
paths, old evidence reused, provider/live/reconnect observations, source changes,
checks, review and residual risks. Accept through the existing ledger_update.py only
when all inherited S10 criteria pass with actual evidence and --checks-passed.
The replacement may complete the pack only when its receipt satisfies original S10;
no successor exists. Preserve the original blocked row exactly and report in Russian.
"""
    with prompt.open("x") as stream:
        stream.write(
            "---\n"
            + json.dumps(
                {
                    "schema_version": "stage-prompt/v1",
                    "prompt_pack_execution": pack.data["prompt_pack_execution"],
                    "stage_contract": contract,
                },
                indent=2,
            )
            + "\n---\n"
            + prose
        )
        stream.flush()
        os.fsync(stream.fileno())
    row = {
        "contract": contract,
        "status": "pending",
        "execution_allowed": True,
        "current_authority": True,
        "executor_claim": None,
        "claimed_at": None,
        "transition_receipt": None,
        "decision_packet": None,
        "replaces_stage": "S10",
        "repair_evidence_ref": "reports/S10-r1/provider-repair.json",
    }
    pack.data["stages"].append(row)
    pack.rows["S10-r1"] = row
    pack.entry("S10-r1")
    now = updater.timestamp()
    executor = "s10-r1-20261014-01a13933-live"
    row.update(status="in_progress", executor_claim=executor, claimed_at=now)
    pack.data.update(ledger_status="active", current_stage="S10-r1")
    pack.entry("S10-r1", executor=executor)
    updater.require(pack.data["stages"][:-1] == old_rows, "Old rows changed")
    for r in old_rows:
        if r["status"] == "accepted":
            pack.receipt(pack.path(r["transition_receipt"]), successor=False)
    pack.data.setdefault("execution_history", []).append(
        {
            "at": now,
            "action": "claim-revisioned-replacement",
            "stage": "S10-r1",
            "executor": executor,
            "receipt": None,
            "reason": "Owner requested retry after tunnel change; four real provider HTTP200 responses repair prior access blocker",
            "evidence": "reports/S10-r1/provider-repair.json",
        }
    )
    updater.require(updater.LEDGER.read_text() == before, "Concurrent ledger change")
    for p, digest in preserved.items():
        updater.require(module.digest(base / p) == digest, "Bound evidence changed")
    updater.write_ledger(before, pack.data, fd)
    checked = module.Pack(root, updater.LEDGER)
    checked.entry("S10-r1", executor=executor)
    record = {
        "status": "ok",
        "stage": "S10-r1",
        "executor": executor,
        "ledger_status": "active",
        "preserved_sha256": preserved,
        "mechanism": "existing directory flock, updater binding, entry validation, CAS and fsync write_ledger; updater unchanged",
    }
    with (base / "reports/S10-r1/replacement-claim.json").open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.fsync(fd)
    print(json.dumps({"status": "ok", "stage": "S10-r1", "executor": executor}))
finally:
    os.close(fd)
