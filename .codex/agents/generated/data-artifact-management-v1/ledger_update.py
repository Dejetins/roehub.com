#!/usr/bin/env python3
"""Atomic updates for this local sequential pack (Python stdlib, POSIX filesystem).

Original implementation authorized on 2026-10-07; reused by this pack.
The 2026-10-08 orchestration repair adds explicit bound revision reconciliation.
All executors use this command; never edit active ledger state by hand.
An exclusive flock on the stable pack directory covers reread, validation,
claim/transition and fsync+atomic replacement. A durable executor claim outlives
that short lock; it is never stolen or expired automatically. Directory moves
and non-cooperating writers during execution are unsupported. No daemon,
background execution, test execution or product code is installed by this tool.
Receipts bind evidence bytes; the executor must establish actual test outcomes.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.util
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

LEDGER = Path(__file__).resolve().parent / "iteration-ledger.md"
VALIDATOR = Path.home() / ".codex/skills/prompt-manager/scripts/validate_pack.py"
MARKER = "<!-- prompt-pack-ledger:v1 -->"


def timestamp():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validator_module(path):
    spec = importlib.util.spec_from_file_location("pack_validator", path)
    require(spec is not None and spec.loader is not None, "Validator unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_ledger(text, data, directory_fd):
    pattern = re.compile(r"(<!-- prompt-pack-ledger:v1 -->\s*```json\n)(.*?)(\n```)", re.S)
    require(len(pattern.findall(text)) == 1, "Expected one canonical ledger record")
    rendered = pattern.sub(lambda m: m[1] + json.dumps(data, indent=2, ensure_ascii=False) + m[3], text)
    descriptor, temporary = tempfile.mkstemp(prefix=".ledger-", dir=LEDGER.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, LEDGER)
        os.fsync(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)


def receipt(pack, module, stage, evidence, next_id, now):
    contract = pack.rows[stage]["contract"]
    def bind(path):
        path = pack.path(str(path))
        require(path.is_file(), f"Missing evidence: {path}")
        return {"path": os.path.relpath(path, pack.base), "sha256": module.digest(path)}
    next_stage = None
    if next_id:
        other = pack.rows[next_id]["contract"]
        next_stage = {"id": next_id, "prompt": bind(other["prompt_path"]),
                      "stage_contract_sha256": module.contract_digest(other)}
    record = {
        "schema_version": "prompt-pack-receipt/v1", "stage_id": stage,
        "stage_contract_sha256": module.contract_digest(contract), "created_at": now,
        "status": "ready", "plan": bind(pack.triad["plan_doc"]),
        "prompt": bind(contract["prompt_path"]), "report": bind(contract["report_path"]),
        "validation": {"profile": contract["validation"]["profile"], "result": "pass",
                       "evidence": [bind(item) for item in evidence]},
        "user_acceptance": None, "next_stage": next_stage,
        "next_stage_allowed": next_id is not None,
        "handoff_reason": "Required checks passed; next prompt may execute when submitted."
                          if next_id else "All required stages complete; no successor.",
    }
    directory = pack.path(contract["receipt_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"receipt-{uuid4().hex}.md"
    with path.open("x", encoding="utf-8") as stream:
        stream.write("# Stage transition receipt\n\n<!-- prompt-pack-receipt:v1 -->\n```json\n")
        stream.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n```\n")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    pack.receipt(path)
    return os.path.relpath(path, pack.base)


def reconcile(args, module, directory_fd, original_text, pack):
    """Apply an explicit blocked-stage repair without resetting terminal history.

    The repair document binds the old ledger/updater and exact new prompt bytes.
    Only directory-to-file entry corrections on pending rows and a scope-identical
    revision of the blocked stage are supported. The existing lock/CAS/writer is
    used; acceptance still requires the normal executor and receipt gates.
    """
    require(args.repair and args.executor, "Reconciliation requires repair evidence and executor")
    repair_path = pack.path(args.repair)
    repair_bytes = repair_path.read_bytes()
    repair = module.document(repair_path, "<!-- prompt-pack-repair:v1 -->")
    require(repair.get("schema_version") == "prompt-pack-repair/v1", "Unsupported repair schema")
    require(repair.get("ledger_sha256") == module.digest(LEDGER), "Stale repair ledger baseline")
    prior = repair["previous_updater"]
    pack.binding(prior)
    capability = pack.data["claim_capability"]
    require(capability["evidence"]["path"] == "ledger_update.py"
            and prior["sha256"] == capability["evidence"]["sha256"], "Wrong previous updater")
    pack.binding(repair["new_updater"], Path(__file__).resolve())
    require(repair.get("authority") and repair.get("reason"), "Missing repair authority/reason")
    require(pack.data["ledger_status"] == "blocked", "Reconciliation needs a blocked ledger")
    blocked_id = pack.data["current_stage"]
    blocked = pack.rows[blocked_id]
    require(blocked_id == repair["blocked_stage"] and blocked["status"] == "blocked",
            "Repair does not match blocked current stage")
    require(not any(row["status"] in {"in_progress", "needs_input"} for row in pack.rows.values()),
            "Another executor owns a live claim")
    require(not blocked.get("replacement_stage") and not any(
        row.get("replaces_stage") == blocked_id for row in pack.rows.values()),
        "Replacement already exists; explicit reconciliation is required")
    candidate = copy.deepcopy(pack.data)
    rows = {row["contract"]["id"]: row for row in candidate["stages"]}
    bindings = [repair["new_updater"], prior]
    changed = set()
    for change in repair["pending_entry_repairs"]:
        sid = change["stage_id"]
        require(sid not in changed, "Duplicate pending repair")
        changed.add(sid)
        row = rows[sid]
        require(row["status"] == "pending" and row["executor_claim"] is None,
                "Only unclaimed pending entry contracts can be repaired")
        old = row["contract"]
        require(module.contract_digest(old) == change["previous_contract_sha256"],
                "Stale pending contract")
        pack.binding(change["prompt"])
        front = module.document(pack.path(change["prompt"]["path"]), frontmatter=True)
        new = front["stage_contract"]
        expected = copy.deepcopy(old)
        require(new["prompt_path"] != old["prompt_path"], "Preserve the original prompt")
        expected["prompt_path"] = new["prompt_path"]
        require(change["prompt"]["path"] == new["prompt_path"], "Wrong revised prompt binding")
        require(len(new["entry_inputs"]) == len(old["entry_inputs"]), "Entry count changed")
        corrections = 0
        for previous, revised in zip(old["entry_inputs"], new["entry_inputs"]):
            if previous == revised:
                continue
            old_path, new_path = pack.path(previous["path"]), pack.path(revised["path"])
            require(previous["producer_stage"] is None and revised["producer_stage"] is None
                    and old_path.is_dir() and new_path.is_file()
                    and new_path.is_relative_to(old_path), "Only contained directory-to-file repairs allowed")
            corrections += 1
        require(corrections > 0, "No entry correction")
        expected["entry_inputs"] = new["entry_inputs"]
        require(new == expected, "Pending repair changes acceptance or scope")
        row["contract"] = new
        bindings.append(change["prompt"])
    revision = repair["replacement"]
    pack.binding(revision["prompt"])
    contract = module.document(pack.path(revision["prompt"]["path"]), frontmatter=True)["stage_contract"]
    sid = contract["id"]
    require(sid == args.stage and sid not in rows, "Replacement ID must be new and selected")
    require(contract["prompt_path"] == revision["prompt"]["path"], "Wrong replacement prompt")
    expected = copy.deepcopy(blocked["contract"])
    for key in ("id", "title", "prompt_path", "report_path", "receipt_dir"):
        expected[key] = contract[key]
    require(contract == expected, "Replacement must preserve the full blocked-stage contract")
    row = {"contract": contract, "status": "pending", "execution_allowed": True,
           "current_authority": True, "executor_claim": None, "claimed_at": None,
           "transition_receipt": None, "decision_packet": None,
           "replaces_stage": blocked_id, "repair_evidence_ref": args.repair}
    candidate["stages"].append(row)
    rows[blocked_id]["replacement_stage"] = sid
    candidate["claim_capability"]["evidence"] = repair["new_updater"]
    candidate["claim_capability"]["mechanism"] += "; explicit revision reconciliation with bound repair evidence"
    bindings.append(revision["prompt"])

    def validate_candidate():
        # Feed proposed ledger data to a fresh, unchanged validator instance.
        # All prompt, receipt, path and lifecycle checks still read live files.
        checker = validator_module(args.validator)
        read_document = checker.document
        def candidate_document(path, *positional, **keywords):
            if Path(path).resolve() == LEDGER:
                return copy.deepcopy(candidate)
            return read_document(path, *positional, **keywords)
        checker.document = candidate_document
        return checker.Pack(args.root, LEDGER)

    preview = validate_candidate()
    preview.entry(sid)
    now = timestamp()
    row.update(status="in_progress", executor_claim=args.executor, claimed_at=now)
    candidate.update(ledger_status="active", current_stage=sid)
    candidate.setdefault("execution_history", []).append({
        "at": now, "action": "reconcile-and-claim", "stage": sid,
        "replaces_stage": blocked_id, "executor": args.executor,
        "repair_evidence_ref": args.repair, "reason": repair["reason"]})
    candidate.setdefault("execution_repairs", []).append({
        "at": now, "repair_evidence_ref": args.repair,
        "previous_updater": prior, "new_updater": repair["new_updater"],
        "pending_contracts": sorted(changed), "replacement_stage": sid})
    validate_candidate().entry(sid, executor=args.executor)
    require(LEDGER.read_text(encoding="utf-8") == original_text, "Concurrent ledger edit")
    require(repair_path.read_bytes() == repair_bytes, "Concurrent repair evidence edit")
    for binding in bindings:
        pack.binding(binding)
    note = ("\n## Bound execution repair\n\n"
            f"Reconciliation evidence: `{args.repair}`. The original byte-identical "
            "updater description above is provenance; the canonical claim capability "
            "now binds its reviewed reconciliation extension. Terminal history and "
            "prior accepted artifacts are preserved. Exact current prompt paths "
            "and replacement semantics are in the canonical stage rows.\n")
    write_ledger(original_text + note, candidate, directory_fd)
    module.Pack(args.root, LEDGER)
    return {"action": "reconcile-and-claim", "stage": sid, "stage_status": "in_progress",
            "ledger_status": "active", "executor": args.executor, "preserved_blocked_stage": blocked_id}


def execute(args, module, directory_fd):
    text = LEDGER.read_text(encoding="utf-8")
    pack = module.Pack(args.root, LEDGER)
    if args.action == "reconcile":
        return reconcile(args, module, directory_fd, text, pack)
    data = pack.data
    capability = data.get("claim_capability", {})
    require(capability.get("evidence", {}).get("path") == "ledger_update.py",
            "This pack must bind this updater as its claim capability")
    pack.binding(capability["evidence"], Path(__file__).resolve())
    if args.action == "status":
        return {"ledger_status": data["ledger_status"], "current_stage": data["current_stage"],
                "stages": [{"id": sid, "status": row["status"],
                            "execution_allowed": row["execution_allowed"]}
                           for sid, row in pack.rows.items()]}
    require(args.executor and args.executor.strip(), "A unique executor ID is required")
    require(args.stage in pack.rows, "Unknown stage")
    row = pack.rows[args.stage]
    require(data["ledger_status"] not in {"completed", "archived", "superseded", "blocked"},
            "Terminal/blocked ledger requires an explicit revision, never an in-place reset")
    now = timestamp()
    if args.action == "claim":
        require(row["status"] == "pending", "Stage is already claimed/terminal; do not steal it")
        pack.entry(args.stage)
        row.update(status="in_progress", executor_claim=args.executor, claimed_at=now)
        data.update(ledger_status="active", current_stage=args.stage)
    else:
        require(data["current_stage"] == args.stage and row["executor_claim"] == args.executor,
                "Executor does not own the current stage")
        if args.action == "resume":
            require(row["status"] in {"in_progress", "needs_input"}, "Stage is not resumable")
            if row["status"] == "needs_input":
                require(args.resolution and pack.path(args.resolution).is_file(),
                        "Readable evidence of the exact owner decision is required")
                row["decision_packet"]["resolution_evidence"] = args.resolution
            pack.entry(args.stage, executor=args.executor)
            row.update(status="in_progress", execution_allowed=True)
            data["ledger_status"] = "active"
        else:
            require(row["status"] == "in_progress", "Stage must be in progress")
            pack.entry(args.stage, executor=args.executor)
            if args.action == "accept":
                require(args.checks_passed and args.evidence,
                        "Accept requires --checks-passed and actual --evidence files")
                require(not row["contract"]["validation"]["requires_user_acceptance"],
                        "This command cannot bypass a user acceptance requirement")
                remaining = [sid for sid, other in pack.rows.items()
                             if sid != args.stage and other["status"] == "pending"]
                eligible = [sid for sid in remaining if all(
                    pack.satisfied(dep, projected=frozenset({args.stage}))
                    for dep in pack.rows[sid]["contract"]["depends_on"])]
                require(len(eligible) == (1 if remaining else 0), "No unique next stage")
                next_id = eligible[0] if eligible else None
                if next_id:
                    pack.entry(next_id, projected=frozenset({args.stage}), require_allowed=False)
                reference = receipt(pack, module, args.stage, args.evidence, next_id, now)
                row.update(status="accepted", execution_allowed=False, transition_receipt=reference)
                if next_id:
                    pack.rows[next_id]["execution_allowed"] = True
                    pack.entry(next_id)
                else:
                    require(all(pack.satisfied(sid) for sid in pack.rows), "Unsatisfied final obligation")
                    data["ledger_status"] = "completed"
            elif args.action in {"block", "needs-input"}:
                require(args.reason and args.evidence, "A reason and actual evidence are required")
                for item in args.evidence:
                    require(pack.path(item).is_file(), "Missing blocker evidence")
                row.update(execution_allowed=False, reason=args.reason, evidence=args.evidence)
                if args.action == "needs-input":
                    require(args.resume_condition, "Exact resume condition is required")
                    row.update(status="needs_input", decision_packet={
                        "question": args.reason, "resume_condition": args.resume_condition,
                        "resolution_evidence": None})
                    data["ledger_status"] = "awaiting_input"
                else:
                    row["status"] = "blocked"
                    data["ledger_status"] = "blocked"
    data.setdefault("execution_history", []).append({
        "at": now, "action": args.action, "stage": args.stage, "executor": args.executor,
        "receipt": row["transition_receipt"], "reason": args.reason})
    # All preconditions, live bindings and transition proofs were checked while locked.
    # CAS also detects non-cooperating edits made since this transaction's reread.
    require(LEDGER.read_text(encoding="utf-8") == text, "Concurrent non-cooperating ledger edit")
    write_ledger(text, data, directory_fd)
    module.Pack(args.root, LEDGER)
    return {"action": args.action, "stage": args.stage, "stage_status": row["status"],
            "ledger_status": data["ledger_status"], "executor": row["executor_claim"],
            "receipt": row["transition_receipt"],
            "next_allowed": [sid for sid, other in pack.rows.items()
                             if other["status"] == "pending" and other["execution_allowed"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["status", "claim", "resume", "accept", "block", "needs-input", "reconcile"])
    parser.add_argument("--stage")
    parser.add_argument("--executor")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--validator", type=Path, default=VALIDATOR)
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--checks-passed", action="store_true")
    parser.add_argument("--reason")
    parser.add_argument("--resume-condition")
    parser.add_argument("--resolution")
    parser.add_argument("--repair")
    args = parser.parse_args()
    fd = None
    try:
        require(not LEDGER.is_symlink(), "Ledger symlinks are unsupported")
        fd = os.open(LEDGER.parent, os.O_RDONLY)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = execute(args, validator_module(args.validator), fd)
        print(json.dumps({"status": "ok", **result}, ensure_ascii=False))
        return 0
    except Exception as error:
        print(json.dumps({"status": "error", "reason": str(error)}, ensure_ascii=False))
        return 1
    finally:
        if fd is not None:
            os.close(fd)


if __name__ == "__main__":
    sys.exit(main())
