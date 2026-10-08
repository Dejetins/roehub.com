#!/usr/bin/env python3
"""Atomic updates for this local sequential pack (Python stdlib, POSIX filesystem).

Owner-authorized on 2026-10-07 to remove the unavailable-updater blocker.
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


def execute(args, module, directory_fd):
    text = LEDGER.read_text(encoding="utf-8")
    pack = module.Pack(args.root, LEDGER)
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
    parser.add_argument("action", choices=["status", "claim", "resume", "accept", "block", "needs-input"])
    parser.add_argument("--stage")
    parser.add_argument("--executor")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--validator", type=Path, default=VALIDATOR)
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--checks-passed", action="store_true")
    parser.add_argument("--reason")
    parser.add_argument("--resume-condition")
    parser.add_argument("--resolution")
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
