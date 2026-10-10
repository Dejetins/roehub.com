# Bound pack entry repair — 2026-10-08

Prepared by the authorized orchestrator using prompt-manager; applied only by staged-plan-runner through the existing updater's reviewed reconcile command. This document is immutable execution evidence, not another control ledger.

The S03 blocker is recorded in S03-acceptance-blocker.md. S01/S02 accepted contracts, plan, original prompts, reports and receipts remain byte-identical. S03 stays blocked with its claim/history/evidence; S03-R2 preserves every original acceptance/proof requirement and obtains its own report/receipt. S04 and S06 keep their IDs, scope, dependencies and acceptance, but get new prompt paths with a contained concrete file replacing the invalid directory input. All other pending inputs were inspected: no other existing input-directory defects found.

The original updater and tests are preserved in pack-repair-before-*.txt. The extension uses the same directory flock, CAS, fsync and atomic replacement. It rejects stale ledger/updater/prompt bindings, existing replacements/live claims, changed acceptance and non-directory entry edits; it validates the full proposed record and dependency receipts before its single write. Ordinary claim/resume still cannot reset a blocked stage.

Required repair checks: test_ledger_update.py disposable suite; independent pack-repair review; exact live source binding validation for S03; immutable S01/S02 dependency receipt checks. See pack-entry-repair-review.md and pack-entry-repair-checks.md for observed results. Product proof is reused only while current; structural checks are not product tests.

<!-- prompt-pack-repair:v1 -->
```json
{
  "schema_version": "prompt-pack-repair/v1",
  "blocked_stage": "S03",
  "ledger_sha256": "d9ff9537501207ce2eb9a7c5439db95014a2d62d7b6fbdfb1bc18fea53fb20d6",
  "previous_updater": {
    "path": "reports/pack-repair-before-ledger_update.py.txt",
    "sha256": "dbae4cb40c4b69572bd275e2350fa1af79489b1d7a7df4fe3654d40429130550"
  },
  "new_updater": {
    "path": "ledger_update.py",
    "sha256": "3e4c07659dc0c5382d908360795fd730b33b5dd2cefaf9417b73e0f6ed1965e6"
  },
  "authority": "Direct user /goal authorization to execute S01–S09 sequentially in separate chats, orchestrate and correct as necessary; continuing user instruction: продолжай задачу. This is a bounded executor-pack defect repair, not a product scope change.",
  "reason": "S03 product proof is complete but successor S04 declares a directory where the validator requires a file; S06 has the same unclaimed serialization defect. Preserve terminal S03 through S03-R2 and correct only those pending entry paths.",
  "pending_entry_repairs": [
    {
      "stage_id": "S04",
      "previous_contract_sha256": "4baaec3dd7f5f2e531dfffe662b612d91270e5b6be72d58364c36bf71ed7446f",
      "prompt": {
        "path": "04-indicator-profiles-r2.md",
        "sha256": "8dda8f50e462ff5646b8094bb04e4f8ed603d32bbe3f00db97440fefae2641c3"
      }
    },
    {
      "stage_id": "S06",
      "previous_contract_sha256": "0fbb670660e65761bc8119374f34c0f83777b954df5c98a5391ba639fbdb4f37",
      "prompt": {
        "path": "06-api-and-storage-boundary-r2.md",
        "sha256": "1d11375c6aaf953cb6dbb9fcb9e2cd3cb6624026446aa450169224d2b79dd6c6"
      }
    }
  ],
  "replacement": {
    "prompt": {
      "path": "03-automatic-base-npy-r2.md",
      "sha256": "bd59feb636e22d7a6be142a3240d92921d2f54bc7ba21c21087711665fb22370"
    }
  }
}
```
