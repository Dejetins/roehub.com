---
{
  "schema_version": "stage-prompt/v1",
  "prompt_pack_execution": {
    "plan_doc": "implementation-plan.md",
    "prompt_pack_dir": ".",
    "stage_ledger": "iteration-ledger.md"
  },
  "stage_contract": {
    "id": "S10-r1",
    "title": "Selected-provider live completion after access repair",
    "prompt_path": "10-r1-live-completion.md",
    "report_path": "reports/S10-r1.md",
    "receipt_dir": "receipts/S10-r1",
    "depends_on": [
      "S09"
    ],
    "expected_touches": [
      "../../../../tools/installation",
      "../../../../docs/runbooks",
      "zone: identified local installation config and market-only derived state",
      "zone: final migration evidence"
    ],
    "acceptance_criteria": [
      "Current local installation uses PG-only market storage, preserved accounts/settings/secrets/selections, and correct fresh history/artifact generation.",
      "Real selected history advances and consecutive closed candles arrive after bounded disconnect/reconnect; independent SQL/API observations and event logs agree.",
      "No reachable CH dependency/process requirement remains; final docs, setup/backup guidance, gates and known data disposition are complete."
    ],
    "proof_boundary": "Actual identified local installation delivery and real-provider evidence; no remote deployment or Git publication.",
    "validation": {
      "profile": "prompt-manager/bundled-v1 plus repository boundary-specific gates",
      "checks": [
        "Follow the stage work/acceptance and section6 proof matrix; record actual commands and output paths.",
        "Reread current repository CI routing/package scripts; run only relevant checks, plus changed-boundary real infrastructure/browser/performance evidence.",
        "Run bundled draft/entry and receipt checks for transitions; artifact structure alone is not implementation proof."
      ],
      "requires_user_acceptance": false
    },
    "entry_inputs": [
      {
        "path": "../../../../tools/installation/local_platform.py",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/runbooks/navigator-local-market-data-recovery.md",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/runbooks/storage-bootstrap-and-migrations.md",
        "producer_stage": null
      },
      {
        "path": "implementation-plan.md",
        "producer_stage": null
      },
      {
        "path": "reports/S09.md",
        "producer_stage": "S09"
      },
      {
        "path": "reports/S10.md",
        "producer_stage": null
      },
      {
        "path": "reports/S10-r1/provider-repair.json",
        "producer_stage": null
      }
    ]
  }
}
---

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
