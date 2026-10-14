# S03 post-transition validation

The atomic updater accepted S03 and allowed S04 with receipt
`receipts/S03/receipt-7c6b7401707e4f629df7b763be16f0b4.md`. This executor stops here.
The linked S03 report/evidence/receipt and all predecessor evidence remain immutable.

Read-only `--check draft` was also attempted against the already active ledger; the
validator correctly refused to reclassify it (`Draft check cannot reclassify an
activated ledger`). This was an inapplicable validation mode, not a runnable-stage
failure or a pass. Entry/receipt checks are the applicable transition gates; the
updater executed them under its lock. The refused output remains `S03/draft-check.log`.
No state was reset or manually edited to bypass the refusal.

After acceptance, generated project-map navigation is refreshed only to register the
new immutable receipt and this note. The sealed pretransition owned-file manifest
remains historical for those two generated navigation files; runtime/schema/test hashes
must still match it. The post-transition navigation hashes are in
`S03/post-transition-navigation.json`. `S03/receipt-check.json`,
`S03/next-entry-check.json` and `S03/post-transition-map-check.log` record final live
checks. No S04 claim or execution is authorized for this executor.

The first standalone receipt-check command mistakenly supplied `--stage`, which this
CLI accepts only for entry. It was rejected before validation; its output is preserved
as `S03/receipt-check-attempt1.json`. Repeating the read-only command without that flag
produced the actual receipt-check result. No report, bound evidence or receipt was
rewritten; the updater had already validated the receipt during the atomic acceptance.
