# Post-transition validation notes

S09 is accepted by the canonical updater. Receipt validation passes and all 537 indexed pre-acceptance evidence files still match their hashes. S10 is pending and eligible, not executed.

The final read-only `--check draft` invocation was inapplicable to this already active ledger and returned “Draft check cannot reclassify an activated ledger”. It performed no mutation. The appropriate own-claim `--check entry` passed before acceptance, and the updater independently validated transition preconditions and next-stage eligibility. The failed draft-mode invocation is retained in the bound evidence index; it is not represented as a pass.
