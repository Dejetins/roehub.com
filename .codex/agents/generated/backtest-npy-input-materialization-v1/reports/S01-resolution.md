# S01 gate resolution — 2026-10-07

Owner follow-up, verbatim:

> Я не понимаю, что тут в него святого. My right. А в чём проблема его исправить-то? Или это запрещено или что?

In the context of the preceding choice between a scoped-gate exception and
minimal repairs, this steers the work toward fixing the type errors rather than
waiving the global gate. The executor proceeds with minimal test typing repairs,
preserving existing market-data behavior and all other foreign changes. No
exclusion, diagnostic suppression or global-pyright waiver is authorized or used.
Accept S01 only after the global gate and relevant tests actually pass.
