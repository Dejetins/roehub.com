# Skill diagnostic profiles

`skill_audit.py` emits `agent-skills-structure-v2`. Its selected standard profile
checks YAML parsing/required strings, Unicode name syntax (1–64 lowercase
alphanumeric characters, no leading/trailing or consecutive hyphens) and the
Agent Skills description limit of
1,024 characters. Standard-invalid v2 metadata rejects a candidate even when
its diagnostic score increases. It is a bounded lint, not a full cross-host certification.
Names are checked as parsed, without trimming quoted whitespace or normalizing
characters. Unicode letters/numbers follow Python’s `str.isalnum()`; cased
characters must already be lowercase. See the [name-field specification](https://agentskills.io/specification#name-field).
Host metadata remains supported and does not grant tool permissions.

Findings carry `kind`: `standard`, `advisory` or `local_policy`. Brevity, headings
and workflow recipes do not affect acceptance. Remaining style hints have zero
points; secret-guidance matches remain local policy diagnostics requiring review.
`standard_status` is separate from those heuristic safety findings. A pass does
not prove task success, safety or routing quality.

Existing numeric metric keys and JSON list shape remain for callers. The score
is a static diagnostic aggregate, not model quality. `audit_profile` is additive
in JSON/TSV. Stored results without it mean `legacy-shape-v1`; legacy-to-legacy
comparisons retain their recorded scale. Cross-profile or unknown-profile
comparisons fail explicitly: re-audit both saved inputs with the same profile.
Historical files, scores, IDs and receipts are never rewritten in place.

The all-skills comparator has no behavioral inputs. Its historical
`task_contract_preserved` field remains a v1 diagnostic-status alias; v2 emits
null and exposes `diagnostic_status_preserved`. Its candidate means only a lint
improvement. Task-score and pairwise comparisons still require their separate
recorded outcomes; lint gains alone cannot establish behavioral improvement.
