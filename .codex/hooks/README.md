# Roehub Codex hooks

The repository keeps four narrow guardrails. They complement `AGENTS.md` and
platform skills; they do not route work, create delivery artifacts, or decide
what evidence a task needs.

| Guardrail | Event | Purpose |
| --- | --- | --- |
| `secret_redaction_guard` | pre/post-tool | Prevent obvious raw secret exposure. |
| `command_safety_guard` | pre-tool | Block deterministic destructive shell commands. |
| `scoped_git_staging_guard` | pre-tool | Block broad or implicit Git staging. |
| `russian_final_answer_guard` + `cold_head_gate` | stop | Keep Russian final reports and a readable review receipt for changed policy, architecture, or reusable prompt artifacts. |

`PreToolUse`, `PostToolUse`, and `Stop` are the only hooked events. The Stop
checks are intentionally retained: they do not prescribe a workflow, but make
the final report reviewable.

Former workflow-specific validators and their fixtures were removed. Recreate a
guard only for a current, narrowly justified invariant.

Validate the active router after changing it:

```bash
/usr/bin/python3 .codex/hooks/tests/run_tests.py
```

Hooks are guardrails, not a security boundary. They cannot undo completed tool
actions or prove that a claimed review took place.
