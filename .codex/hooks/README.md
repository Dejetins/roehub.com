# Roehub Codex hooks

The repository keeps three narrow guards:

| Guard | Event | Purpose |
| --- | --- | --- |
| `secret_redaction_guard` | pre/post-tool | Detect obvious raw secret exposure. |
| `command_safety_guard` | pre-tool | Block deterministic destructive shell commands. |
| `scoped_git_staging_guard` | pre-tool | Block broad or implicit Git staging. |

Only `PreToolUse` and `PostToolUse` are registered. `Bash` is the native
unified-exec matcher. Stop events do not infer edits, authority or review
completion from final-answer wording. Russian reports and the selected review
policy belong to `AGENTS.md`; no machine receipt or answer rewrite is required.

Validate registration and behavior with the narrow CI job:

```bash
python -m unittest tools.ci.test_agent_policy -v
python .codex/hooks/tests/run_tests.py
```

Registration validation checks the repository's supported hook configuration,
not every possible host schema. Hooks cannot undo completed actions, prove a
review occurred, or cover every shell expression. They are not a security
boundary. These fixtures do not establish native runtime trust/activation.
