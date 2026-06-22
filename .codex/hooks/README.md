# Roehub Codex Hook Policy

This directory contains repo-local Codex hook scripts for Roehub.

The active hook source is:

- `.codex/hooks.json`

The executable router is:

- `.codex/hooks/roehub_hook_router.py`

## Intent

Hooks mechanically enforce or remind the agent about recurring Roehub workflow
requirements. They do not replace `AGENTS.md`, skills, or cold-head review.

Responsibility split:

- `AGENTS.md`: normative repository contract.
- Skills: expert workflow and judgment.
- Cold-head reviewer: independent read-only artifact review.
- `.codex/rules/*.rules`: command-prefix execpolicy layer.
- Hooks: lifecycle checks around prompt, tool, and stop events.

## Enforcement Levels

| Level | Meaning | Default behavior |
| --- | --- | --- |
| `FATAL_BLOCK` | Deterministic unsafe action | Block before execution where possible |
| `CONTINUE_BEFORE_FINAL` | Missing required workflow evidence | Continue the turn or replace tool result with feedback |
| `WARN_WITH_CONTEXT` | Judgment-adjacent gap | Add model-visible context or UI warning |
| `OBSERVE` | Diagnostic signal | No persistence unless explicitly enabled |

Hard blocks are reserved for deterministic violations such as raw secrets,
wrong Mac Studio git path, destructive commands, and floating Playwright CLI.
Command-execution hard blocks apply only to real `Bash` tool events. File edits
may contain negative examples in docs, fixtures, and rules without being treated
as attempted shell execution.

## Validators

- `secret_redaction_guard.py`: blocks raw secret-like values.
- `command_safety_guard.py`: blocks obvious destructive commands.
- `branch_workflow_guard.py`: blocks unapproved branch creation, stage-specific prompt-pack branches, and unapproved `git worktree add` folder creation.
- `macstudio_path_guard.py`: blocks git operations inside `/opt/roehub/app`.
- `remote_payload_quoting_guard.py`: blocks inline SSH + ClickHouse SQL quoting and warns on inline SSH JSON payloads.
- `playwright_wrapper_guard.py`: blocks floating Playwright CLI invocations.
- `prompt_pack_stage_ledger_linter.py`: requires generated prompts to carry ledger and manifest anchors.
- `prompt_pack_branch_policy_guard.py`: requires generated prompt packs that mention branch work to use one shared branch policy and rejects unrequested worktree/folder instructions.
- `docs_index_drift_guard.py`: reminds agents to refresh/check generated docs indexes after docs edits.
- `architecture_doc_linter.py`: warns about missing architecture documentation anchors.
- `validation_depth_linter.py`: flags tests-only validation for runtime/integration surfaces.
- `runtime_proof_boundary_guard.py`: requires Mac Studio prompt/docs wording to distinguish pre-main host/read-only checks from post-main changed-code production runtime proof.
- `performance_evidence_guard.py`: requires comparable baseline/candidate evidence for performance claims.
- `cold_head_gate.py`: requires cold-head evidence before finalizing architecture or prompt artifacts.
- `skill_lint_guard.py`: warns about missing `SKILL.md` frontmatter.

## Event Output Contract

- `PreToolUse`: deterministic `FATAL_BLOCK` returns `permissionDecision: "deny"`.
- `PermissionRequest`: deterministic `FATAL_BLOCK` returns `decision.behavior: "deny"`.
- `PostToolUse`: blocking findings return `decision: "block"` as feedback. This does not undo side effects.
- `UserPromptSubmit`: secret findings block prompt submission; warnings add additional context.
- `Stop`: required final gates return `decision: "block"` to create one continuation prompt.

`Stop` validators must avoid loops. The router suppresses repeated continuation
when `stop_hook_active` is already true or when the same reason marker is already
present in the last assistant message.

## Secrets And Logging

Hook payloads may contain commands, output, cookies, provider payloads, or raw
credentials. Do not persist payloads by default.

Optional diagnostic logging is disabled unless `ROEHUB_HOOK_OBSERVE_LOG` is set.
When enabled, the router writes only event name, cwd, and finding text; it does
not persist raw hook payloads.

Summarize an observe log without reading raw payloads:

```bash
/usr/bin/python3 .codex/hooks/hook_observe_report.py "$ROEHUB_HOOK_OBSERVE_LOG"
```

## Validation

Run fixture tests before trusting changed hooks:

```bash
/usr/bin/python3 .codex/hooks/tests/run_tests.py
```

Run execpolicy checks for rules when Codex CLI is available:

```bash
codex execpolicy check --pretty --rules .codex/rules/roehub.rules -- git -C /opt/roehub/app status
codex execpolicy check --pretty --rules .codex/rules/roehub.rules -- git -C /Users/daniildegtyarev/Projects/roehub.com status
```

The hook router allows top-level `codex execpolicy check ... -- <command>` dry
runs so negative rule examples can be tested without the PreToolUse command
guards blocking the dry run itself.

After edits, review and trust hook definitions through `/hooks` in Codex.

## Known Limits

- Codex hooks are guardrails, not a complete security boundary.
- `PostToolUse` cannot undo a command or edit that already ran.
- Some Codex tool paths may not emit hook events consistently.
- Repo-local hooks run only when the project `.codex/` layer is trusted.
- `.codex/rules/*.rules` are an experimental execpolicy layer and should be tested with `codex execpolicy check`.
- Hooks cannot prove that a named cold-head review actually happened; they only
  prevent finalizing relevant artifacts without a structured reported cold-head
  receipt that names scope, instructions, verdict, fixes, local follow-up, and
  residual risks.
