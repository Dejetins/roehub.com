# Repository Agent Instructions

This root file exists for standard `AGENTS.md` discovery.

In this workspace, the detailed repository engineering contract lives in `.codex/AGENTS.md`.

For work in this repository:
- read and follow `.codex/AGENTS.md` after this file when it is present;
- treat `.codex/PLANS.md` as long-horizon planning state, not as default startup context;
- if `.codex/AGENTS.md` cannot be read, fall back to this file plus the global baseline and state that limitation.

Always preserve these repository invariants:
- make minimal, task-bounded changes;
- do not silently change external contracts or persisted behavior;
- keep verification proportional to the change;
- distinguish implemented facts from proposals, inference, and unverified assumptions;
- report skipped checks and residual risks.
