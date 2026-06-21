# Web Execution Telegram Notifications v1 - Branch And Stage Execution Contract

This prompt-contract is mandatory context for every future executor prompt in this pack.

## Hard Branch Rule

All implementation, review, documentation, validation and delivery work for the Notifications v1 plan must happen in the normal repository checkout:

`/Users/daniildegtyarev/Projects/roehub.com`

The only working branch for this plan is:

`codex/web-execution-telegram-notifications-v1`

Executors must not create a new per-stage branch or a sibling worktree for this plan. The branch above is the durable coordination branch for Stage `00` through the end of the Notifications v1 cycle unless the user explicitly changes the branch contract in a later repository commit.

## Required Start Check

Before reading broad context or editing files, every future stage prompt must require:

```bash
git -C /Users/daniildegtyarev/Projects/roehub.com status --short --branch
```

The result must show the current branch as:

`codex/web-execution-telegram-notifications-v1`

If it does not, the executor must switch the same checkout to that branch before continuing. If the checkout has unrelated dirty changes, the executor must preserve them safely and explicitly before switching. Do not discard user or other-agent work.

## Prompt Pack Rule

Every future prompt file for this plan must include this file in its required context:

`.codex/agents/generated/web-execution-telegram-notifications-v1/00-branch-and-stage-execution-contract.md`

Every future stage prompt must also include the stage ledger:

`docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md`

## Forbidden Work Patterns

Do not:

- create `../roehub-*` sibling worktrees for Notifications v1 stages;
- create a new `codex/*` branch per stage;
- continue work from `codex/backtest-futures-funding-v1`;
- mix backtest-futures edits with Notifications v1 edits;
- report a stage as accepted merely because it was committed to a temporary branch or worktree;
- leave a future executor to infer the branch from chat history.

## Allowed Exception

The only exception is an explicit user request in the current turn to change the branch strategy. If that happens, the executor must update this prompt-contract, the architecture plan and the stage ledger in the same commit before doing implementation work.

## Reporting Requirement

Every stage report must state:

- checkout path used;
- branch used;
- whether any unrelated dirty changes were observed;
- how unrelated changes were preserved;
- confirmation that no per-stage branch/worktree was created.
