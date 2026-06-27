# Web Execution Telegram Notifications v1 - Main Execution Contract

This prompt-contract is mandatory context for every future executor prompt in this pack.

## Hard Main-Branch Rule

All implementation, review, documentation, validation and delivery work for the Notifications v1 plan must happen in the normal repository checkout:

`/Users/daniildegtyarev/Projects/roehub.com`

The working branch is:

`main`

Executors must not create a new branch, per-stage branch, sibling worktree, temporary checkout, local coordination folder, or stash-based workflow for this plan unless the user explicitly changes this contract in a later repository commit.

## Required Start Check

Before reading broad context or editing files, every future stage prompt must require:

```bash
git -C /Users/daniildegtyarev/Projects/roehub.com status --short --branch
```

The result must show the current branch as:

`main`

If it does not, the executor must stop and report the blocker unless the user explicitly asks to switch. If unrelated dirty changes are present, the executor must not create a worktree or stash as a workaround; it must either work with an explicitly scoped file list or report the blocker.

## Access And Secret Contract

Executors must never ask the user to paste secrets in chat.

Allowed secret/access sources:

- local env from `$ROEHUB_ENV_FILE`;
- local env from `/Users/daniildegtyarev/.config/roehub/roehub.env`;
- local env from `/etc/roehub/roehub.env`;
- Mac Studio env from `/Users/daniildegtyarev/.config/roehub/roehub.env`;
- existing runtime/service config after redacted presence checks;
- authenticated browser smoke account username `smoke_e2e_keycloak`, with password only from `ROEHUB_SMOKE_E2E_PASSWORD` in the host-local env source above.

Expected Telegram-related keys for implementation and canary stages:

- `ROEHUB_NOTIFICATIONS_TELEGRAM_BOT_TOKEN` - preferred Notifications v1 bot token env key;
- `TELEGRAM_BOT_TOKEN` - compatibility fallback only if the implementation deliberately reuses the existing Strategy bot token source;
- `ROEHUB_NOTIFICATIONS_ADMIN_TELEGRAM_CHAT_ID` - optional canary/admin route bootstrap key; store/use only through redacted checks or persisted admin route migration;
- `ROEHUB_SMOKE_E2E_PASSWORD` - authenticated browser smoke password source, never printed.

Reports may state whether a key is present. They must not print raw values, chat ids, cookies, authorization headers, provider payloads, exchange credentials, DSNs, HMACs, ciphertext, or passphrases.

## User Presence Contract

Most stages must be executable without the user present. The executor must stop only when a real external action is required and no host-local source exists.

User presence is required for:

- creating or rotating the Telegram bot token through BotFather if no host-local token exists;
- setting host-local Telegram token/admin recipient env on Mac Studio if the executor cannot do it through existing deployment/config mechanisms;
- sending `/start <code>` from the test Telegram account during real binding canary;
- confirming receipt of a real Telegram canary message when Stage `09` needs human-visible provider proof;
- approving any expansion beyond the test/smoke account or admin canary recipient.

User presence is not required for:

- schema/domain/port work;
- fake/log provider tests;
- synthetic notification matrix tests;
- stats query fixtures;
- scheduled report fixtures;
- docs, ledgers, prompt updates, unit/integration tests.

Every stage prompt and final report must state `User required before start: ...`.

## Prompt Pack Rule

Every future prompt file for this plan must include this file in its required context:

`.codex/agents/generated/web-execution-telegram-notifications-v1/00-main-and-stage-execution-contract.md`

Every future stage prompt must also include the stage ledger:

`docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md`

## Forbidden Work Patterns

Do not:

- create `codex/*` branches for Notifications v1 stages;
- create `../roehub-*` sibling worktrees for Notifications v1 stages;
- continue work from `codex/backtest-futures-funding-v1`;
- mix backtest-futures edits with Notifications v1 edits;
- create stashes as a routine workflow for this plan;
- report a stage as accepted merely because files exist locally without main/delivery evidence where required;
- leave a future executor to infer branch/access rules from chat history.

## Reporting Requirement

Every stage report must state:

- checkout path used;
- branch used;
- whether unrelated dirty changes were observed;
- whether user presence was required and why;
- which secret/access sources were used, by key name only;
- confirmation that no branch/worktree/stash workflow was created for the stage.
