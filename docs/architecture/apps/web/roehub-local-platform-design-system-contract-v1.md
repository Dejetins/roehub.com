# Roehub local-platform design-system contract v1

Этот implementation-independent контракт задаёт проверяемую будущую библиотеку для self-hosted локальной платформы до product review, не создавая артефакт Penpot и не меняя продуктовый интерфейс.

## Статус и граница полномочий

- Статус: `ready_for_product_review`.
- Design-tool amendment: future Roehub library work uses the registered Figma
  workspace from
  [`roehub-figma-design-delivery-standard-v1.md`](../../ui/roehub-figma-design-delivery-standard-v1.md).
  Penpot wording below remains truthful `2026-07-20` historical evidence and
  is not active tool routing.
- Historical contract evidence: future implementation authority, including the
  six-theme target, is superseded by
  [the Linear-workspace UI transition specification](../../../../.codex/delivery/specs/roehub-linear-workspace-ui-transition.md).
  The JSON companions remain retained evidence and are not runtime targets.
- Scope: `self_hosted_local_platform_only`.
- Владелец принятия: product owner. Только он может перевести этот документ и
  его JSON-компаньоны в принятое состояние.
- Это целевой контракт, а не доказательство существования токенов, компонентов,
  тем, ECharts, маршрутов или экранов в runtime.
- Канонические маршруты, server capabilities, роли `owner`, `admin`,
  `operator`, `trader`, `viewer` и overlay `installation_owner` остаются во
  [входном access/route contract](roehub-local-platform-access-and-route-contract-v1.json).
  Дизайн только отображает уже возвращённое сервером состояние; он не принимает
  решение о доступе.

## Цель, источники и не-цели

Цель — сделать последующее product review и реализацию детерминированными:
одни и те же screen ID, состояния, темы, размеры и design-to-code имена должны
давать одну библиотечную композицию без создания новых продуктовых решений.

Источники истины:

- [accepted local information architecture](roehub-local-platform-information-architecture-v1.md);
- [accepted screen registry](roehub-local-platform-screen-registry-v1.json);
- [accepted access and route contract](roehub-local-platform-access-and-route-contract-v1.json);
- [product requirements](../../platform/roehub-product-transformation-requirements-v1.md);
- [UI design and delivery architecture](roehub-ui-design-and-delivery-architecture-v1.md).

Не входят: работа с Penpot, public-site `roehub.com`, телефонный размер `390`,
SSR/React/API/CSS-код, выбор новых маршрутов, ролей, capabilities, copy,
производственных данных или разрешение на реализацию.

Текущий `apps/web/` остаётся наблюдаемым SSR-фактом, а
`prototypes/roehub-v2/` — historical evidence only. Ни один из них не является
источником целевого маршрута, мобильной цели или полномочия.

## Машиночитаемые компаньоны

- [token contract](roehub-local-platform-design-token-contract-v1.json) —
  значения namespaces, шесть тем, типографика, responsive grid и проверяемые
  accessibility targets;
- [component registry](roehub-local-platform-component-registry-v1.json) —
  семейства, варианты, композиции всех 29 visual screen IDs и будущие package
  bindings.

`screen_compositions.required_states` в registry намеренно повторяет ровно
accepted `required_states`; non-visual contracts и historical prototype явно
исключены, а не получают вымышленные UI-компоненты.

## Решение 1 — темы, tokens и responsive baseline

Поддерживаются только следующие theme IDs и размеры local platform:

| Contract | Exact IDs / widths | Правило |
|---|---|---|
| Themes | `abyss`, `graphite`, `slate`, `frost`, `paper`, `sand` | Тема меняет semantic colour values, но не component IDs, routes или capabilities. `frost` — непрозрачная светлая тема, не glass effect. |
| Widths | `820`, `1024`, `1440` | 820 — compact shell; 1024 — compact grouped navigation; 1440 — expanded navigation допустима. Содержание и доступ не меняются. |
| Exclusion | `390` | Нет mobile navigation, boards или mobile-only variants local platform. |

Неймспейсы токенов неизменны: `rh.color.*`, `rh.space.*`, `rh.type.*`,
`rh.grid.*`, `rh.density.*`, `rh.radius.*`, `rh.elevation.*`,
`rh.motion.*`, `rh.focus.*`, `rh.icon.*`. Base primitives не используются
непосредственно в screen composition: компоненты принимают только semantic
aliases. Полные значения и минимальные contrast pairs хранятся в token JSON.

Типографика использует system sans для UI и табличных данных, mono только для
кодовых значений, ID, keyboard shortcuts и чисел при необходимости выравнивания.
Шкала: `display`, `title`, `section`, `body`, `label`, `meta`, `mono`; каждый
уровень определён в JSON в rem, line-height и weight. Плотность — preference
`comfortable`/`compact`, одинаковая для всех размеров: она меняет rows и gaps,
но не touch/keyboard target, порядок tab или видимость серверного состояния.

Grid uses a minimum `0` width for content columns, explicit table overflow
container and 12-column 1440 / 8-column 1024 / 4-column 820 templates. Поэтому
никакая карточка не создаёт новую responsive breakpoint или горизонтальный scroll
document; прокрутка разрешена только у явно названных data-grid containers.

Радиусы сдержанные, непрозрачные surfaces и тонкие borders поддерживают calm
functional language. Motion is opt-in feedback, not state evidence: transition
timing never fabricates progress; `prefers-reduced-motion: reduce` removes
non-essential transition and animation while preserving visible completion,
error and focus changes.

## Решение 2 — семейства и состояния компонентов

Registry определяет общие primitives (`rh.action`, `rh.field`, `rh.status`,
`rh.feedback`, `rh.data-table`, `rh.dialog`, `rh.shell`, `rh.chart`,
`rh.progress`) и local-platform patterns (`rh.catalog`, `rh.preflight`,
`rh.runtime-control`, `rh.service-health`, `rh.connection`, `rh.recovery`).
Композиция может использовать только перечисленные family IDs и variants.

Обязательные общие состояния реализуются как variants/patterns, не как новые
маршруты:

| Состояние | Визуальный контракт | Безопасное действие |
|---|---|---|
| loading / cursor loading | `aria-busy`, shape-safe skeleton, сохранённый layout | Нет подмены stale data как fresh. |
| empty / empty selection | причина, текущий scope и безопасный next action | CTA только после server-projected eligibility. |
| error / unavailable | code/category, retry boundary, timestamp | Retry не повторяет unknown mutation без reconciliation. |
| stale / degraded / coverage unknown | freshness/source/impact рядом с данными | Не скрывает последнюю известную дату и источник. |
| forbidden / server filtered | явное `403` или permission pattern, без утечки скрытых data | UI не выводит capability и не эмулирует grant. |
| destructive confirmation / recent auth | summary, consequence, irreversible marker, server gate pending | Confirmation не заменяет server authorization or recent auth. |
| recovery / operation unknown | known facts, reconciliation and safe recovery link | Нельзя показывать success или 100% без terminal server result. |
| completed / failed / cancelled | terminal status and duration/history | Active progress bar is removed after terminal state. |

Каждый destructive, promotion, rollback, installation or recovery pattern
requires a named confirmation variant; `rh.dialog.confirmation` is presentation
only and cannot acquire authority. Secret-bearing values are never valid
component props, cell content, chart tooltips or accessibility text.

## Решение 3 — Apache ECharts boundary и data alternative

`rh.chart` is the only future design-to-code boundary for new product charts.
It accepts declarative `ChartSpec/v1`, not an unrestricted ECharts option:

```text
ChartSpec/v1 = chart_id, kind, title_key, description_key, units, timezone,
source, freshness, series[], axes[], legend, interaction, renderer,
table_alternative, accessibility_summary
```

- Permitted `kind`: `timeseries`, `candlestick`, `bar`, `area`, `scatter`,
  `distribution`, `heatmap`, `status_timeline`, `comparison`.
- Permitted `renderer`: `canvas` or `svg`, selected by a bounded wrapper policy;
  chart data and point count drive a documented future benchmark decision, not
  an unbounded consumer option.
- `series`, `axes`, legend and tooltip content use schema-checked fields and
  localization IDs. Tooltip is text-only; no raw HTML, raw JavaScript callback,
  executable formatter, plugin, `graphic`, arbitrary dataset transform,
  arbitrary renderer or forwarded ECharts option is allowed.
- Chart adapter receives a redacted, typed projection. It has no secrets,
  credential fields, arbitrary URLs or client capability data.
- Every rendered chart exposes title, units, timezone, source and freshness,
  an accessible summary and a keyboard-reachable `rh.data-table` alternative
  with the same filtered/aggregated projection. The table alternative is not a
  download claim and retains server filtering.
- Live charts visibly distinguish update timestamp, stale/degraded source,
  reconnecting and unavailable states. Motion is disabled by reduced-motion;
  updates do not silently imply trade or execution success.

This selects Apache ECharts only for future product implementation. It neither
migrates the observed chart renderer nor asserts performance; Canvas/SVG proof
requires a later measured implementation ticket.

## Решение 4 — настоящий progress, queue и ETA

`rh.progress.job` receives typed server facts:
`job_state`, `completed_units`, `total_units`, `queued_at`, `started_at`,
`observed_at`, `queue_position`, `queue_eta`, `execution_eta`,
`eta_confidence`, `terminal_at`, `failure_category`, `cancellation_reason`.

| Job state | Visible progress / ETA |
|---|---|
| `queued` | Queue position/wait are separate from execution. Show queue ETA only with `high` or `medium` confidence. |
| `running` | `completed_units / total_units` is the only percentage source; show execution ETA only with sufficient confidence and a last-observed timestamp. |
| `insufficient_confidence` / `eta_unavailable` | State the reason and next refresh/reconciliation path; do not estimate decoratively. |
| `materializing` | Execution is terminal only when server says so; materialization has its own labelled stage rather than false 100%. |
| `completed`, `failed`, `cancelled` | Hide active bar, retain terminal status, duration and source of result. |
| `operation_unknown` | Show neither success nor 100%; offer reconciliation before retry or duplicate submission. |

`eta_confidence` values are `high`, `medium`, `low`, `insufficient`. `low` and
`insufficient` suppress a duration. The renderer must preserve queue ETA and
execution ETA as different fields, recalibrate from server measurements, and
announce meaningful changes via a polite status region without repetitive
screen-reader noise.

## Accessibility, localization and input invariants

- WCAG 2.2 AA contrast target is 4.5:1 for normal text/interactive text and
  3:1 for non-text focus indicators. Token JSON records machine-checked normal
  text, accent and focus pairs for every theme.
- Keyboard uses native controls first. Menus, listboxes, tabs, dialogs and
  data-grid patterns have documented roving/focus-trap/escape/return-focus
  behaviour in the component registry. Focus never disappears on async change.
- Focus is a 2px semantic ring with 2px offset, never colour-only status. Error,
  warning, live/paper and permission states have textual labels.
- `ru` and `en` content use localization keys, ICU/plural-safe formatting, local
  date/time and units. Charts and tables include timezone and locale-aware
  formatting without changing semantic value.
- At 200% browser zoom, grid reflows inside the same 820/1024/1440 contract;
  table containers may scroll horizontally with a visible keyboard focus and
  labels. Reduced motion does not suppress status, focus or table alternative.

## Future library, versioning and package mapping

No Penpot resource has been read or created. If and only if this contract later
receives product approval, the future file name is `Roehub — Design System` and
its proposed page hierarchy is:

```text
00 Cover & release notes
01 Foundations / colour / type / spacing / density / iconography
02 Themes / abyss / graphite / slate / frost / paper / sand
03 Components / actions / fields / navigation / data / feedback / overlays
04 Charts / ChartSpec patterns / table alternatives / freshness
05 Patterns / jobs / queue-and-ETA / recovery / authorization feedback
06 Accessibility / keyboard / focus / reduced motion / localization
07 Handoff / export manifest / deprecated aliases
```

Names use `Roehub/<Family>/<Variant>` and properties use stable lower camel-case
values from the registry (`state`, `size`, `density`, `appearance`, `intent`,
`interactive`). A future export manifest may map only approved library components
and contract version; it must not contain Penpot IDs in this ticket or assert a
published library. Breaking component/schema changes require a new contract
version, explicit deprecated alias and consumer migration note.

| Future package | Owns | Stable mapping |
|---|---|---|
| `@roehub/tokens` | token JSON and generated platform aliases | `rh.*` token namespaces |
| `@roehub/ui` | primitives and local compositions | `rh.*` component family IDs |
| `@roehub/charts` | typed `ChartSpec/v1` wrapper and table alternative | `rh.chart`, `rh.data-table` |
| `@roehub/localization` | locale keys and number/date semantics | `label_key`, `description_key`, text templates |
| `@roehub/web-contracts` | transport-safe view models | progress, freshness, permission and chart projections |

`apps/platform-web` and future `apps/site` may depend on explicit shared
packages but must not import each other. This mapping does not create packages
or permit public-site composition.

## Compatibility, migration and rollback

Classification: `compatible-change` for documentation and future names only;
`none` for current product routes, roles, data, packages and runtime. Adoption
is additive: implementation consumes a versioned manifest after product
approval, retains the current UI until a separately authorized vertical slice
passes its contract/browser/accessibility proof, and can roll back by removing
that consumer binding. There is no persisted token migration, route migration,
or fallback that weakens server denial.

## Proof boundary and review checklist

This ticket proves static contract consistency only. Focused tests verify:

- exact six theme IDs, 820/1024/1440 widths and 390 exclusion;
- all 29 visual local screen IDs, with exact accepted state lists and component
  mappings; six non-visual/historical registry entries remain excluded;
- safe ECharts restrictions, progress/ETA state distinctions, contrast pairs,
  accessibility and package binding declarations;
- local Markdown links, architecture index, project map and `git diff --check`.

It does not prove runtime CSS, ECharts rendering, browser accessibility,
server authorization, performance, Penpot content, release or deployment.

## Риски и следующий разрешённый шаг

Остаточные риски: product owner может reject visual values or component anatomy;
future server projections may lack typed freshness/progress/chart fields; chart
renderer selection needs a measured workload; and no runtime evidence exists.
Следующий разрешённый шаг — product-owner review этого contract и обоих JSON
companions. Даже принятие ticket не разрешает Penpot или Web implementation;
они требуют отдельной готовой задачи и явной authority.

## Как проверить

```bash
python -m pytest -q tests/unit/docs/test_roehub_local_platform_design_system_contract.py
python -m tools.docs.generate_docs_index --check
python -m tools.docs.generate_project_map --check
git diff --check
```
