# Backtest Compute Acceleration Negative Results v1

Документ фиксирует методы ускорения backtest compute, которые уже проверялись
или были явно отклонены и не должны повторно попадать в production-план как
доказанное ускорение без нового benchmark-gated плана.

Актуально на 2026-06-14. Канонический журнал выполнения:
`docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md`.
Каноническая evidence-папка:
`docs/architecture/backtest/benchmark_iterations/`.

## Правило Повторного Рассмотрения

Метод из этого документа можно возвращать в план только если новый prompt/stage
фиксирует:

- новый доминирующий cost center, которого не было в старом замере;
- сопоставимый baseline на Mac Studio через тот же API-runner или явно
  обоснованную более узкую surface;
- top-N identity/order или explicit bounded metric diff для той же request
  semantics;
- service wall без регресса, а не только локальный timer win;
- memory cleanup и отсутствие изменения public API, request hash, variant hash,
  cache identity, persisted schema и canonical artifact publisher без отдельного
  одобрения.

## Краткая Карта

| Метод | Статус | Причина не считать ускорением |
|---|---|---|
| Notebook top-k no-risk exact record | failure record | `exact_scoring` проходил, но `heap_update`, `top_result_proxy_fill`, strict hash и benchmark boundary были некорректны |
| Quality gate / ranking-only exact с отдельным `confirm_prefilter` | rejected | локальный no-risk exact win был съеден вторым проходом; service wall ухудшился, `long_short_reversal` дал `top_count=0` |
| Signal row dedup как production pruning | no-op on accepted rows | Stage 02 нашел `0/36` duplicate rows на accepted arity-6 rows |
| `matrix_bitset_no_risk_v1` на `none/arity_2/long_only` | accepted_for_learning only | tiny-row case дал speed fail, fixed overhead больше выигрыша |
| Consensus signature cache | rejected | cache hit-rate был, но exact scoring и service wall сильно регрессировали |
| Sidecar `.npy` bitset load | accepted_for_learning only | sidecar load медленнее runtime pack и добавляет небольшой overhead |
| TP/SL selected-cell shadow | accepted_for_learning only | это parity/layout shadow для 8x8 cells, не production acceleration |
| TP/SL full-grid block shape `16 x 16` | diagnostic fail | parity прошла, но speed gate failed; accepted shape была `64 x 64` |
| High-arity min-trade pruning Python traversal | accepted_for_learning only | exact-safe rule полезен, но traversal cost съел потенциальный выигрыш |
| Lazy detail sparse trade tape reuse | rejected | latency delta меньше 1%, недостаточно для runtime change |
| TP/SL block-shape production gate | rejected and removed | ни одна форма block scoring не дала accepted service-wall win на обеих TP/SL heavy rows |
| TP/SL selective production selector | rejected and removed | исходный threshold не выбрал mandatory fixture, а `47 x 32` retest регрессировал |
| TP/SL reversal diagnostics | learning only, removed from runtime | counters полезны, но current-exact diagnostics дали `+99.5%` overhead |
| TP/SL split-by-side reversal repair | rejected and removed | parity прошла, но service wall и exact scoring регрессировали |
| TP/SL total-return early abandon | runtime rejected; learning retained | exact-safe bound сохранил parity, но не отрезал кандидатов и добавил service-wall overhead |
| Full pair cache / dense tensor | not accepted | memory/cost risk dominates; dense tensor explicitly rejected |

## Подробности По Методам

### Notebook Top-K No-Risk Exact Record

Evidence reference:
`docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`
упоминает `2026-04-27_iteration_4_no_risk_exact_scoring_notebook_topk` как
непринятый benchmark record.

Что показал record:

- semantic metrics parity `14 / 14`, proxy metadata parity `14 / 14` и
  `exact_scoring` latency `14 / 14` были полезны как локальная диагностика;
- `heap_update` fail `13 / 14`;
- `top_result_proxy_fill` fail для arity 2;
- strict hash drift на arity 1/2;
- `total_without_warmup` сравнивался с `service_total_without_warmup`, то есть
  boundary был неверным.

Вывод: не использовать как accepted baseline или target timing. Это checklist
против повторения ошибок в heap/result-shape и benchmark accounting.

### Quality Gate / Ranking-Only Exact / Confirm Prefilter

Evidence:
`docs/architecture/backtest/benchmark_iterations/2026-05-15_iteration_16_quality_gate_ranking_exact_arity6_cpu_memory/benchmark_summary.md`.

Идея:

- auto `min_closed_trades` через `timeframe_sqrt_v1`;
- row/candidate prefilter по `confirm_count`;
- ranking-only no-risk exact с гидратацией полных метрик только для shortlist.

Результат:

| Job | Previous exact s | New exact s | New service wall s | Result |
|---|---:|---:|---:|---|
| `none/arity_6/long_only` | 15.968 | 6.754 | 28.719 | fail |
| `none/arity_6/long_short_reversal` | 15.810 | 14.283 | 29.419 | fail, `top_count=0` |
| `tp_sl_grid/arity_6/long_only` | 16.566 | 17.223 | 38.545 | pass only by old gate, no quality telemetry |
| `tp_sl_grid/arity_6/long_short_reversal` | 15.504 | 15.366 | 16.726 | pass only by old gate, no quality telemetry |

Почему не ускорило:

- `confirm_prefilter` добавил отдельный полный проход `14.870..15.221s`;
- no-risk exact timer частично улучшился, но end-to-end child/service wall стал
  около `29s` вместо прежних `~16s`;
- `min_closed_trades=300` изменил result shape и отфильтровал все candidates для
  `long_short_reversal`;
- TP/SL path не получил consistent source of truth по quality telemetry.

Запрет на повтор: не возвращать отдельный expensive `confirm_prefilter` и эту
формулу `min_closed_trades=300` как production gate без нового correctness и
wall-clock gate.

### Signal Row Dedup As Production Pruning

Evidence:
`docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_02_row_signature_telemetry/`.

Что проверялось: возможность убрать duplicate signal rows до scoring.

Результат Stage 02 на accepted arity-6 rows:

- rows after prefilter: `36`;
- unique rows after dedup: `36`;
- duplicate rows: `0`;
- `consensus_signature_count=46656`;
- collision count: `0`;
- row signature overhead около `10..11ms/job`.

Вывод: row-signature telemetry полезна для безопасности и будущей диагностики,
но на текущей heavy fixture duplicate rows отсутствуют. Production pruning на
этой основе не дает ускорения, пока новые artifacts не покажут реальные
duplicates.

### Stage 04 Arity 2 Matrix Bitset MVP

Evidence:
`docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_04_no_risk_mvp/`.

Scope: `matrix_bitset_no_risk_v1` для `risk.mode=none`,
`direction_mode=long_only`, arity 2 и 3.

Результат:

| Job | Exact current s | May2 exact s | Ratio | Decision |
|---|---:|---:|---:|---|
| `none/arity_2/long_only` | 0.003684 | 0.002545 | 0.691 | speed fail |
| `none/arity_3/long_only` | 0.018385 | 0.047619 | 2.590 | speed pass |

Вывод: arity-2 слишком маленький workload, fixed overhead доминирует. Stage 04
разрешил Stage 05 learning path, но не production default. После production gate
2026-06-10 default включен только для `none/arity_6/long_only` и
`none/arity_6/long_short_reversal`, не для arity 2/3.

### Stage 06 Consensus Signature Cache

Evidence:
`docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_06_signature_cache/`.

Идея: группировать одинаковые consensus bitsets внутри matrix no-risk scoring и
переиспользовать exact result после collision-safe payload check.

Результат против принятого Stage 05 baseline:

| Job | Stage 05 exact s | Stage 06 exact s | Stage 05 wall s | Stage 06 wall s | Hit-rate |
|---|---:|---:|---:|---:|---:|
| `none/arity_6/long_only` | 1.010 | 4.932 | 1.590 | 5.366 | 0.202396 |
| `none/arity_6/long_short_reversal` | 2.887 | 6.166 | 3.135 | 6.414 | 0.202396 |

Почему не ускорило:

- cache hit-rate был реальным: `9443` hits, `37213` unique consensus,
  `collision_count=0`;
- стоимость keying/grouping/checking оказалась выше экономии scoring;
- both exact timer and service wall regressed on the hot API-runner boundary.

Запрет на повтор: не включать runtime consensus signature cache и не делать
следующие stages зависимыми от Stage 06. Возвращать можно только с более дешевой
identity/keying model и полным A/B against Stage 05/default baseline.

### Stage 07 Sidecar Bitset `.npy` Artifacts

Evidence:

- generation/report:
  `docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_07_sidecar_bitsets/`;
- API-runner:
  `docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_07_sidecar_bitsets_final/`.

Идея: заранее сгенерировать `.npy` bitsets:

- `signals_pos_bits.u64.npy`;
- `signals_neg_bits.u64.npy`;
- `signal_row_hashes.u64.npy`;
- `unique_signal_row_ids.u32.npy`;
- `duplicate_signal_row_ids.u32.npy`;
- `duplicate_unique_signal_row_ids.u32.npy`;
- `matrix_sidecar_manifest.json`.

Результат:

| Job | Stage 05 exact s | Stage 07 exact s | Sidecar load ms | Signal prep ms | Runtime pack ref |
|---|---:|---:|---:|---:|---:|
| `none/arity_6/long_only` | 1.010 | 1.028 | 81.530 | 102.640 | ~24.5ms |
| `none/arity_6/long_short_reversal` | 2.887 | 2.990 | 75.238 | 97.899 | ~24.5ms |

Почему не ускорило:

- generated payload около `519M`;
- sidecar load `75..82ms/job` медленнее Stage 03 runtime pack reference
  `~24.5ms/job`;
- exact-scoring/service path показал небольшой overhead.

Что можно сохранить: generator, loader, manifest validation, dtype/shape/padding
checks, duplicate-map validation и fallback как benchmark/test infrastructure.

Запрет на повтор: не менять `backtest_artifacts` publisher, canonical manifests,
`current.yaml`, active slots или production path ради этих sidecar `.npy` без
нового publisher-level плана и доказанного end-to-end speedup.

### Stage 08 TP/SL Selected-Cell Shadow

Evidence:
`docs/architecture/backtest/benchmark_iterations/2026-06-07_matrix_bitset_stage_08_tp_sl_selected_cells/`.

Идея: проверить TP/SL selected cells для `tp_count <= 8`, `sl_count <= 8`,
by-entry hit-times layout и `SL wins` tie rule.

Результат:

| Job | Cells | Shadow status | Selected scores | By-entry bytes | Exact ratio |
|---|---:|---|---:|---:|---:|
| `tp_sl_grid/arity_1/long_only` | 64 | passed | 384 | 7,145,344 | 1.269 |
| `tp_sl_grid/arity_2/long_short_reversal` | 64 | passed | 512 | 212,608 | 0.857 |

Вывод: это корректностный shadow и layout proof, а не production acceleration.
Он разрешил Stage 09 full-grid cell-block work, но не включил production
TP/SL backend и не является доказательством ускорения для top-N.

### Stage 09 TP/SL Full-Grid Cell Blocks, Shape `16 x 16`

Evidence:
`docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_09_tp_sl_full_grid/`.

Идея: `matrix_cell_tp_sl_v1` с block shape `16 x 16`.

Результат:

| Job | Exact current s | May2 exact s | Ratio | Result |
|---|---:|---:|---:|---|
| `tp_sl_grid/arity_6/long_only` | 25.032 | 17.446 | 0.697 | speed fail |
| `tp_sl_grid/arity_6/long_short_reversal` | 17.289 | 16.204 | 0.937 | partial/pass by ratio only |

Вывод: parity, instrumentation и memory прошли, но speed gate failed на
`long_only`. Не использовать `16 x 16` как accepted runtime shape. Принятая
Stage 09 shape - `64 x 64` из
`2026-06-10_matrix_bitset_stage_09_tp_sl_full_grid_64x64_rerun/`.

### Stage 10 High-Arity Min-Trade Pruning

Evidence:
`docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_10_high_arity_pruning_arity7_partial/`.

Идея: exact-safe branch-and-bound rule `monotonic_min_closed_trades`.

Что доказано:

- rule exact-safe для eligibility по `quality_constraints.min_closed_trades`;
- это не score upper bound и не разрешение на approximate beam/ranking.

Результат по первой completed arity-7 row:

| Metric | Value |
|---|---:|
| `combo_count_planned` | 279,936 |
| `candidates_after_proxy` | 116,640 |
| `combo_pruning_pruned_subtrees` | 3,246 |
| `combo_pruning_pruned_candidate_upper_bound` | 163,296 |
| `combo_iteration` | 59.350s |
| `exact_scoring` | 58.182s |
| `service_total_without_warmup` | 119.252s |

Почему не ускорило:

- Python branch traversal добавил cost уровня самого exact scoring;
- full arity-7 API-runner matrix не была завершена;
- comparable baseline-off speedup не был получен;
- arity-10 acceptance заблокирован, потому что текущий canonical fixture имеет
  только семь indicators.

Что можно сохранить: exact-safe proof как learning handoff для будущего более
дешевого bound. Runtime pruning implementation не считать ускорением.

### Stage 11 Lazy Detail Sparse Trade Tape Reuse

Evidence:

- candidate:
  `docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_11_lazy_detail_reuse/`;
- baseline:
  `docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_11_lazy_detail_reuse_baseline/`;
- comparison:
  `docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_11_lazy_detail_reuse/benchmark_comparison.md`.

Идея: reuse sparse trade tape backend только для TP/SL lazy selected-variant
materialization, без bulk top-N scoring change.

Результат:

| Risk mode | Baseline miss s | Candidate miss s | Miss delta | Baseline hit s | Candidate hit s | Hit delta |
|---|---:|---:|---:|---:|---:|---:|
| `none` | 2.869005 | 2.855623 | -0.466% | 0.000305 | 0.000299 | -2.090% |
| `tp_sl_grid` | 4.334214 | 4.292836 | -0.955% | 0.000301 | 0.000301 | -0.207% |

Вывод: parity прошла и регрессии нет, но delta меньше `1%` на miss path. Это не
material speedup и не оправдывает production runtime complexity.

### Stage 13/14 TP/SL Production Candidate Branch

Cleanup note: raw Stage 13/13S/13S2/13R/14R evidence directories, dedicated
benchmark harnesses and executed prompt files were removed from the active tree
on 2026-06-14 to avoid keeping a rejected branch as executable code/docs. This
section is the durable stop-list summary.

Rejected attempts:

- Stage 13 TP/SL block-shape production gate kept parity, but no tested shape
  (`64 x 64`, `128 x 32`, `32 x 128`, `128 x 64`, `64 x 128`) improved both
  mandatory TP/SL heavy rows by the required `>=15%` service-wall threshold.
- Stage 13S selective selector with `tp_count >= 64`, `sl_count >= 32` did not
  select the mandatory full-grid fixture because the real fixture was
  `tp_count=47`, `sl_count=47`.
- Stage 13S2 changed the threshold to `tp_count >= 47`, `sl_count >= 32`; the
  selector chose `matrix_cell_tp_sl_v1`, but long-only regressed
  `17.901s -> 19.573s` (`-9.342%`) and combined mandatory rows regressed
  `33.419s -> 35.265s` (`-5.524%`).
- Stage 13R diagnostics were useful only as learning: current-exact reversal
  diagnostics changed `15.528s -> 30.980s` (`+99.5%` overhead), so diagnostic
  counters are not accepted runtime behavior.
- Stage 14R split-by-side reversal repair preserved parity but regressed
  reversal service wall `15.578s -> 16.574s` (`-6.393%`) and exact scoring
  `15.182s -> 16.133s` (`-6.261%`), with higher sampled RSS.

Запрет на повтор:

- не возвращать Stage 13 block autotune как production gate without a new cost
  model and Mac Studio A/B evidence;
- не включать `ROEHUB_BACKTEST_TP_SL_BACKEND_MODE=stage_13s_selector` or an
  equivalent narrow TP/SL selector;
- не добавлять current-exact reversal diagnostic counters to the default scoring
  hot path;
- не продолжать split-by-side / entry-major reversal kernel repair as the next
  implementation direction.

### Stage 15 TP/SL Total-Return Early Abandon

Evidence:

- preflight:
  `docs/architecture/backtest/benchmark_iterations/2026-06-14_matrix_bitset_stage_05_12_production_default_stage15_preflight/`;
- control:
  `docs/architecture/backtest/benchmark_iterations/2026-06-14_matrix_bitset_stage_15_tp_sl_early_abandon_control/`;
- candidate:
  `docs/architecture/backtest/benchmark_iterations/2026-06-14_matrix_bitset_stage_15_tp_sl_early_abandon_candidate/`.

Идея: для `ranking=total_return_pct desc` и all-in sizing посчитать
optimistic log-return upper bound по кандидату и не запускать точный TP/SL
scoring, если bound строго ниже текущего top-N порога.

Результат:

| Job | Control wall s | Candidate wall s | Wall delta | Pruned candidates | Bound ms |
|---|---:|---:|---:|---:|---:|
| `tp_sl_grid/arity_6/long_only` | `17.728` | `31.298` | `-76.541%` | `0` | `13751.296` |
| `tp_sl_grid/arity_6/long_short_reversal` | `15.474` | `15.502` | `-0.180%` | `0` | `0.000` |

Почему не ускорило:

- на mandatory TP/SL fixture bound оказался слишком широким и не отрезал ни
  одного кандидата;
- long-only row получил дополнительный `13751.296ms` bound overhead до exact
  scoring;
- parity и memory cleanup прошли, но service-wall gate failed against current
  exact control.

Learning handoff: the exact-safe total-return bound shape is not viable on the
mandatory TP/SL fixture. The previously planned Stage 16 trade-window reuse
telemetry was closed without execution during the 2026-06-14 scope cleanup; it
is no longer an executable prompt or active roadmap item.

Запрет на повтор: не возвращать Python/Numba candidate-level total-return bound
как production hot path без предварительного дешевого reject-rate proof на той
же TP/SL fixture. Если идея будет переоткрыта, новый план должен сначала
доказать, что bound отрезает существенную долю candidates при overhead ниже
экономии exact scoring.

### Full Pair Cache And Dense Tensor Designs

Source:
`docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md`.

Full pair cache:

- не является default plan;
- допускается только как bounded/shadow optimization;
- должен доказывать memory peak, cleanup, hit-rate и service wall;
- failure to prove end-to-end speedup должен записываться как `rejected`.

Dense tensor `all_combos x all_bars x all_tp x all_sl`:

- explicitly rejected;
- не является viable production shape из-за memory/cost risk;
- не должен появляться как runtime или publisher artifact без нового
  architecture decision.

## Практический Стоп-Лист Для Будущих Prompt Pack

- Не планировать Stage 06 signature cache как dependency.
- Не менять artifact publisher ради Stage 07 sidecar `.npy` без отдельного
  publisher plan и нового benchmark gate.
- Не включать `matrix_bitset_no_risk_v1` default для arity 2/3.
- Не включать TP/SL selected-cell shadow как production backend.
- Не использовать TP/SL block shape `16 x 16` как accepted shape.
- Не возвращать Stage 13 block-shape production gate, Stage 13S/13S2 selector,
  Stage 13R current-exact diagnostics or Stage 14R split-by-side repair without
  a new benchmark-gated architecture decision.
- Не возвращать Stage 15 total-return early-abandon candidate без дешевого
  reject-rate proof; Mac Studio A/B на mandatory TP/SL rows дал `0` pruned
  candidates и service-wall regression.
- Не запускать Stage 16 trade-window reuse telemetry или Stage 21 exact/coarse
  product-mode work from this prompt pack; both were closed without execution
  and require a separate approved plan if reopened.
- Не возвращать Python high-arity branch traversal без более дешевого exact-safe
  bound и comparable baseline-off run.
- Не принимать sub-1% lazy detail delta как production optimization.
- Не считать локальный timer speedup достаточным, если service wall, top-N shape,
  memory или benchmark boundary регрессируют.

## Что Остается Принятым После Negative Review

На дату документа единственное default production acceleration из этого rollout:
`matrix_bitset_no_risk_v1` только для:

- `risk.mode=none`;
- arity `6`;
- `direction_mode in {long_only, long_short_reversal}`;
- rollback/comparison через `ROEHUB_BACKTEST_MATRIX_BACKEND_MODE=off`.

Отдельно accepted, но не default production:

- Stage 09 `matrix_cell_tp_sl_v1` full-grid TP/SL backend с accepted shape
  `64 x 64`, opt-in через internal env mode;
- Stage 01/02/03/07/08/10 learning/telemetry/shadow artifacts только как
  диагностическая база, не как доказанное production speedup.
- Stage 12 `compiled_prefix_product_traversal_v1` production composite default
  для no-risk arity `7`, while Stage 05 remains default for no-risk arity `6`.
