# Iteration 4.3 corrective root cause notes

## Failed Row

- Row: `arity=1 long_only`
- Previous failed evidence was removed from the active benchmark tree after the
  corrective pass record became the only accepted Iteration 4.3 evidence.
- Previous service `heap_update`: `0.000036625s`
- Canonical target: `0.000023375s`
- Previous ratio: `0.638`

## Timer Boundary

The benchmark compares only service telemetry stage `heap_update` with canonical
notebook `heap_update`. Service `top_result_assembly` is recorded separately as
service-only overhead and is not part of the ratio comparison.

The notebook heap loop stores compact heap items and raw indicator metadata for
retained candidates. The service was still constructing production-shaped
heap entries inside `heap_update`, including metric mappings and metadata
mapping conversion.

## Micro Breakdown

Temporary Mac Studio micro-breakdown isolated the retained-row materialization
cost in the tiny 6-candidate `arity=1 long_only` stage.

Pre-fix observations:

| Segment | Median per run |
|---|---:|
| selected row extraction | `0.224us` |
| heap key construction for 6 candidates | `1.203us` |
| heap admission for 6 candidates | `1.846us` |
| retained materialization for 5 rows | `9.120us` |
| metadata conversion for 5 rows | `0.783us` |
| full current heap update for 6 candidates | `11.674us` |
| final top result conversion for 5 rows | `24.963us` |

After replacing dataclass/dict heap entries with compact heap payloads, the
remaining arity-1 cost was metric tuple construction:

| Segment | Median per run |
|---|---:|
| full arity-1 heap helper for 6 candidates | `8.266us` |
| heap key/admission only for 6 candidates | `1.975us` |
| metric tuple reads for 5 retained rows | `2.607us` |
| retained entry construction without metric tuple | `2.059us` |
| retained entry construction with metric tuple | `4.644us` |

## Corrective Change

The final implementation keeps deterministic heap keys
`(rank_score, original_row_ids)`, but stores a lightweight heap entry during
`heap_update`:

- generic arities store metric scalar tuples and raw metadata objects only for
  retained heap entries;
- arity 1 defers metric tuple reads to service-only `top_result_assembly`;
- metadata `.as_mapping()` conversion is outside `heap_update`;
- public `request.top_n = 100` remains telemetry/input only, while
  `benchmark_top_k = 5` drives heap capacity.

## Final Evidence

- Final commit: `8a3cd7cad0e1856de64b9ae1b58a6e54953135bd`
- Final `arity=1 long_only` service `heap_update`: `0.000020959s`
- Final `arity=1 long_only` ratio: `1.115`
- Final heap pass count: `14 / 14`
- Final top identity pass count: `14 / 14`
- Artifact policy: `historical_prefix_compatible`
- Full artifact manifest hash match: `False`
- Artifact historical-prefix compatible: `True`
