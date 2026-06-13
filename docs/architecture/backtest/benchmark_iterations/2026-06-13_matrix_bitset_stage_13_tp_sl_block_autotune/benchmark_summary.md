# Stage 13 TP/SL block autotune production gate

## Короткий вывод

- Stage status: `rejected`.
- Reason: `no_shape_met_stage_13_service_wall_gate`.
- Best shape: `64x128`.
- Production default change: `not performed by this gate report`.

## Shape matrix

| Shape | Job | wall s | vs Stage 09 | vs current exact | peak RSS | memory vs Stage 09 | top parity | evals/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `64x64` | `tp_sl_grid/arity_6/long_only` | 27.060185 | 0.000% | 30.009% | 775028736 | 0.000% | pass | 4018766.227042 |
| `64x64` | `tp_sl_grid/arity_6/long_short_reversal` | 17.818542 | 0.000% | -14.231% | 619315200 | 0.000% | pass | 5913589.897852 |
| `128x32` | `tp_sl_grid/arity_6/long_only` | 20.399236 | 24.615% | 47.238% | 652214272 | -15.846% | pass | 5436982.950125 |
| `128x32` | `tp_sl_grid/arity_6/long_short_reversal` | 17.816628 | 0.011% | -14.219% | 642318336 | 3.714% | pass | 5914425.428449 |
| `32x128` | `tp_sl_grid/arity_6/long_only` | 20.303614 | 24.969% | 47.485% | 485343232 | -37.377% | pass | 5461722.060557 |
| `32x128` | `tp_sl_grid/arity_6/long_short_reversal` | 17.879099 | -0.340% | -14.620% | 622247936 | 0.474% | pass | 5892723.416099 |
| `128x64` | `tp_sl_grid/arity_6/long_only` | 19.990900 | 26.124% | 48.294% | 626966528 | -19.104% | pass | 5586523.032735 |
| `128x64` | `tp_sl_grid/arity_6/long_short_reversal` | 18.153907 | -1.882% | -16.381% | 486326272 | -21.474% | pass | 5818387.805018 |
| `64x128` | `tp_sl_grid/arity_6/long_only` | 19.602944 | 27.558% | 49.297% | 491175936 | -36.625% | pass | 5683399.566339 |
| `64x128` | `tp_sl_grid/arity_6/long_short_reversal` | 17.833881 | -0.086% | -14.330% | 508641280 | -17.870% | pass | 5911723.125284 |

## Gate

- Required: every TP/SL row keeps top sample identity/order, `best_tp`/`best_sl`, service wall improves >= 15% vs Stage 09 64x64, and peak RSS worsens <= 10%.
- Controls pass: `yes`.
- Missing payloads: `[]`.
- Crashed runs: `[]`.
