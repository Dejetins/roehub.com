# Backtest AI Configurator Pipeline Smoke

Direct API readiness smoke for supported, repair and unsupported prompts.

## Summary

- accepted: True
- blocking_reason: None
- next_prompt_allowed: True
- supported prompts: 10/10 ready
- 10/10 ready: True
- repair prompts: 5/5 ready
- unsupported/off-topic prompts: 2/2 expected

## Supported Ready

| case_id | status | load_action | codes |
| --- | --- | --- | --- |
| ready_btc_rsi_ru | ready | True |  |
| ready_btc_rsi_en | ready | True |  |
| ready_btc_ema_en | ready | True |  |
| ready_btc_hma_en | ready | True |  |
| ready_btc_dema_en | ready | True |  |
| ready_btc_sma_en | ready | True |  |
| ready_btc_tp_sl_en | ready | True |  |
| ready_edit_btc_ema_en | ready | True |  |
| ready_repair_btc_invalid_en | ready | True |  |
| ready_btc_top10_en | ready | True |  |

## Repair

| case_id | status | load_action | codes |
| --- | --- | --- | --- |
| repair_btc_unsupported_timeframe | ready | True |  |
| repair_eth_unsupported_indicator | ready | True |  |
| repair_sol_missing_risk | ready | True |  |
| repair_btc_bad_symbol | ready | True |  |
| repair_btc_invalid_range | ready | True |  |

## Unsupported and Off-topic

| case_id | status | load_action | codes |
| --- | --- | --- | --- |
| unsupported_off_topic_ru | blocked_by_policy | False | off_topic |
| unsupported_doge_bollinger_en | needs_clarification | False | schema_validation_failed |

## Rollout Decision

- Accepted: True
- Reason: accepted pipeline readiness smoke
- Blockers: none
