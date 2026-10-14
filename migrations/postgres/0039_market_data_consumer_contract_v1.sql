-- Coordinated PG-only consumer contract. Applied migrations remain immutable.
UPDATE market_data_candle_repair_events e SET sources_attempted_json = (
    SELECT jsonb_agg(CASE WHEN x->>'source'='clickhouse'
        THEN jsonb_set(x,'{source}','"canonical_store"'::jsonb) ELSE x END ORDER BY ord)
    FROM jsonb_array_elements(e.sources_attempted_json) WITH ORDINALITY AS a(x,ord)
) WHERE sources_attempted_json @> '[{"source":"clickhouse"}]'::jsonb;
UPDATE market_data_candle_repair_events SET
    error_code=regexp_replace(error_code,'^clickhouse_', 'canonical_store_')
WHERE error_code LIKE 'clickhouse\_%';
-- Embedded per-attempt error identifiers follow the same domain source name.
UPDATE market_data_candle_repair_events e SET sources_attempted_json = (
    SELECT jsonb_agg(CASE WHEN x->>'error_code' LIKE 'clickhouse\_%'
        THEN jsonb_set(x,'{error_code}',to_jsonb(regexp_replace(x->>'error_code',
            '^clickhouse_', 'canonical_store_'))) ELSE x END ORDER BY ord)
    FROM jsonb_array_elements(e.sources_attempted_json) WITH ORDINALITY AS a(x,ord)
) WHERE sources_attempted_json::text LIKE '%clickhouse_%';

-- Legacy revisions cannot establish equality after the storage generation changes.
UPDATE backtest_artifact_requests SET state='error',reason='legacy_source_target_invalid',
    control_version=control_version+1
WHERE snapshot->'source' IS NOT NULL AND snapshot->'source'<>'null'::jsonb
  AND (snapshot->'source'->>'source_generation' IS NULL
    OR snapshot->'source'->>'content_revision' IS NULL
    OR snapshot->'source'->>'funding_revision' IS NULL)
  AND state IN ('queued','waiting_for_source','running','retry_wait');
UPDATE backtest_artifact_builds SET state='quarantined'
WHERE state IN ('running','retry_wait')
  AND (snapshot->'source'->>'source_generation' IS NULL
    OR snapshot->'source'->>'content_revision' IS NULL
    OR snapshot->'source'->>'funding_revision' IS NULL);
UPDATE backtest_artifact_source_state SET desired_target='{"source":null,"state":"legacy_invalid"}',
    published='{}'::jsonb
WHERE source_contract_version IS DISTINCT FROM 2;
