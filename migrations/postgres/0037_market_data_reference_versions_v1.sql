-- Status updates omit enrichment. Independent clocks prevent order-dependent loss
-- across the two existing reference ports; explicit enrichment NULL is authoritative.
ALTER TABLE market_data.ref_instruments
    ADD COLUMN enrichment_updated_at timestamptz(3) NOT NULL DEFAULT '-infinity',
    ADD COLUMN enrichment_order bytea NOT NULL DEFAULT ''::bytea;
