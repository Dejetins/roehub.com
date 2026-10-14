-- Preserve global content revisions while allowing a frozen prefix to publish
-- after strictly later candles arrive. Unattested (older-writer) revisions fail closed.
ALTER TABLE market_data.instrument_generations
    ADD COLUMN last_correction_revision bigint NOT NULL DEFAULT 0
        CHECK (last_correction_revision >= 0),
    ADD COLUMN append_safe_revision bigint NOT NULL DEFAULT 0
        CHECK (append_safe_revision >= 0);
UPDATE market_data.instrument_generations SET
    last_correction_revision=content_revision,
    append_safe_revision=content_revision;
