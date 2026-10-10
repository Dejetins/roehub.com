BEGIN;

-- Browser-profile identity is separate from the rotating bearer session credential.
CREATE TABLE IF NOT EXISTS identity_browser_devices (
    device_id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES identity_users(user_id) ON DELETE CASCADE,
    device_key TEXT NOT NULL CHECK (length(device_key) = 64),
    label TEXT NOT NULL CHECK (length(label) BETWEEN 1 AND 120),
    current_session_id UUID NOT NULL REFERENCES identity_sessions(session_id) ON DELETE CASCADE,
    UNIQUE (user_id, device_key)
);
ALTER TABLE identity_sessions ADD COLUMN IF NOT EXISTS device_id UUID NULL
    REFERENCES identity_browser_devices(device_id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_identity_sessions_device ON identity_sessions(device_id);
CREATE INDEX IF NOT EXISTS idx_identity_browser_devices_owner
    ON identity_browser_devices(user_id, device_id DESC);

-- The user row serializes registration and owner revocation, including first login races.
CREATE OR REPLACE FUNCTION identity_bind_browser_device(
    p_session UUID, p_key TEXT, p_label TEXT, p_now TIMESTAMPTZ, p_device UUID
) RETURNS UUID LANGUAGE plpgsql AS $$
DECLARE
    v_owner UUID;
    v_session identity_sessions%ROWTYPE;
    v_device UUID;
BEGIN
    SELECT user_id INTO v_owner FROM identity_sessions WHERE session_id = p_session;
    IF v_owner IS NULL THEN RAISE EXCEPTION 'inactive_session'; END IF;
    PERFORM 1 FROM identity_users WHERE user_id = v_owner FOR UPDATE;
    SELECT * INTO v_session FROM identity_sessions WHERE session_id = p_session FOR UPDATE;
    IF v_session.revoked_at IS NOT NULL OR v_session.idle_expires_at <= p_now
       OR v_session.absolute_expires_at <= p_now THEN
        RAISE EXCEPTION 'inactive_session';
    END IF;
    IF v_session.device_id IS NOT NULL THEN RETURN v_session.device_id; END IF;
    INSERT INTO identity_browser_devices(device_id, user_id, device_key, label, current_session_id)
    VALUES (p_device, v_owner, p_key, p_label, p_session)
    ON CONFLICT (user_id, device_key) DO UPDATE
        SET label = EXCLUDED.label, current_session_id = EXCLUDED.current_session_id
    RETURNING device_id INTO v_device;
    UPDATE identity_sessions SET revoked_at = GREATEST(p_now, created_at)
        WHERE user_id = v_owner AND device_id = v_device
          AND session_id <> p_session AND revoked_at IS NULL;
    UPDATE identity_sessions SET device_id = v_device, last_seen_at = GREATEST(last_seen_at, p_now)
        WHERE session_id = p_session;
    RETURN v_device;
END;
$$;

CREATE OR REPLACE FUNCTION identity_revoke_browser_device(
    p_owner UUID, p_device UUID, p_now TIMESTAMPTZ
) RETURNS VOID LANGUAGE plpgsql AS $$
BEGIN
    PERFORM 1 FROM identity_users WHERE user_id = p_owner FOR UPDATE;
    UPDATE identity_sessions SET revoked_at = GREATEST(p_now, created_at)
        WHERE user_id = p_owner AND device_id IS NOT DISTINCT FROM p_device
          AND revoked_at IS NULL;
END;
$$;

COMMIT;
