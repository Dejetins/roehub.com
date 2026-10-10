from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from apps.cli.wiring.db.clickhouse import ClickHouseSettingsLoader
from tools.installation.clickhouse_workloads import WORKLOADS, private_users, provision


@pytest.mark.parametrize("workload", WORKLOADS)
def test_class_identity_is_composition_owned_and_missing_secret_fails_closed(tmp_path, workload):
    credential = tmp_path / "password"
    credential.write_text("x" * 64)
    env = {
        "ROEHUB_CH_WORKLOADS_REQUIRED": "1",
        "CH_USER": "default",
        "CH_PASSWORD": "admin",
        "CH_PASSWORD_FILE": "/does/not/exist",
        f"ROEHUB_CH_{workload.upper()}_PASSWORD_FILE": str(credential),
    }
    settings = ClickHouseSettingsLoader(env).load(workload=workload)
    assert settings.user == "roehub_" + workload and settings.password == "x" * 64
    credential.unlink()
    with pytest.raises(ValueError, match="unreadable"):
        ClickHouseSettingsLoader(env).load(workload=workload)
    with pytest.raises(ValueError, match="workload is required"):
        ClickHouseSettingsLoader(env).load()


def test_additive_provisioning_preserves_credentials_and_refuses_unsafe_paths(tmp_path):
    root = tmp_path / "private"
    provision(root)
    first = {p.name: p.read_bytes() for p in root.iterdir()}
    provision(root)
    assert first == {p.name: p.read_bytes() for p in root.iterdir()}
    assert all(p.stat().st_mode & 0o077 == 0 for p in root.iterdir())
    (root / "stream.password").chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        provision(root)
    (root / "stream.password").chmod(0o600)
    (root / "users.xml").write_text("foreign configuration")
    with pytest.raises(ValueError, match="explicit migration"):
        provision(root)


def test_scoped_grants_do_not_give_interactive_or_artifacts_writes():
    users = ET.fromstring(private_users({w: "x" * 64 for w in WORKLOADS})).find("users")
    assert users is not None
    for name in ("interactive", "artifacts"):
        queries = [node.text or "" for node in users.findall(f"roehub_{name}/grants/query")]
        assert queries and all(q.startswith("GRANT SELECT ON market_data.") for q in queries)
        assert not any("*" in q for q in queries)
    xml = Path("configs/dev/clickhouse-local-profiles.xml").read_text()
    assert "<roehub_stream>" in xml


def test_entrypoint_disabled_default_is_not_recreated_without_authentication(tmp_path):
    root = tmp_path / "private"
    provision(root, default_user_disabled=True)
    users = ET.fromstring((root / "users.xml").read_bytes()).find("users")
    assert users is not None
    assert users.find("default") is None
    assert {node.tag for node in users} == {"roehub_" + w for w in WORKLOADS}
    first = {p.name: p.read_bytes() for p in root.iterdir()}
    provision(root, default_user_disabled=True)
    assert first == {p.name: p.read_bytes() for p in root.iterdir()}
    with pytest.raises(ValueError, match="explicit migration"):
        provision(root)
