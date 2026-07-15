from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_artifact_store_migration_has_ownership_and_lifecycle_constraints() -> None:
    sql = (_REPO_ROOT / "migrations/postgres/0018_artifact_store_v1.sql").read_text()
    for fragment in (
        "artifact_store_objects",
        "artifact_store_object_locations",
        "artifact_store_org_blobs",
        "artifact_store_quotas",
        "artifact_store_manifests",
        "artifact_store_manifest_entries",
        "artifact_store_pins",
        "artifact_store_leases",
        "artifact_store_gc_candidates",
        "FOREIGN KEY (organization_id, digest)",
        "ON DELETE RESTRICT",
    ):
        assert fragment in sql
    object_table = sql.split("CREATE TABLE artifact_store_objects (", 1)[1].split(");", 1)[0]
    assert "media_type" not in object_table
    assert "backend" not in object_table
    assert "PRIMARY KEY (digest, backend)" in sql
