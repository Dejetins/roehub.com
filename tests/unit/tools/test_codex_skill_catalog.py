from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools.codex_quality_benchmark.models import BenchmarkError
from tools.codex_quality_benchmark.skill_catalog import (
    _apply_effective_metadata,
    create_rollback_snapshots,
    mark_catalog_records,
    resolve_skill,
    verify_rollback_manifest,
    write_catalog_pair,
)


def test_resolve_skill_prefers_alias_target() -> None:
    catalog = {
        "skills": [
            {
                "skill_id": "S056",
                "logical_name": "Presentations",
                "canonical_name": "presentations",
                "aliases": ["Presentations"],
                "implementation_channel": "overlay",
            }
        ]
    }
    assert resolve_skill(catalog, "Presentations")["skill_id"] == "S056"


def test_resolve_deprecated_id_returns_canonical() -> None:
    catalog = {
        "skills": [
            {
                "skill_id": "S005",
                "logical_name": "gh-address-comments",
                "canonical_name": "gh-address-comments",
                "aliases": [],
                "implementation_channel": "deprecated",
            },
            {
                "skill_id": "S020",
                "logical_name": "gh-address-comments",
                "canonical_name": "gh-address-comments",
                "aliases": [],
                "implementation_channel": "overlay",
            },
        ]
    }
    assert resolve_skill(catalog, "S005")["skill_id"] == "S020"


def test_resolve_skill_rejects_a_not_exposed_audit_record() -> None:
    catalog = {
        "skills": [
            {
                "skill_id": "S001",
                "logical_name": "cache-only",
                "canonical_name": "cache-only",
                "aliases": [],
                "implementation_channel": "supplemental",
                "session_exposed": "not_exposed",
            }
        ]
    }

    with pytest.raises(BenchmarkError, match="not session-exposed"):
        resolve_skill(catalog, "cache-only")


def test_write_catalog_pair_has_identical_hashes(tmp_path: Path) -> None:
    catalog = {"spec": "fixture"}
    repo = tmp_path / "repo.json"
    global_path = tmp_path / "global.json"
    digest = write_catalog_pair(catalog, repo, global_path)
    assert digest == hashlib.sha256(repo.read_bytes()).hexdigest()
    assert repo.read_bytes() == global_path.read_bytes()


def test_mark_catalog_records_updates_only_named_ids(tmp_path: Path) -> None:
    skill = tmp_path / "SKILL.md"
    skill.write_text("contract\n", encoding="utf-8")
    catalog = {
        "skills": [
            {
                "skill_id": "S001",
                "effective_path": str(skill),
                "effective_sha256": None,
                "implementation_status": "pending",
                "verification_status": "pending",
                "result_contract_evidence": "pending",
                "evidence_refs": [],
            }
        ]
    }
    marked = mark_catalog_records(
        catalog,
        {"S001"},
        implementation_status="implemented",
        verification_status="structural_pass",
        evidence_ref="stage-00",
    )
    assert marked["skills"][0]["implementation_status"] == "implemented"
    assert marked["skills"][0]["effective_sha256"] is not None


def test_effective_v1_metadata_drives_catalog_behavior_and_relations(tmp_path: Path) -> None:
    primary = tmp_path / "primary" / "SKILL.md"
    companion = tmp_path / "companion" / "SKILL.md"
    primary.parent.mkdir()
    companion.parent.mkdir()
    primary.write_text(
        """---
name: primary
description: "Use to inspect a primary target. Do not use for mutation."
metadata:
  short-description: "Inspect primary"
  skill-spec: "v1"
  role: "gate"
  visibility: "public"
  owner: "user"
  mutability: "source"
  side-effect-class: "read-only"
  primary-output: "report"
  companions: ["companion"]
  conflicts: []
---
# Primary
""",
        encoding="utf-8",
    )
    records = [
        _catalog_record("S001", "historical-primary", primary),
        _catalog_record("S002", "companion", companion, channel="supplemental"),
    ]
    _apply_effective_metadata(records)
    assert records[0]["canonical_name"] == "primary"
    assert records[0]["role"] == "gate"
    assert records[0]["side_effect_class"] == "read-only"
    assert records[0]["companions"] == ["S002"]
    assert records[0]["effective_sha256"] is not None


def test_rollback_manifest_uses_content_addressed_blob(tmp_path: Path) -> None:
    source = tmp_path / "SKILL.md"
    source.write_text("safe contract\n", encoding="utf-8")
    manifest = tmp_path / "rollback" / "manifest.json"
    blob_dir = tmp_path / "rollback" / "blobs"
    data = create_rollback_snapshots(manifest, blob_dir, [source, tmp_path / "new.md"])
    present = next(row for row in data["entries"] if row["before_state"] == "present")
    assert Path(present["blob"]).name == f"{present['sha256']}.md"
    added = tmp_path / "added.md"
    added.write_text("later source\n", encoding="utf-8")
    merged = create_rollback_snapshots(manifest, blob_dir, [added])
    assert len(merged["entries"]) == 3
    assert any(row["path"] == str(source) for row in merged["entries"])
    verify_rollback_manifest(manifest)


def test_rollback_snapshot_rejects_secret_values(tmp_path: Path) -> None:
    source = tmp_path / "secret.md"
    source.write_text("Bearer " + "a" * 30, encoding="utf-8")
    with pytest.raises(BenchmarkError, match="possible secret"):
        create_rollback_snapshots(tmp_path / "manifest.json", tmp_path / "blobs", [source])


def _catalog_record(
    skill_id: str,
    name: str,
    effective_path: Path,
    *,
    channel: str = "direct",
) -> dict[str, object]:
    return {
        "skill_id": skill_id,
        "logical_name": name,
        "canonical_name": name,
        "effective_path": str(effective_path),
        "effective_sha256": None,
        "implementation_channel": channel,
        "role": "workflow",
        "visibility": "public",
        "owner": "user",
        "mutability": "source",
        "side_effect_class": "local-write",
        "primary_output": "patch",
        "aliases": [],
        "companions": [],
        "conflicts": [],
        "result_contract_provider": "skill_body",
    }
