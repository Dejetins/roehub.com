from __future__ import annotations

from tools.release.runtime_input_inventory import OUTPUT_PATH, inventory_bytes, inventory_payload


def test_runtime_input_inventory_is_current_and_value_free() -> None:
    payload = inventory_payload()

    assert OUTPUT_PATH.read_bytes() == inventory_bytes()
    assert payload["total"] == len(payload["entries"])
    assert payload["file_input_total"] == len(payload["file_inputs"])
    assert payload["contract"] == {
        "legacy_env_is_user_contract": False,
        "raw_values_collected": False,
        "secret_values_allowed": False,
        "user_installation_input": "configs/installation/roehub.yaml",
    }
    assert all(
        set(entry) == {"classification", "key", "owner_stage", "sources"}
        for entry in payload["entries"]
    )
    assert all(entry["sources"] for entry in payload["entries"])
    assert all(set(item) == {"path", "sources"} for item in payload["file_inputs"])
