from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Type

from pydantic import BaseModel

from trading.integration import JobEnvelope, JobResultManifest

_SCHEMAS: tuple[tuple[str, Type[BaseModel], str], ...] = (
    (
        "job-envelope-v1.schema.json",
        JobEnvelope,
        "https://schemas.roehub.io/jobs/job-envelope-v1.schema.json",
    ),
    (
        "job-result-manifest-v1.schema.json",
        JobResultManifest,
        "https://schemas.roehub.io/jobs/job-result-manifest-v1.schema.json",
    ),
)

_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"


def _case_insensitive(value: str) -> str:
    return "".join(f"[{character.lower()}{character.upper()}]" for character in value)


_SECRET_KEY_PATTERN = (
    r"(?:^|[._-])(?:"
    + _case_insensitive("api")
    + r"[_-]?"
    + _case_insensitive("key")
    + "|"
    + "|".join(
        _case_insensitive(value)
        for value in (
            "authorization",
            "cookie",
            "credential",
            "dsn",
            "password",
            "secret",
            "token",
        )
    )
    + r")(?:$|[._-])"
)


def _job_config_schema() -> dict[str, object]:
    value_ref = {"$ref": "#/$defs/JobConfigValue"}
    object_schema: dict[str, object] = {
        "type": "object",
        "maxProperties": 256,
        "propertyNames": {
            "minLength": 1,
            "maxLength": 128,
            "not": {"pattern": _SECRET_KEY_PATTERN},
        },
        "additionalProperties": value_ref,
    }
    return {
        "oneOf": [
            {"type": "string", "maxLength": 16_384},
            {
                "type": "integer",
                "minimum": -(2**53 - 1),
                "maximum": 2**53 - 1,
            },
            {"type": "boolean"},
            {"type": "null"},
            {"type": "array", "maxItems": 4096, "items": value_ref},
            object_schema,
        ]
    }


def _strengthen_public_contract(*, name: str, payload: dict[str, object]) -> None:
    definitions = payload.setdefault("$defs", {})
    if not isinstance(definitions, dict):
        raise TypeError("generated schema definitions must be an object")
    if name == "job-envelope-v1.schema.json":
        payload["x-roehub-enforcement-boundary"] = (
            "trading.integration.JobEnvelope.model_validate"
        )
        definitions.pop("JsonValue", None)
        definitions["JobConfigValue"] = _job_config_schema()
        properties = payload["properties"]
        assert isinstance(properties, dict)
        properties["config_snapshot"] = {
            "type": "object",
            "maxProperties": 256,
            "propertyNames": {
                "minLength": 1,
                "maxLength": 128,
                "not": {"pattern": _SECRET_KEY_PATTERN},
            },
            "additionalProperties": {"$ref": "#/$defs/JobConfigValue"},
            "x-roehub-max-canonical-json-bytes": 256 * 1024,
        }
        inputs = properties["input_artifact_digests"]
        assert isinstance(inputs, dict)
        inputs["uniqueItems"] = True
        command = properties["command"]
        assert isinstance(command, dict)
        command["items"] = {
            "type": "string",
            "minLength": 1,
            "maxLength": 4096,
            "not": {"pattern": "\\u0000"},
        }
        command["x-roehub-max-utf8-bytes"] = 32 * 1024
        payload["allOf"] = [
            {
                "if": {
                    "properties": {"capability": {"const": "custom_strategy"}},
                    "required": ["capability"],
                },
                "then": {
                    "properties": {
                        "runtime": {
                            "properties": {
                                "plugin_package_digest": {
                                    "type": "string",
                                    "pattern": "^[0-9a-f]{64}$",
                                }
                            },
                            "required": ["plugin_package_digest"],
                        }
                    }
                },
            }
        ]
        return
    properties = payload["properties"]
    payload["x-roehub-enforcement-boundary"] = (
        "trading.integration.JobResultManifest.model_validate"
    )
    assert isinstance(properties, dict)
    outputs = properties["outputs"]
    assert isinstance(outputs, dict)
    outputs["x-roehub-unique-by"] = "path"
    outputs["x-roehub-max-aggregate-size-bytes"] = 64 * 1024 * 1024
    output_descriptor = definitions.get("JobOutputDescriptor")
    assert isinstance(output_descriptor, dict)
    descriptor_properties = output_descriptor.get("properties")
    assert isinstance(descriptor_properties, dict)
    output_path = descriptor_properties.get("path")
    assert isinstance(output_path, dict)
    output_path["pattern"] = (
        r"^(?!/)(?!.*//)(?!.*\\)(?!.*(?:^|/)(?:\.|\.\.)(?:/|$))" r"(?!.*\/$)[^\u0000]+$"
    )
    payload["allOf"] = [
        {
            "if": {
                "properties": {"outcome": {"const": "succeeded"}},
                "required": ["outcome"],
            },
            "then": {
                "properties": {
                    "output_artifact_manifest_digest": {
                        "type": "string",
                        "pattern": _DIGEST_PATTERN,
                    }
                },
                "required": ["output_artifact_manifest_digest"],
            },
            "else": {"properties": {"output_artifact_manifest_digest": {"type": "null"}}},
        }
    ]


def generate(*, output_root: Path, check: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    stale: list[str] = []
    for name, model, schema_id in _SCHEMAS:
        payload = model.model_json_schema(by_alias=True)
        _strengthen_public_contract(name=name, payload=payload)
        payload["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        payload["$id"] = schema_id
        content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        target = output_root / name
        if check:
            if not target.exists() or target.read_bytes() != content:
                stale.append(name)
        elif not target.exists() or target.read_bytes() != content:
            target.write_bytes(content)
    if stale:
        raise SystemExit(f"job schemas are stale: {', '.join(stale)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("schemas/jobs"),
    )
    args = parser.parse_args()
    generate(output_root=args.output_root, check=args.check)


if __name__ == "__main__":
    main()
