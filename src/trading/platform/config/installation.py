"""Strict, deterministic installation configuration for self-hosted Roehub."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

import jsonschema
import yaml

from trading.platform.secrets import SecretKind, SecretReference, SecretReferenceError

INSTALLATION_SCHEMA_ID = "io.roehub.installation/v1alpha1"
RELEASE_SCHEMA_ID = "io.roehub.release/v1alpha1"
RUNTIME_SCHEMA_ID = "io.roehub.runtime-config/v1alpha1"
GENERATION_SCHEMA_ID = "io.roehub.config-generation/v1alpha1"
SUPPORTED_PROFILES = ("base", "trading", "ml")

PROFILE_FEATURES: dict[str, tuple[str, ...]] = {
    "base": ("core", "local_artifacts", "notifications", "telegram_capability"),
    "trading": (
        "core",
        "local_artifacts",
        "market_data",
        "notifications",
        "paper_or_testnet_execution",
        "telegram_capability",
        "trading",
    ),
    "ml": (
        "core",
        "local_artifacts",
        "market_data",
        "ml",
        "notifications",
        "paper_or_testnet_execution",
        "rl",
        "telegram_capability",
        "trading",
    ),
}

PROFILE_STORES: dict[str, tuple[str, ...]] = {
    "base": ("postgresql", "redis"),
    "trading": ("postgresql", "clickhouse", "redis"),
    "ml": ("postgresql", "clickhouse", "redis"),
}

PROFILE_METRICS_TARGETS: dict[str, tuple[str, ...]] = {
    "base": ("api:8000", "notification-dispatcher:9204"),
    "trading": (
        "api:8000",
        "exchange-execution:9203",
        "market-data:9201",
        "notification-dispatcher:9204",
    ),
    "ml": (
        "api:8000",
        "exchange-execution:9203",
        "market-data:9201",
        "notification-dispatcher:9204",
        "rl-inference:9207",
        "rl-trainer:9208",
    ),
}

SENSITIVE_KEY_RE = re.compile(
    r"(?:^|_)(?:api_key|credential|credentials|password|passphrase|private_key|secret|token)"
    r"(?:$|_)",
    re.IGNORECASE,
)
SECRET_VALUE_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{12,}"),
    re.compile(r"\b(?:ghp|github_pat|xoxb)-?[A-Za-z0-9_]{12,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{12,}\b"),
    re.compile(r"(?i)\b(?:postgres(?:ql)?|redis|https?)://[^\s:/]+:[^\s@]+@"),
)
ALLOWED_SENSITIVE_KEYS = {
    "bot_token_ref",
    "certificate_ref",
    "client_secret_ref",
    "credentials_ref",
    "private_key_ref",
    "secret_root",
}
DANGEROUS_KEYS = {
    "cap_add",
    "command",
    "docker_socket",
    "entrypoint",
    "env_file",
    "environment",
    "mainnet",
    "network_mode",
    "privileged",
    "volumes",
}
IMAGE_DIGEST_RE = re.compile(r"^[^\s@]+@sha256:[a-f0-9]{64}$")


class InstallationConfigError(RuntimeError):
    """Raised when installation input or generated output is unsafe or ambiguous."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mappings."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise InstallationConfigError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def json_bytes(payload: Any) -> bytes:
    """Return byte-stable UTF-8 JSON."""

    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def yaml_bytes(payload: Any) -> bytes:
    """Return byte-stable UTF-8 YAML without aliases or document markers."""

    return yaml.safe_dump(
        payload,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
        width=1000,
    ).encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_yaml_bytes(payload: bytes, *, source: str) -> dict[str, Any]:
    try:
        value = yaml.load(payload.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise InstallationConfigError(f"invalid YAML in {source}: {error}") from error
    if not isinstance(value, dict):
        raise InstallationConfigError(f"YAML root must be an object: {source}")
    if not all(isinstance(key, str) for key in value):
        raise InstallationConfigError(f"YAML root keys must be strings: {source}")
    return value


def load_json_bytes(payload: bytes, *, source: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InstallationConfigError(f"invalid JSON in {source}: {error}") from error
    if not isinstance(value, dict):
        raise InstallationConfigError(f"JSON root must be an object: {source}")
    return value


def _schema_errors(payload: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    validator = jsonschema.Draft202012Validator(
        schema,
        format_checker=jsonschema.FormatChecker(),
    )
    return [
        f"{'/'.join(str(item) for item in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(payload), key=lambda item: list(item.path))
    ]


def _walk(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, (*path, str(index)))


def _validate_no_raw_secrets_or_dangerous_overrides(payload: dict[str, Any]) -> None:
    for path, value in _walk(payload):
        if not path:
            continue
        key = path[-1]
        lowered_key = key.lower()
        dotted = ".".join(path)
        if lowered_key in DANGEROUS_KEYS:
            raise InstallationConfigError(f"dangerous installation key is forbidden: {dotted}")
        if lowered_key == "mainnet" or (
            isinstance(value, str) and value.strip().lower() == "mainnet"
        ):
            raise InstallationConfigError(f"mainnet value is forbidden in v1 config: {dotted}")
        if SENSITIVE_KEY_RE.search(lowered_key) and lowered_key not in ALLOWED_SENSITIVE_KEYS:
            raise InstallationConfigError(
                f"raw secret-shaped installation key is forbidden; use *_ref: {dotted}"
            )
        if not isinstance(value, str):
            continue
        for pattern in SECRET_VALUE_PATTERNS:
            if pattern.search(value):
                raise InstallationConfigError(f"raw secret-shaped value is forbidden: {dotted}")


def _validate_release_images(release_manifest: dict[str, Any]) -> None:
    for name, image in release_manifest["images"].items():
        reference = image["reference"]
        if not IMAGE_DIGEST_RE.fullmatch(reference):
            raise InstallationConfigError(f"release image is not digest-pinned: {name}")
        if ":latest" in reference.lower():
            raise InstallationConfigError(f"release image uses latest: {name}")


def validate_installation(
    config: dict[str, Any],
    release_manifest: dict[str, Any],
    installation_schema: dict[str, Any],
    release_schema: dict[str, Any],
) -> None:
    """Validate schema, secret, profile, architecture, and safe-default invariants."""

    _validate_no_raw_secrets_or_dangerous_overrides(config)
    installation_errors = _schema_errors(config, installation_schema)
    if installation_errors:
        raise InstallationConfigError(
            "installation schema validation failed: " + "; ".join(installation_errors)
        )
    release_errors = _schema_errors(release_manifest, release_schema)
    if release_errors:
        raise InstallationConfigError(
            "release manifest schema validation failed: " + "; ".join(release_errors)
        )
    _validate_release_images(release_manifest)

    ports = list(config["ports"].values())
    if len(ports) != len(set(ports)):
        raise InstallationConfigError("installation ports must be unique")
    for name, raw_path in config["paths"].items():
        path = Path(raw_path)
        if ".." in path.parts:
            raise InstallationConfigError(f"installation path must not contain '..': {name}")
    if config["domain"] != "localhost" and config["tls"]["mode"] == "disabled":
        raise InstallationConfigError("TLS may be disabled only for localhost")
    if "ml" in config["profiles"] and "trading" not in config["profiles"]:
        raise InstallationConfigError("ml profile requires trading profile")
    if config["artifacts"]["mode"] == "local_cas" and (
        config["artifacts"]["path"] != config["paths"]["artifacts"]
    ):
        raise InstallationConfigError("local artifact path must match paths.artifacts")
    if config["artifacts"]["mode"] == "s3":
        endpoint = urlsplit(config["artifacts"]["endpoint"])
        if endpoint.scheme == "http" and endpoint.hostname not in {
            "127.0.0.1",
            "localhost",
            "minio",
        }:
            raise InstallationConfigError(
                "plaintext S3 endpoint is restricted to an embedded or loopback fixture"
            )
        try:
            SecretReference.parse(
                config["artifacts"]["credentials_ref"],
                expected_root=config["openbao"]["secret_root"],
                expected_kind=SecretKind.STORAGE,
            )
        except SecretReferenceError as error:
            raise InstallationConfigError(
                "S3 credentials_ref must use the configured OpenBao storage root"
            ) from error
    if config["openbao"]["mode"] == "external" and not config["openbao"]["address"].startswith(
        "https://"
    ):
        raise InstallationConfigError("external OpenBao address must use HTTPS")
    for path, value in _walk(config):
        if path and path[-1].lower().endswith("_ref") and isinstance(value, str):
            try:
                SecretReference.parse(
                    value,
                    expected_root=config["openbao"]["secret_root"],
                )
            except SecretReferenceError as error:
                raise InstallationConfigError(
                    f"secret reference is invalid or outside configured OpenBao root: "
                    f"{'.'.join(path)}"
                ) from error
    if config["architecture"] not in release_manifest["supported_architectures"]:
        raise InstallationConfigError(
            f"unsupported architecture for release: {config['architecture']}"
        )
    for name, image in release_manifest["images"].items():
        if config["architecture"] not in image["platforms"]:
            raise InstallationConfigError(
                f"release image {name} lacks architecture {config['architecture']}"
            )


def _redact_secret_refs(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<secret-reference:redacted>"
            if key.lower().endswith("_ref")
            else _redact_secret_refs(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_redact_secret_refs(item) for item in value]
    return value


def _effective_profile_config(config: dict[str, Any], profile: str) -> dict[str, Any]:
    stores = {name: copy.deepcopy(config["stores"][name]) for name in PROFILE_STORES[profile]}
    payload: dict[str, Any] = {
        "architecture": config["architecture"],
        "artifacts": copy.deepcopy(config["artifacts"]),
        "capabilities": list(PROFILE_FEATURES[profile]),
        "domain": config["domain"],
        "installation_id": config["installation_id"],
        "notifications": copy.deepcopy(config["notifications"]),
        "oidc": copy.deepcopy(config["oidc"]),
        "openbao": copy.deepcopy(config["openbao"]),
        "paths": copy.deepcopy(config["paths"]),
        "ports": copy.deepcopy(config["ports"]),
        "profile": profile,
        "proxy": copy.deepcopy(config["proxy"]),
        "resources": copy.deepcopy(config["resources"][profile]),
        "schema": RUNTIME_SCHEMA_ID,
        "stores": stores,
        "tls": copy.deepcopy(config["tls"]),
        "trading": copy.deepcopy(config["trading"]),
        "update_checks": copy.deepcopy(config["update_checks"]),
    }
    return payload


def _oidc_input(config: dict[str, Any], profile: str) -> dict[str, Any]:
    return {
        "oidc": copy.deepcopy(config["oidc"]),
        "profile": profile,
        "schema": "io.roehub.oidc-input/v1alpha1",
    }


def _openbao_input(config: dict[str, Any], profile: str) -> dict[str, Any]:
    references = sorted(
        value
        for path, value in _walk(config)
        if path and path[-1].lower().endswith("_ref") and isinstance(value, str)
    )
    return {
        "address": config["openbao"]["address"],
        "authentication": {
            "delivery": "file",
            "reload": "per-request",
            "token_file": "/run/secrets/roehub-openbao-token",
        },
        "mode": config["openbao"]["mode"],
        "profile": profile,
        "schema": "io.roehub.openbao-input/v1alpha1",
        "secret_references": references,
        "secret_root": config["openbao"]["secret_root"],
    }


def _prometheus_input(profile: str) -> dict[str, Any]:
    return {
        "global": {"scrape_interval": "15s"},
        "scrape_configs": [
            {
                "job_name": f"roehub-{profile}",
                "static_configs": [{"targets": list(PROFILE_METRICS_TARGETS[profile])}],
            }
        ],
    }


def _compose_fragment(
    config: dict[str, Any], release_manifest: dict[str, Any], profile: str
) -> dict[str, Any]:
    resources = config["resources"][profile]
    command = (
        "test -s /etc/roehub/service-config.json; "
        "test -s /etc/roehub/oidc.json; "
        "test -s /etc/roehub/openbao.json; "
        "test -s /etc/roehub/prometheus.yml; "
        f'grep -F \'"profile": "{profile}"\' /etc/roehub/service-config.json >/dev/null; '
        "printf 'config-consumer-ok\\n'"
    )
    return {
        "name": f"roehub-config-{profile}",
        "services": {
            "config-consumer": {
                "cap_drop": ["ALL"],
                "command": ["/bin/sh", "-eu", "-c", command],
                "deploy": {
                    "resources": {
                        "limits": {
                            "cpus": str(resources["cpus"]),
                            "memory": f"{resources['memory_mb']}M",
                        }
                    }
                },
                "environment": {"ROEHUB_PROFILE": profile},
                "image": release_manifest["images"]["config_consumer"]["reference"],
                "labels": {
                    "io.roehub.config.schema": INSTALLATION_SCHEMA_ID,
                    "io.roehub.profile": profile,
                },
                "network_mode": "none",
                "platform": config["architecture"],
                "pull_policy": "missing",
                "read_only": True,
                "security_opt": ["no-new-privileges:true"],
                "tmpfs": ["/tmp:rw,noexec,nosuid,size=16m"],
                "user": "65534:65534",
                "volumes": [
                    "./service-config.json:/etc/roehub/service-config.json:ro",
                    "./oidc.json:/etc/roehub/oidc.json:ro",
                    "./openbao.json:/etc/roehub/openbao.json:ro",
                    "./prometheus.yml:/etc/roehub/prometheus.yml:ro",
                ],
            }
        },
        "x-roehub": {
            "capabilities": list(PROFILE_FEATURES[profile]),
            "domain": config["domain"],
            "profile": profile,
            "release": release_manifest["version"],
        },
    }


def render_profile(
    config: dict[str, Any],
    release_manifest: dict[str, Any],
    profile: str,
    *,
    config_source: bytes,
    manifest_source: bytes,
) -> dict[str, bytes]:
    """Render one profile to deterministic files relative to its output directory."""

    if load_yaml_bytes(config_source, source="config_source") != config:
        raise InstallationConfigError("config_source bytes do not match parsed configuration")
    if load_json_bytes(manifest_source, source="manifest_source") != release_manifest:
        raise InstallationConfigError("manifest_source bytes do not match parsed release manifest")
    if profile not in SUPPORTED_PROFILES:
        raise InstallationConfigError(f"unsupported profile: {profile}")
    if profile not in config["profiles"]:
        raise InstallationConfigError(f"profile is not enabled by roehub.yaml: {profile}")

    effective = _effective_profile_config(config, profile)
    outputs = {
        "compose.yaml": yaml_bytes(_compose_fragment(config, release_manifest, profile)),
        "effective-config.redacted.json": json_bytes(_redact_secret_refs(effective)),
        "oidc.json": json_bytes(_oidc_input(config, profile)),
        "openbao.json": json_bytes(_openbao_input(config, profile)),
        "prometheus.yml": yaml_bytes(_prometheus_input(profile)),
        "service-config.json": json_bytes(effective),
    }
    generation_manifest = {
        "inputs": {
            "release_manifest_sha256": sha256_bytes(manifest_source),
            "roehub_yaml_sha256": sha256_bytes(config_source),
        },
        "outputs": {
            name: {"sha256": sha256_bytes(content)} for name, content in sorted(outputs.items())
        },
        "profile": profile,
        "schema": GENERATION_SCHEMA_ID,
    }
    outputs["generation-manifest.json"] = json_bytes(generation_manifest)
    return dict(sorted(outputs.items()))


def write_outputs(output_root: Path, profile: str, outputs: Mapping[str, bytes]) -> None:
    profile_root = output_root / profile
    profile_root.mkdir(parents=True, exist_ok=True)
    expected_names = set(outputs)
    for existing in profile_root.iterdir():
        if existing.is_file() and existing.name not in expected_names:
            raise InstallationConfigError(f"unexpected generated file: {existing}")
    for name, content in outputs.items():
        path = profile_root / name
        if not path.exists() or path.read_bytes() != content:
            path.write_bytes(content)


def check_outputs(output_root: Path, profile: str, outputs: Mapping[str, bytes]) -> None:
    profile_root = output_root / profile
    stale = [
        name
        for name, content in outputs.items()
        if not (profile_root / name).exists() or (profile_root / name).read_bytes() != content
    ]
    actual_names = (
        {path.name for path in profile_root.iterdir()} if profile_root.exists() else set()
    )
    extra = sorted(actual_names - set(outputs))
    if stale or extra:
        raise InstallationConfigError(
            f"generated installation config is stale: profile={profile}, "
            f"stale={sorted(stale)}, extra={extra}"
        )
