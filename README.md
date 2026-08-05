# Roehub

## Self-hosted release

Roehub `0.1.0` is assembled as digest-pinned `linux/amd64` and `linux/arm64`
OCI images. Generated profiles live under
`configs/installation/generated/{base,trading,ml}` and do not require a Git
checkout at installation time.

The signed offline bundle contains the images, platform-specific SPDX SBOMs,
release metadata, schemas, configuration, migrations, runbooks, notices and the
Roehub wheel. From the unpacked bundle, verify trust and prepare an immutable
Compose override with:

```bash
./tools/release/install-offline.sh \
  --trusted-public-key /secure/path/roehub-release-signing-key.pub \
  --profile base \
  --runtime-smoke
```

The trusted public key must come from a channel independent of the bundle. The
installer verifies the SSHSIG-Ed25519 signature and every payload digest before
loading host-platform images. It requires Python 3.9 or newer, OpenSSH `ssh-keygen`,
`skopeo`, Docker Engine and Docker Compose v2, but does not access a registry or
source checkout.

## Operations Notes

- Целевая модель Roehub — независимая self-hosted установка из подписанного
  multi-arch bundle.
- Прежняя host-specific схема выведена из эксплуатации и не используется как
  обязательная среда разработки, доказательства или поставки.
- Базовые требования следующей продуктовой трансформации:
  `docs/architecture/platform/roehub-product-transformation-requirements-v1.md`.
- Отдельный Compose-контур `market_data`
  (`infra/docker/docker-compose.market_data.yml`) остаётся только историческим
  локальным контуром разработки.

Полные исторические runbooks уже архивированы в `docs/runbooks/legacy/`;
прежние активные пути содержат только tombstone-уведомления:
- `docs/runbooks/market-data-autonomous-docker.md`
- `docs/runbooks/market-data-metrics.md`
- `docs/runbooks/market-data-metrics-reference-ru.md`
