# Автономная установка подписанного выпуска

## Область применения

Инструкция относится к распакованному комплекту
`io.roehub.offline-release-bundle/v1alpha1`. Она не переносит текущую базу,
пользователей, секреты или артефакты и предназначена только для новой установки.

## Предварительные условия

- Docker Engine и Docker Compose v2 доступны локально;
- установлены Python 3.9 или новее, OpenSSH `ssh-keygen` и `skopeo`;
- комплект перенесён на целевой хост целиком;
- доверенный `ssh-ed25519` public key получен отдельно от комплекта.

Не используйте `trust/release-signing-key.pub` из комплекта как единственный
источник доверия. Встроенный ключ позволяет сверить identity, но не доказывает,
кто передал комплект.

## Проверка без активации

Из корня распакованного комплекта выполните:

```bash
python3 tools/release/offline_bundle.py verify \
  --bundle "$PWD" \
  --trusted-public-key /secure/path/roehub-release-signing-key.pub
```

Команда до любых изменений Docker проверяет:

- `SSHSIG-Ed25519` подпись манифеста;
- identity внешнего доверенного ключа;
- точный список, размер, режим и SHA-256 каждого файла;
- digest OCI index и child manifests для `linux/amd64`/`linux/arm64`;
- два непустых SPDX 2.3 SBOM для каждого образа;
- соответствующие исходники Grafana/Loki и запрет phone-home по умолчанию.

Любая ошибка является fail-closed: не переходите к импорту или запуску.

## Импорт и подготовка профиля

```bash
./tools/release/install-offline.sh \
  --trusted-public-key /secure/path/roehub-release-signing-key.pub \
  --state-directory "$HOME/.local/share/roehub/offline" \
  --profile base \
  --runtime-smoke
```

Допустимые профили: `base`, `trading`, `ml`. Installer повторяет полную
проверку, извлекает из локальных OCI archive только архитектуру текущего хоста,
загружает образы через Docker без registry и создаёт:

- `offline-image-lock.json` с immutable Docker image ID;
- `compose.<profile>.offline.yaml` с `pull_policy: never`;
- результат `docker compose config` для исходного и offline override.

Опция `--runtime-smoke` запускает Roehub с `--network none`, read-only rootfs и
writable tmpfs. Она не поднимает хранилища и не меняет пользовательские данные.

## Явная активация

После успешной проверки оператор может отдельно запустить выбранный профиль:

```bash
docker compose \
  -f "configs/installation/generated/base/compose.yaml" \
  -f "$HOME/.local/share/roehub/offline/compose.base.offline.yaml" \
  up -d
```

Перед `up -d` настройте секретные ссылки OpenBao и persistent paths согласно
`configs/installation/roehub.yaml`. Не помещайте значения секретов в YAML или
командную строку.

## Остановка и диагностика

Остановка не удаляет данные:

```bash
docker compose \
  -f "configs/installation/generated/base/compose.yaml" \
  -f "$HOME/.local/share/roehub/offline/compose.base.offline.yaml" \
  down
```

При ошибке сохраните stderr, `offline-image-lock.json`, версию Docker/Compose и
SHA-256 `offline-release-manifest.json`. Не заменяйте образы тегами и не
редактируйте подписанные файлы: получите новый полный комплект.
