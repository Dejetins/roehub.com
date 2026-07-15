# syntax=docker/dockerfile:1.7

FROM --platform=$BUILDPLATFORM docker.io/library/node@sha256:ba36e9b2705008e63e354214f0e3011c528af9df2ca13ac2bd2c0114650302e6 AS source

ARG OPENBAO_SOURCE_SHA256=f4ed87e3b6ec213e3aada0b4eba3ce1a9525620b14a0dde5be4de0c778be0291
WORKDIR /build
COPY openbao-v2.5.4.tar.gz openbao-2.5.4-licensed-qr.patch openbao-2.5.4-licensed-qr.NOTICE ./
COPY openbao-runtime-packages-amd64.tar.gz openbao-runtime-packages-arm64.tar.gz ./
RUN printf '%s  %s\n' "${OPENBAO_SOURCE_SHA256}" openbao-v2.5.4.tar.gz | sha256sum -c - \
    && printf '%s  %s\n' 3e26a337cf92c8ca21797fbf622f27b9b492f2e74d9a5af980190330fa733b8f openbao-runtime-packages-amd64.tar.gz | sha256sum -c - \
    && printf '%s  %s\n' a537d976b5ef3dfae7ea89f86494c71a2ce077b6c22dbf077f6f2d4dc3586196 openbao-runtime-packages-arm64.tar.gz | sha256sum -c - \
    && ! tar -tzf openbao-runtime-packages-amd64.tar.gz | grep -E '(^|/)(bao|vault)$|github.com/yeqown' \
    && ! tar -tzf openbao-runtime-packages-arm64.tar.gz | grep -E '(^|/)(bao|vault)$|github.com/yeqown' \
    && mkdir /src \
    && tar -xzf openbao-v2.5.4.tar.gz --strip-components=1 -C /src \
    && patch --fuzz=0 --batch --forward -d /src -p1 < openbao-2.5.4-licensed-qr.patch \
    && ! grep -R -E 'github.com/yeqown/(go-qrcode|reedsolomon)' /src/go.mod /src/go.sum /src/builtin/credential/jwt/cli_qr.go \
    && grep -F 'github.com/skip2/go-qrcode v0.0.0-20200617195104-da1b6568686e' /src/go.mod

FROM source AS ui-build

ENV COREPACK_ENABLE_DOWNLOAD_PROMPT=0 \
    DO_NOT_TRACK=1 \
    JOBS=1 \
    NODE_OPTIONS=--max-old-space-size=2048 \
    SCARF_ANALYTICS=false \
    YARN_ENABLE_TELEMETRY=0
WORKDIR /src/ui
RUN corepack enable \
    && yarn install --immutable \
    && npm rebuild node-sass \
    && yarn run build \
    && test -s /src/http/web_ui/index.html

FROM --platform=$BUILDPLATFORM docker.io/library/golang@sha256:8d22e29d960bc50cd025d93d5b7c7d220b1ee9aa7a239b3c8f55a57e987e8d45 AS go-build

ARG TARGETOS
ARG TARGETARCH
ARG OPENBAO_REVISION=4f6d47246a053375271a5fd8af85c3b75695aa46+roehub-licensed-qr.1
ARG OPENBAO_BUILD_DATE=2026-05-20T15:53:48Z
ENV CGO_ENABLED=0 \
    GOMEMLIMIT=1GiB \
    GOMAXPROCS=2
WORKDIR /src
COPY --from=source /src ./
COPY --from=ui-build /src/http/web_ui ./http/web_ui
RUN --mount=type=cache,target=/go/pkg/mod,sharing=locked \
    go mod download \
    && go mod verify \
    && ! go mod graph | grep -E 'github.com/yeqown/(go-qrcode|reedsolomon)' \
    && go list -deps -tags=ui . | grep -F 'github.com/skip2/go-qrcode' \
    && go test ./builtin/credential/jwt -run '^TestPrintQR$' -count=1
RUN --mount=type=cache,target=/go/pkg/mod,sharing=locked \
    --mount=type=cache,target=/root/.cache/go-build,sharing=locked \
    mkdir -p /out \
    && GOOS="${TARGETOS}" GOARCH="${TARGETARCH}" go build \
        -buildvcs=false \
        -trimpath \
        -tags=ui \
        -ldflags="-buildid= -s -w -X github.com/openbao/openbao/version.fullVersion=2.5.4 -X github.com/openbao/openbao/version.GitCommit=${OPENBAO_REVISION} -X github.com/openbao/openbao/version.BuildDate=${OPENBAO_BUILD_DATE}" \
        -o /out/bao \
        . \
    && cp /go/pkg/mod/github.com/skip2/go-qrcode@v0.0.0-20200617195104-da1b6568686e/LICENSE /out/skip2-go-qrcode-LICENSE \
    && go version -m /out/bao > /out/bao.buildinfo \
    && grep -F 'github.com/skip2/go-qrcode' /out/bao.buildinfo \
    && ! grep -E 'github.com/yeqown/(go-qrcode|reedsolomon)' /out/bao.buildinfo

FROM --platform=$BUILDPLATFORM docker.io/library/alpine@sha256:5b10f432ef3da1b8d4c7eb6c487f2f5a8f096bc91145e68878dd4a5019afde11 AS runtime-common

RUN addgroup openbao \
    && adduser -S -G openbao openbao \
    && mkdir -p /overlay/openbao/logs /overlay/openbao/file /overlay/openbao/config /overlay/licenses /overlay/bin \
    && chown -R openbao:openbao /overlay/openbao \
    && ln -s /bin/bao /overlay/bin/vault

FROM docker.io/library/alpine@sha256:5b10f432ef3da1b8d4c7eb6c487f2f5a8f096bc91145e68878dd4a5019afde11 AS runtime

ARG OPENBAO_BUILD_DATE=2026-05-20T15:53:48Z
LABEL description="OpenBao 2.5.4 with Roehub's licensed JWT QR dependency replacement." \
    maintainer="Roehub contributors" \
    name="Roehub OpenBao" \
    org.opencontainers.image.base.name="docker.io/library/alpine:3.23.4" \
    org.opencontainers.image.created="${OPENBAO_BUILD_DATE}" \
    org.opencontainers.image.description="OpenBao 2.5.4 with the unlicensed yeqown/reedsolomon dependency replaced by MIT-licensed skip2/go-qrcode." \
    org.opencontainers.image.documentation="https://github.com/Dejetins/roehub.com" \
    org.opencontainers.image.licenses="MPL-2.0 AND MIT" \
    org.opencontainers.image.revision="4f6d47246a053375271a5fd8af85c3b75695aa46+roehub-licensed-qr.1" \
    org.opencontainers.image.source="https://github.com/Dejetins/roehub.com" \
    org.opencontainers.image.title="roehub-openbao" \
    org.opencontainers.image.url="https://github.com/Dejetins/roehub.com" \
    org.opencontainers.image.vendor="Roehub contributors and OpenBao" \
    org.opencontainers.image.version="2.5.4-roehub-licensed-qr.1" \
    release="2.5.4-roehub-licensed-qr.1" \
    revision="4f6d47246a053375271a5fd8af85c3b75695aa46+roehub-licensed-qr.1" \
    summary="OpenBao for the Roehub self-hosted distribution." \
    vendor="Roehub contributors and OpenBao" \
    version="2.5.4-roehub-licensed-qr.1"

ENV NAME=openbao \
    VERSION=2.5.4-roehub-licensed-qr.1

ARG TARGETARCH
COPY --from=runtime-common /etc/passwd /etc/group /etc/shadow /etc/
COPY --from=runtime-common /overlay/ /
ADD openbao-runtime-packages-${TARGETARCH}.tar.gz /

COPY --from=source /src/LICENSE /licenses/openbao-MPL-2.0.txt
COPY --from=source /build/openbao-2.5.4-licensed-qr.NOTICE /licenses/NOTICE
COPY --from=go-build /out/skip2-go-qrcode-LICENSE /licenses/skip2-go-qrcode-MIT.txt
COPY --from=go-build /out/bao /bin/bao
COPY --from=source /src/.release/docker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

VOLUME ["/openbao/logs", "/openbao/file"]
EXPOSE 8200
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["server", "-dev", "-dev-no-store-token"]
