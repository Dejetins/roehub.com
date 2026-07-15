# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-bookworm@sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /plugin

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --cache-dir /root/.cache/pip cryptography==43.0.3

COPY server.py /plugin/server.py
COPY fixture-config.json /plugin/fixture-config.json

RUN chown -R 10001:10001 /plugin

USER 10001:10001
EXPOSE 8080

ENTRYPOINT ["python", "/plugin/server.py"]
