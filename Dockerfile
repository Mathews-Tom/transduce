# syntax=docker/dockerfile:1.9

FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/transduce-venv

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /usr/local/bin/uv

WORKDIR /build

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/transduce-venv/bin:${PATH}" \
    TRANSDUCE_HOST=0.0.0.0 \
    TRANSDUCE_PORT=8080

RUN groupadd --system --gid 1000 transduce \
    && useradd --system --uid 1000 --gid transduce --home /home/transduce --create-home transduce

COPY --from=builder /opt/transduce-venv /opt/transduce-venv

WORKDIR /etc/transduce

USER transduce

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, socket; s=socket.socket(); s.settimeout(2); \
    s.connect((os.environ.get('TRANSDUCE_HOST','127.0.0.1'), int(os.environ.get('TRANSDUCE_PORT','8080')))); \
    s.close()" || exit 1

ENTRYPOINT ["python", "-m", "transduce.cli"]
CMD ["serve", "--config", "/etc/transduce/config.yaml"]
