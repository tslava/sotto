# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:0.11.21 AS uv

# ---- builder: build the virtual environment ----------------------------------
FROM python:3.12-slim AS builder

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock ./
COPY src ./src

RUN uv sync --frozen --no-dev

# ---- runtime: minimal image without build tools -------------------------------
FROM python:3.12-slim AS runtime

RUN groupadd --gid 10001 bot \
    && useradd --uid 10001 --gid 10001 --system --no-create-home bot


WORKDIR /app

COPY --from=builder --chown=bot:bot /app/.venv /app/.venv
COPY --chown=bot:bot pyproject.toml uv.lock ./
COPY --chown=bot:bot src ./src
COPY --chown=bot:bot CHANGELOG.md ./
# Persona packs (personas/_legacy_ru until the demo pack lands). The image sets no
# PERSONA_DIR: the pack is selected at run time and a missing mount fails closed.
COPY --chown=bot:bot personas ./personas
# The engine's own owner-UI texts (English); a pack's locale/<code>/ overrides them.
COPY --chown=bot:bot locales ./locales

# /app/data is the bind-mount point from docker-compose.yml (see README: the host
# directory needs chown 10001:10001 before the first start).
RUN mkdir -p /app/data && chown -R bot:bot /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

USER bot

# Nothing listens for inbound connections, so no EXPOSE (long polling).
# The token is read from the container environment and never written to the log.
HEALTHCHECK --interval=60s --timeout=10s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import os, sys, httpx\nurl = 'https://api.telegram.org/bot' + os.environ['BOT_TOKEN'] + '/getMe'\nr = httpx.get(url, timeout=5.0)\nsys.exit(0 if r.status_code == 200 else 1)"]

# No uv at runtime: the bot user has no home directory and uv fails on its cache.
CMD ["python", "-m", "sotto"]
