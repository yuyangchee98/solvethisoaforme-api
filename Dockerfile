# syntax=docker/dockerfile:1
# Backend image for Cloudflare Containers (and any Docker host).
# Must build for linux/amd64 — Cloudflare Containers requirement.

# ---- Python dependencies -------------------------------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# The two heaviest packages ship as their own image layers below: a single
# ~2 GB venv layer times out while the Cloudflare registry finalizes the
# blob upload (seen consistently from Workers Builds machines).
FROM builder AS venv-light
RUN rm -rf /app/.venv/lib/python3.12/site-packages/torch \
           /app/.venv/lib/python3.12/site-packages/en_core_web_trf

# ---- Runtime ---------------------------------------------------------------
FROM python:3.12-slim

# Node.js + the Claude Code CLI: the Claude Agent SDK spawns `claude` from
# PATH for every agent session.
COPY --from=node:22-slim /usr/local/bin/node /usr/local/bin/node
COPY --from=node:22-slim /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && npm install -g @anthropic-ai/claude-code@2.1.280 \
    && claude --version

WORKDIR /app
COPY --from=venv-light /app/.venv /app/.venv
COPY --from=builder /app/.venv/lib/python3.12/site-packages/torch /app/.venv/lib/python3.12/site-packages/torch
COPY --from=builder /app/.venv/lib/python3.12/site-packages/en_core_web_trf /app/.venv/lib/python3.12/site-packages/en_core_web_trf
COPY . .

ENV PATH="/app/.venv/bin:$PATH" \
    DATA_PATH=/data \
    HOME=/root \
    PYTHONUNBUFFERED=1

RUN mkdir -p /data

EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
