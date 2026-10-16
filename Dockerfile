# syntax=docker/dockerfile:1
#
# The image serves the API only. Indexing is run natively (`uv run`) so that it
# can use Metal: a container on macOS has no access to the GPU.
#
# EXTRAS=ml adds sentence-transformers and a CPU-only torch, so that the API
# can answer, not just report health (docker-compose.yml builds it so); the
# default keeps the image small for CI, which only checks that it builds.

FROM python:3.14.7-slim-trixie AS builder

ARG EXTRAS=""

COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, from the lock file alone: this layer stays cached until
# pyproject.toml or uv.lock actually change.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-dev --no-install-project ${EXTRAS:+--extra $EXTRAS}

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev ${EXTRAS:+--extra $EXTRAS}


FROM python:3.14.7-slim-trixie

RUN useradd --create-home --uid 10001 app \
    # A named volume takes its owner from the image's directory when first
    # mounted: without this one, the model cache would belong to root.
    && mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /home/app/.cache

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app
EXPOSE 8000

CMD ["uvicorn", "nsu_rag_llm.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
