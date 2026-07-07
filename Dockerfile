FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    UV_PROJECT_ENVIRONMENT=/home/user/.venv
WORKDIR /app

COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
# Bake the embedding model into the image so startup doesn't hit the HF Hub.
RUN uv run --no-sync python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')"
COPY --chown=user . .
RUN uv sync --frozen --no-dev

EXPOSE 7860
CMD ["uv", "run", "--no-sync", "uvicorn", "--factory", "papers_mcp.server:create_app", \
     "--host", "0.0.0.0", "--port", "7860"]
