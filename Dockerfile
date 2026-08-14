# Pi (Python port) runtime image — isolates the coding agent's tools.
#
# The 7 built-in tools (read/write/edit/bash/grep/find/ls) run *inside* this
# container with the in-container user's privileges. We drop to a non-root user
# so a compromised agent cannot touch root-owned files. Mount a single host
# directory as the workspace and supply only the LLM key you need (see
# docker-compose.yml / README "Docker 隔离运行前提").

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# uv is the workspace package manager.
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy the uv workspace (root pyproject + packages/*).
COPY . /app

# Install runtime deps + optional extras (textual, opentelemetry-otlp),
# but skip the dev group (pytest, ruff) to keep the image lean.
RUN uv sync --all-extras --no-dev

# Run as a non-root user so a compromised agent cannot escalate to root.
RUN useradd --create-home --uid 1000 agent
USER agent

WORKDIR /workspace
ENTRYPOINT ["uv", "run", "--no-sync"]
CMD ["python", "-m", "pi_coding_agent"]
