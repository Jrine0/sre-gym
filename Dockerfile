# SRE Gym - Self-Healing Kubernetes SRE Gym
# Optimized for <2GB image size and HF Spaces compatibility

FROM python:3.11-slim

WORKDIR /app

# Install minimal dependencies for HF Spaces
# Note: kubectl/kind not needed in container - they are for local dev only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy project files
COPY pyproject.toml .
COPY sre_gym/ ./sre_gym/
COPY inference.py .
COPY app.py .
COPY openenv.yaml .

# Install dependencies (minimal for HF Spaces)
# Note: kubernetes package is only for local kubectl access
# For production, set KUBECONFIG env var to connect to cluster
RUN pip install --no-cache-dir \
    "pydantic>=2.9" \
    "openai>=1.30" \
    "typer>=0.12" \
    "gradio>=4.0" \
    "huggingface_hub>=0.20" \
    || pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from sre_gym.env import SREGymEnv; print('OK')" || exit 1

# Default command: run HF Spaces app
CMD ["python", "app.py"]
