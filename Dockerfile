# SRE Gym - Self-Healing Kubernetes SRE Gym
# Optimized for <2GB image size and HF Spaces compatibility

FROM python:3.11-slim

WORKDIR /app

# Install kubectl
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && chmod +x kubectl \
    && mv kubectl /usr/local/bin/ \
    && curl -Lo /usr/local/bin/kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64 \
    && chmod +x /usr/local/bin/kind \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY sre_gym/ ./sre_gym/
COPY tests/ ./tests/
COPY inference.py .
COPY app.py .
COPY openenv.yaml .

# Install base dependencies + HF Spaces dependencies
RUN pip install --no-cache-dir ".[spaces]" || pip install --no-cache-dir .

# For HF Spaces: entrypoint.sh is not needed, app.py runs directly
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from sre_gym.env import SREGymEnv; print('OK')" || exit 1

# Default command: run HF Spaces app
CMD ["python", "app.py"]
