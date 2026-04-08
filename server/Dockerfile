FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY sre_gym/ ./sre_gym/
COPY server/ ./server/
COPY pyproject.toml .

# Install package
RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run server
CMD ["server"]
