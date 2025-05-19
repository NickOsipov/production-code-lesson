FROM python:3.11.4-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y curl wget

# Install Python dependencies with uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh

RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

COPY pyproject.toml .
COPY uv.lock .

RUN uv sync --no-install-project

ENV PATH="/app/.venv/bin/:$PATH"

COPY . .

ENV PYTHONPATH=/app
ENV FLASK_APP=/app/src/app.py

RUN chmod +x /app/scripts/entrypoint.sh

CMD ["bash", "/app/scripts/entrypoint.sh"]