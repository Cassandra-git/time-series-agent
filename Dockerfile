FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installer uv
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

# Copier les fichiers de configuration
COPY pyproject.toml uv.lock .python-version ./

# Copier le code source avant l'installation
COPY src/ ./src/
COPY main.py ./

# Installer les dépendances et le projet
RUN uv sync --locked

# Utiliser l'environnement virtuel
ENV PATH="/app/.venv/bin:$PATH"

CMD ["uv", "run", "python", "main.py"]
