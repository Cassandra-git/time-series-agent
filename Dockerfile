
FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e .

COPY main.py ./

CMD ["python", "main.py"]
