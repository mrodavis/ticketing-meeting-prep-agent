FROM python:3.13-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[postgres]" && useradd --create-home app && mkdir -p /app/.data && chown app /app/.data
USER app
EXPOSE 8000
# AWS credentials come from the task role and secrets from Secrets Manager at runtime;
# nothing is baked into the image.
CMD ["uvicorn", "meeting_prep.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
