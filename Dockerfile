FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

RUN groupadd --system appuser \
    && useradd --system --gid appuser --home-dir /app --shell /usr/sbin/nologin appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser app.py baseline.py client.py data.py environment.py gym_wrapper.py inference.py models.py rubrics.py tasks.py test_api.py test_env.py ui.py validate.py openenv.yaml pyproject.toml ./
COPY --chown=appuser:appuser evaluation ./evaluation
COPY --chown=appuser:appuser project_skills ./project_skills

RUN python -m pytest -q

USER appuser

# Hugging Face Spaces expects port 7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD ["python", "-c", "import requests; r=requests.get('http://localhost:7860/health', timeout=3); assert r.status_code==200"]

CMD ["python", "app.py"]
